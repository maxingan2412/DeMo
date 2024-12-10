import torch.nn as nn
import torch


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class simpleNet(nn.Module):
    def __init__(self, input_dim):
        super(simpleNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class Expert(nn.Module):
    def __init__(self, input_dim):
        super(Expert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class ExpertHead(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(ExpertHead, self).__init__()
        self.expertHead = nn.ModuleList([Expert(input_dim) for _ in range(num_experts)])

    def forward(self, x_chunk, gate_head):
        expert_outputs = [expert(x_chunk[i]) for i, expert in enumerate(self.expertHead)]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        expert_outputs = expert_outputs * gate_head.squeeze(1).unsqueeze(2)
        return expert_outputs


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.linear_re = nn.Sequential(nn.Linear(7 * dim, dim), QuickGELU(), nn.BatchNorm1d(dim))
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q_ = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_ = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, x, y):
        B, N, C = y.shape
        x = self.linear_re(x)
        q = self.q_(x).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        gates = attn.softmax(dim=-1)
        return gates

    def forward_(self, x):
        x = self.direct_gate(x)
        return x.unsqueeze(1)


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, head):
        super(GatingNetwork, self).__init__()
        self.gate = CrossAttention(input_dim, head)

    def forward(self, x, y):
        gates = self.gate(x, y)
        return gates


class MoM(nn.Module):
    def __init__(self, input_dim, num_experts, head):
        super(MoM, self).__init__()
        self.head_dim = input_dim // head
        self.head = head
        self.experts = nn.ModuleList(
            [ExpertHead(self.head_dim, num_experts) for _ in range(head)])
        self.gating_network = GatingNetwork(input_dim, head)

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
            x3 = x3.unsqueeze(0)
            x4 = x4.unsqueeze(0)
            x5 = x5.unsqueeze(0)
            x6 = x6.unsqueeze(0)
            x7 = x7.unsqueeze(0)

        x1_chunk = torch.chunk(x1, self.head, dim=-1)
        x2_chunk = torch.chunk(x2, self.head, dim=-1)
        x3_chunk = torch.chunk(x3, self.head, dim=-1)
        x4_chunk = torch.chunk(x4, self.head, dim=-1)
        x5_chunk = torch.chunk(x5, self.head, dim=-1)
        x6_chunk = torch.chunk(x6, self.head, dim=-1)
        x7_chunk = torch.chunk(x7, self.head, dim=-1)
        head_input = [[x1_chunk[i], x2_chunk[i], x3_chunk[i], x4_chunk[i], x5_chunk[i], x6_chunk[i], x7_chunk[i]] for i
                      in range(self.head)]
        query = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=-1)
        key = torch.stack([x1, x2, x3, x4, x5, x6, x7], dim=1)
        gate_heads = self.gating_network(query, key)
        expert_outputs = [expert(head_input[i], gate_heads[:, i]) for i, expert in enumerate(self.experts)]
        outputs = torch.cat(expert_outputs, dim=-1).flatten(start_dim=1, end_dim=-1)
        loss = 0
        if self.training:
            return outputs, loss
        return outputs


class GeneralFusion(nn.Module):
    def __init__(self, feat_dim, num_experts, head, reg_weight=0.1, dropout=0.1, cfg=None):
        super(GeneralFusion, self).__init__()
        self.reg_weight = reg_weight
        self.feat_dim = feat_dim

        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM
        if self.HDM:
            self.dropout = dropout
            scale = self.feat_dim ** -0.5
            self.r_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.n_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.t_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.rn_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.rt_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.nt_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            self.rnt_token = nn.Parameter(scale * torch.randn(1, 1, self.feat_dim))
            head_num_attn = self.feat_dim // 64
            self.r = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.n = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.t = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.rn = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.rt = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.nt = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
            self.rnt = nn.MultiheadAttention(embed_dim=self.feat_dim, num_heads=head_num_attn, dropout=self.dropout)
        if self.ATM:
            self.moe = MoM(input_dim=self.feat_dim, num_experts=num_experts, head=head)

    def forward_HDM(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input
        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)
        # get the embedding
        RGB = torch.cat([r_global, RGB_cash], dim=0)
        NI = torch.cat([n_global, NI_cash], dim=0)
        TI = torch.cat([t_global, TI_cash], dim=0)
        RGB_NI = torch.cat([RGB, NI], dim=0)
        RGB_TI = torch.cat([RGB, TI], dim=0)
        NI_TI = torch.cat([NI, TI], dim=0)
        RGB_NI_TI = torch.cat([RGB, NI, TI], dim=0)
        batch = RGB.size(1)
        # get the learnable token
        r_embedding = self.r_token.repeat(1, batch, 1)
        n_embedding = self.n_token.repeat(1, batch, 1)
        t_embedding = self.t_token.repeat(1, batch, 1)
        rn_embedding = self.rn_token.repeat(1, batch, 1)
        rt_embedding = self.rt_token.repeat(1, batch, 1)
        nt_embedding = self.nt_token.repeat(1, batch, 1)
        rnt_embedding = self.rnt_token.repeat(1, batch, 1)

        # for single modality
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze()
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared

    def forward_ATM(self, RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared):
        if self.training:
            moe_feat, loss_reg = self.moe(RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared,
                                          RNT_shared)
            return moe_feat, self.reg_weight * loss_reg
        else:
            moe_feat = self.moe(RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared)
            return moe_feat

    def forward(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDM(
            RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        if self.training:
            if self.HDM and not self.ATM:
                moe_feat = torch.cat(
                    [RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared], dim=-1)
                return moe_feat, 0
            elif self.HDM and self.ATM:
                moe_feat, loss_reg = self.forward_ATM(RGB_special, NI_special, TI_special, RN_shared, RT_shared,
                                                      NT_shared, RNT_shared)
                return moe_feat, loss_reg
        else:
            if self.HDM and not self.ATM:
                moe_feat = torch.cat(
                    [RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared], dim=-1)
                if moe_feat.dim() == 1:
                    moe_feat = moe_feat.unsqueeze(0)
                return moe_feat
            elif self.HDM and self.ATM:
                moe_feat = self.forward_ATM(RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared,
                                            RNT_shared)
                return moe_feat
