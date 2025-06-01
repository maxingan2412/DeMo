import torch.nn as nn
import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from modeling.backbones.vit_pytorch import trunc_normal_

import logging

import torch.utils.checkpoint as cp
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import textwrap



# logger = logging.getLogger(__name__)

T_MAX = 256
HEAD_SIZE = 64
userwkvblock = False
if userwkvblock:
    from torch.utils.cpp_extension import load
    from mmcv.runner.base_module import BaseModule, ModuleList

    cur_dir = os.path.dirname(os.path.abspath(__file__))  # 当前是 backbones/
    cuda_dir = os.path.join(cur_dir, "cuda_v6")
    wkv6_cuda = load(name="wkv6",
                     sources=[
                         os.path.join(cuda_dir, "wkv6_op.cpp"),
                         os.path.join(cuda_dir, "wkv6_cuda.cu"),
                     ],
                     verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math",
                     "-O3", "-Xptxas -O3",
                     "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}",
                     f"-D_T_={T_MAX}"])
#
    class WKV_6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ew = (-torch.exp(w.float())).contiguous()
                ctx.save_for_backward(r, k, v, ew, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6_cuda.forward(B, T, C, H, r, k, v, ew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.float32, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                wkv6_cuda.backward(B, T, C, H, r, k, v, ew, u, gy, gr, gk, gv, gw, gu)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)
    def RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u):
        return WKV_6.apply(B, T, C, H, r, k, v, w, u)

    def q_shift_multihead(input, shift_pixel=1, head_dim=HEAD_SIZE,
                          patch_resolution=None, with_cls_token=False):
        B, N, C = input.shape
        assert C % head_dim == 0
        assert head_dim % 4 == 0
        if with_cls_token:
            cls_tokens = input[:, [-1], :]
            input = input[:, :-1, :]
        input = input.transpose(1, 2).reshape(
            B, -1, head_dim, patch_resolution[0], patch_resolution[1])  # [B, n_head, head_dim H, W]
        B, _, _, H, W = input.shape
        output = torch.zeros_like(input)
        output[:, :, 0:int(head_dim*1/4), :, shift_pixel:W] = \
            input[:, :, 0:int(head_dim*1/4), :, 0:W-shift_pixel]
        output[:, :, int(head_dim/4):int(head_dim/2), :, 0:W-shift_pixel] = \
            input[:, :, int(head_dim/4):int(head_dim/2), :, shift_pixel:W]
        output[:, :, int(head_dim/2):int(head_dim/4*3), shift_pixel:H, :] = \
            input[:, :, int(head_dim/2):int(head_dim/4*3), 0:H-shift_pixel, :]
        output[:, :, int(head_dim*3/4):int(head_dim), 0:H-shift_pixel, :] = \
            input[:, :, int(head_dim*3/4):int(head_dim), shift_pixel:H, :]
        if with_cls_token:
            output = output.reshape(B, C, N-1).transpose(1, 2)
            output = torch.cat((output, cls_tokens), dim=1)
        else:
            output = output.reshape(B, C, N).transpose(1, 2)
        return output
    class VRWKV_SpatialMix_V6(BaseModule):
        def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                     shift_pixel=1, init_mode='fancy', key_norm=False, with_cls_token=False,
                     with_cp=False):
            super().__init__()
            self.layer_id = layer_id
            self.n_layer = n_layer
            self.n_embd = n_embd
            self.attn_sz = n_embd

            self.n_head = n_head
            self.head_size = self.attn_sz // self.n_head
            assert self.head_size == HEAD_SIZE
            self.device = None
            self._init_weights(init_mode)
            self.with_cls_token = with_cls_token
            self.shift_pixel = shift_pixel
            self.shift_mode = shift_mode


            self.shift_func = eval(shift_mode)

            self.key = nn.Linear(self.n_embd, self.attn_sz, bias=False)
            self.value = nn.Linear(self.n_embd, self.attn_sz, bias=False)
            self.receptance = nn.Linear(self.n_embd, self.attn_sz, bias=False)
            self.gate = nn.Linear(self.n_embd, self.attn_sz, bias=False)
            if key_norm:
                self.key_norm = nn.LayerNorm(n_embd)
            else:
                self.key_norm = None
            self.output = nn.Linear(self.attn_sz, n_embd, bias=False)

            self.ln_x = nn.GroupNorm(self.n_head, self.attn_sz, eps=1e-5)
            self.with_cp = with_cp

        def _init_weights(self, init_mode):
            if init_mode == 'fancy':
                with torch.no_grad():
                    ratio_0_to_1 = self.layer_id / (self.n_layer - 1)  # 0 to 1
                    ratio_1_to_almost0 = 1.0 - (self.layer_id / self.n_layer)  # 1 to ~0
                    ddd = torch.ones(1, 1, self.n_embd)
                    for i in range(self.n_embd):
                        ddd[0, 0, i] = i / self.n_embd

                    # fancy time_mix
                    self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                    self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                    self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
                    self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
                    self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                    self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

                    TIME_MIX_EXTRA_DIM = 32  # generate TIME_MIX for w,k,v,r,g
                    self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_MIX_EXTRA_DIM * 5).uniform_(-1e-4, 1e-4))
                    self.time_maa_w2 = nn.Parameter(torch.zeros(5, TIME_MIX_EXTRA_DIM, self.n_embd).uniform_(-1e-4, 1e-4))

                    # fancy time_decay
                    decay_speed = torch.ones(self.attn_sz)
                    for n in range(self.attn_sz):
                        decay_speed[n] = -6 + 5 * (n / (self.attn_sz - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
                    self.time_decay = nn.Parameter(decay_speed.reshape(1, 1, self.attn_sz))

                    TIME_DECAY_EXTRA_DIM = 64
                    self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, TIME_DECAY_EXTRA_DIM).uniform_(-1e-4, 1e-4))
                    self.time_decay_w2 = nn.Parameter(torch.zeros(TIME_DECAY_EXTRA_DIM, self.attn_sz).uniform_(-1e-4, 1e-4))

                    tmp = torch.zeros(self.attn_sz)
                    for n in range(self.attn_sz):
                        zigzag = ((n + 1) % 3 - 1) * 0.1
                        tmp[n] = ratio_0_to_1 * (1 - (n / (self.attn_sz - 1))) + zigzag

                    self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
            else:
                raise NotImplementedError

        def jit_func(self, x, patch_resolution):
            # Mix x with the previous timestep to produce xk, xv, xr
            B, T, C = x.size()

            xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
                                 with_cls_token=self.with_cls_token) - x  # shiftq - x
            xxx = x + xx * self.time_maa_x  # [B, T, C]
            xxx = torch.tanh(xxx @ self.time_maa_w1).view(B * T, 5, -1).transpose(0, 1)
            # [5, B*T, TIME_MIX_EXTRA_DIM]
            xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
            # [5, B, T, C]
            mw, mk, mv, mr, mg = xxx.unbind(dim=0)

            xw = x + xx * (self.time_maa_w + mw)
            xk = x + xx * (self.time_maa_k + mk)
            xv = x + xx * (self.time_maa_v + mv)
            xr = x + xx * (self.time_maa_r + mr)
            xg = x + xx * (self.time_maa_g + mg)

            r = self.receptance(xr)
            k = self.key(xk)
            v = self.value(xv)
            g = F.silu(self.gate(xg))

            ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
            # [B, T, C]
            w = self.time_decay + ww

            return r, k, v, g, w

        def jit_func_2(self, x, g):
            B, T, C = x.size()
            x = x.view(B * T, C)

            x = self.ln_x(x).view(B, T, C)
            x = self.output(x * g)
            return x

        def forward(self, x, patch_resolution=None):
            def _inner_forward(x):
                B, T, C = x.size()
                self.device = x.device

                r, k, v, g, w = self.jit_func(x, patch_resolution)
                #x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r, k, v, w, u=self.time_faaaa)
                x = RUN_CUDA_RWKV6(B, T, C, self.n_head, r.float(), k.float(), v.float(), w.float(), u=self.time_faaaa.float())

                if self.key_norm is not None:
                    x = self.key_norm(x)
                return self.jit_func_2(x, g)

            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = _inner_forward(x)
            return x


    class VRWKV_ChannelMix(BaseModule):
        def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                     shift_pixel=1, hidden_rate=4, init_mode='fancy', key_norm=False,
                     with_cls_token=False, with_cp=False):
            super().__init__()
            self.layer_id = layer_id
            self.n_layer = n_layer
            self.n_embd = n_embd
            self.attn_sz = n_embd
            self.n_head = n_head
            self.head_size = self.attn_sz // self.n_head
            assert self.head_size == HEAD_SIZE
            self.with_cp = with_cp
            self._init_weights(init_mode)
            self.with_cls_token = with_cls_token
            self.shift_pixel = shift_pixel
            self.shift_mode = shift_mode
            self.shift_func = eval(shift_mode)

            hidden_sz = hidden_rate * n_embd
            self.key = nn.Linear(n_embd, hidden_sz, bias=False)
            if key_norm:
                self.key_norm = nn.LayerNorm(hidden_sz)
            else:
                self.key_norm = None
            self.receptance = nn.Linear(n_embd, n_embd, bias=False)
            self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        def _init_weights(self, init_mode):
            if init_mode == 'fancy':
                with torch.no_grad():  # fancy init of time_mix
                    ratio_1_to_almost0 = (1.0 - (self.layer_id / self.n_layer))  # 1 to ~0
                    x = torch.ones(1, 1, self.n_embd)
                    for i in range(self.n_embd):
                        x[0, 0, i] = i / self.n_embd
                    self.spatial_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
                    self.spatial_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            else:
                raise NotImplementedError

        def forward(self, x, patch_resolution=None):
            def _inner_forward(x):
                xx = self.shift_func(x, self.shift_pixel, patch_resolution=patch_resolution,
                                     with_cls_token=self.with_cls_token)
                xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
                xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)

                k = self.key(xk)
                k = torch.square(torch.relu(k))
                if self.key_norm is not None:
                    k = self.key_norm(k)
                kv = self.value(k)
                x = torch.sigmoid(self.receptance(xr)) * kv
                return x

            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = _inner_forward(x)
            return x


    class Block(BaseModule):
        def __init__(self, n_embd, n_head, n_layer, layer_id, shift_mode='q_shift_multihead',
                     shift_pixel=1, drop_path=0., hidden_rate=4, init_mode='fancy',
                     init_values=None, post_norm=False, key_norm=False, with_cls_token=False,
                     with_cp=False):
            super().__init__()
            self.layer_id = layer_id
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
            #self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.drop_path = nn.Identity()
            if self.layer_id == 0:
                self.ln0 = nn.LayerNorm(n_embd)

            self.att = VRWKV_SpatialMix_V6(n_embd, n_head, n_layer, layer_id, shift_mode,
                                           shift_pixel, init_mode, key_norm=key_norm,
                                           with_cls_token=with_cls_token)

            self.ffn = VRWKV_ChannelMix(n_embd, n_head, n_layer, layer_id, shift_mode,
                                        shift_pixel, hidden_rate, init_mode, key_norm=key_norm,
                                        with_cls_token=with_cls_token)
            self.layer_scale = (init_values is not None)
            self.post_norm = post_norm
            if self.layer_scale:
                self.gamma1 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
                self.gamma2 = nn.Parameter(init_values * torch.ones((n_embd)), requires_grad=True)
            self.with_cp = with_cp

        def forward(self, x, patch_resolution=None):
            def _inner_forward(x):
                if self.layer_id == 0:
                    x = self.ln0(x)
                if self.post_norm:
                    if self.layer_scale:
                        x = x + self.drop_path(self.gamma1 * self.ln1(self.att(x, patch_resolution)))
                        x = x + self.drop_path(self.gamma2 * self.ln2(self.ffn(x, patch_resolution)))
                    else:
                        x = x + self.drop_path(self.ln1(self.att(x, patch_resolution)))
                        x = x + self.drop_path(self.ln2(self.ffn(x, patch_resolution)))
                else:
                    if self.layer_scale:
                        x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), patch_resolution))
                        x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), patch_resolution))
                    else:
                        x = x + self.drop_path(self.att(self.ln1(x), patch_resolution))
                        x = x + self.drop_path(self.ffn(self.ln2(x), patch_resolution))
                return x

            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(_inner_forward, x)
            else:
                x = _inner_forward(x)
            return x




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
        expert_outputs = [expert(x_chunk[i]) for i, expert in enumerate(self.expertHead)] #这里相对于把 x_chunk[i] 都做了个MLP
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
        gates = attn.softmax(dim=-1) #其实就是拿attn做门控gates
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


# class MoM(nn.Module):
#     def __init__(self, input_dim, num_experts, head):
#         super(MoM, self).__init__()
#         self.head_dim = input_dim // head
#         self.head = head
#         self.experts = nn.ModuleList(
#             [ExpertHead(self.head_dim, num_experts) for _ in range(head)])
#         self.gating_network = GatingNetwork(input_dim, head)
#
#     def forward(self, x1, x2, x3, x4, x5, x6, x7):
#         if x1.dim() == 1:
#             x1 = x1.unsqueeze(0)
#             x2 = x2.unsqueeze(0)
#             x3 = x3.unsqueeze(0)
#             x4 = x4.unsqueeze(0)
#             x5 = x5.unsqueeze(0)
#             x6 = x6.unsqueeze(0)
#             x7 = x7.unsqueeze(0)
#
#         x1_chunk = torch.chunk(x1, self.head, dim=-1)
#         x2_chunk = torch.chunk(x2, self.head, dim=-1)
#         x3_chunk = torch.chunk(x3, self.head, dim=-1)
#         x4_chunk = torch.chunk(x4, self.head, dim=-1)
#         x5_chunk = torch.chunk(x5, self.head, dim=-1)
#         x6_chunk = torch.chunk(x6, self.head, dim=-1)
#         x7_chunk = torch.chunk(x7, self.head, dim=-1)
#         head_input = [[x1_chunk[i], x2_chunk[i], x3_chunk[i], x4_chunk[i], x5_chunk[i], x6_chunk[i], x7_chunk[i]] for i
#                       in range(self.head)] #按head 分块，每个head 有7个输入
#         query = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=-1)
#         key = torch.stack([x1, x2, x3, x4, x5, x6, x7], dim=1)
#         gate_heads = self.gating_network(query, key) # Multi-HeadAttentionGating 似乎思这个，就是拿 query 和key 生成的attn作为门控
#         expert_outputs = [expert(head_input[i], gate_heads[:, i]) for i, expert in enumerate(self.experts)]
#         outputs = torch.cat(expert_outputs, dim=-1).flatten(start_dim=1, end_dim=-1)
#         loss = 0
#         if self.training:
#             return outputs, loss
#         return outputs



import torch
import torch.nn as nn
import torch.nn.functional as F

class MoM(nn.Module):
    def __init__(self, input_dim, num_experts, head,
                 lb_weight=0.1, div_weight=0.1, margin_weight=0.1, margin=0.3):
        super(MoM, self).__init__()
        self.head_dim = input_dim // head
        self.head = head
        self.num_experts = num_experts
        self.lb_weight = lb_weight
        self.div_weight = div_weight
        self.margin_weight = margin_weight
        self.margin = margin

        self.experts = nn.ModuleList(
            [ExpertHead(self.head_dim, num_experts) for _ in range(head)])
        self.gating_network = GatingNetwork(input_dim, head)

    def compute_uniform_kl_loss(self, gate_weights):
        """
        KL divergence between average gate distribution and uniform prior.
        gate_weights: [B, H, E]
        """
        avg = gate_weights.mean(dim=(0, 1))  # [E]
        uniform = torch.full_like(avg, 1.0 / avg.numel())
        return F.kl_div(avg.log(), uniform, reduction='batchmean')

    def compute_js_divergence(self, gate_weights):
        """
        Jensen-Shannon divergence across heads to encourage diversity.
        gate_weights: [B, H, E]
        """
        B, H, E = gate_weights.shape
        # pairwise distributions p1, p2: [B, H, H, E]
        p1 = gate_weights.unsqueeze(2).expand(-1, -1, H, -1)
        p2 = gate_weights.unsqueeze(1).expand(-1, H, -1, -1)
        m = 0.5 * (p1 + p2)
        # compute KL divergences, sum over experts dim
        kl1 = F.kl_div(p1.log(), m, reduction='none').sum(-1)  # [B,H,H]
        kl2 = F.kl_div(p2.log(), m, reduction='none').sum(-1)  # [B,H,H]
        js = 0.5 * (kl1 + kl2)  # [B,H,H]
        # mask out self-pairs
        mask = (1 - torch.eye(H, device=gate_weights.device)).unsqueeze(0)  # [1,H,H]
        js = js * mask  # [B,H,H]
        # average over batch and head pairs
        return js.sum() / (B * H * (H - 1))

    def compute_margin_loss(self, gate_weights):
        """
        Margin-based loss between top-2 experts per head.
        gate_weights: [B, H, E]
        """
        top2 = torch.topk(gate_weights, k=2, dim=-1).values  # [B, H, 2]
        margin_diff = top2[..., 0] - top2[..., 1]
        return F.relu(self.margin - margin_diff).mean()

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        if x1.dim() == 1:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
            x3 = x3.unsqueeze(0)
            x4 = x4.unsqueeze(0)
            x5 = x5.unsqueeze(0)
            x6 = x6.unsqueeze(0)
            x7 = x7.unsqueeze(0)

        # split into heads
        chunks = [torch.chunk(x, self.head, dim=-1)
                  for x in (x1, x2, x3, x4, x5, x6, x7)]
        head_inputs = [[chunks[v][i] for v in range(7)]
                        for i in range(self.head)]

        # build query and key
        query = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=-1)
        key   = torch.stack([x1, x2, x3, x4, x5, x6, x7], dim=1)

        # gating
        gate_heads = self.gating_network(query, key)  # [B, H, 1, E]
        all_gw = []
        outputs = []
        for i, expert in enumerate(self.experts):
            gh = gate_heads[:, i]
            out = expert(head_inputs[i], gh)
            outputs.append(out)
            # collect gate weights [B, E]
            all_gw.append(gh.squeeze(1))

        # combine expert outputs
        features = torch.cat(outputs, dim=-1).flatten(start_dim=1)

        if not self.training:
            return features

        # compute auxiliary losses
        gate_weights = torch.stack(all_gw, dim=1)  # [B, H, E]
        lb_loss  = self.compute_uniform_kl_loss(gate_weights)
        js_loss  = self.compute_js_divergence(gate_weights)
        m_loss   = self.compute_margin_loss(gate_weights)

        loss = (self.lb_weight   * lb_loss
              + self.div_weight  * js_loss
              + self.margin_weight * m_loss)

        return features, loss


##
class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, ksize, share
    ):

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor

        self.ksize = ksize
        self.stride = stride
        kk = self.ksize
        pad_size = 0
        self.share_offset = share
        if self.share_offset:
            self.conv_offset = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            )
        else:
            self.conv_offset_r = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_n = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_t = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def _get_ref_points(self, H_in, W_in, B, kernel_size, stride, dtype, device):
        """
        生成参考点在每个卷积块的中心位置。

        :param H_in: 输入特征图的高度 (如 16)
        :param W_in: 输入特征图的宽度 (如 8)
        :param B: 批次大小
        :param kernel_size: 卷积核大小
        :param stride: 卷积步幅
        :param dtype: 数据类型
        :param device: 设备类型
        :return: 参考点张量，形状为 (B * n_groups, H_out, W_out, 2)
        """

        # 计算输出特征图的高度和宽度
        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1

        # 计算每个卷积位置的中心点在原图坐标上的位置
        center_y = torch.arange(H_out, dtype=dtype, device=device) * stride + (kernel_size // 2)
        center_x = torch.arange(W_out, dtype=dtype, device=device) * stride + (kernel_size // 2)

        # 生成网格
        ref_y, ref_x = torch.meshgrid(center_y, center_x, indexing='ij')
        ref = torch.stack((ref_y, ref_x), dim=-1)  # Shape: (H_out, W_out, 2)

        # 归一化到 [-1, 1]
        ref[..., 1].div_(W_in - 1.0).mul_(2.0).sub_(1.0)  # x 坐标归一化
        ref[..., 0].div_(H_in - 1.0).mul_(2.0).sub_(1.0)  # y 坐标归一化

        # 扩展批次和组维度
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # Shape: (B * n_groups, H_out, W_out, 2)

        return ref

    @torch.no_grad()
    def visualize_sampling_with_offset(self, feature_maps, sampled_pointss, img_paths, reference_pointss, writer=None,
                                       epoch=0, title='Sampling Points with Offset', pattern=0, patch_size=(16, 16)):
        """
        在原始图像上标记采样点的偏移效果并保存
        :param feature_map: 原始特征图，形状为 (C, H, W)
        :param sampled_points: 目标采样点，形状为 (N, 2)，其中 N 是采样点的数量，2 对应于 (y, x)
        :param img_path: 原始图像路径
        :param reference_points: 参考采样点，用于标记偏移起始位置，形状为 (N, 2)
        :param writer: 用于保存图像的writer (比如Tensorboard的SummaryWriter)
        :param epoch: 当前epoch，用于保存不同epoch的结果
        :param title: 图形标题
        :param patch_size: 每个 patch 的尺寸 (height, width)
        """
        # 根据模式设置图像路径前缀
        modality = ['RGB', 'NI', 'TI']
        if pattern == 0:
            prefix = '../RGBNT201/test/RGB/'
        elif pattern == 1:
            prefix = '../RGBNT201/test/NI/'
        elif pattern == 2:
            prefix = '../RGBNT201/test/TI/'
        for i in range(len(img_paths)):
            img_path = prefix + img_paths[i]

            # 加载并调整图像大小
            original_image = Image.open(img_path).resize((128, 256))
            original_image = np.array(original_image)

            # 处理特征图
            feature_map = torch.mean(feature_maps[i], dim=0, keepdim=True)
            feature_map = feature_map.detach().cpu().numpy()
            feature_map = (feature_map - np.min(feature_map)) / np.ptp(feature_map)

            # 转换采样点和参考点为 numpy 格式
            sampled_points = sampled_pointss[i].detach().cpu().numpy()
            reference_points = reference_pointss[i].detach().cpu().numpy()

            # 获取特征图和原始图像的尺寸
            H_feat, W_feat = feature_map.shape[1:]
            H_orig, W_orig = original_image.shape[:2]

            # 计算下采样比例
            scale_x = W_orig / W_feat
            scale_y = H_orig / H_feat

            # 转换坐标到原图系
            sampled_points[:, 1] = (sampled_points[:, 1] + 1) / 2 * (W_feat - 1) * scale_x
            sampled_points[:, 0] = (sampled_points[:, 0] + 1) / 2 * (H_feat - 1) * scale_y
            reference_points[:, 1] = (reference_points[:, 1] + 1) / 2 * (W_feat - 1) * scale_x
            reference_points[:, 0] = (reference_points[:, 0] + 1) / 2 * (H_feat - 1) * scale_y

            # 绘制原图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(original_image, aspect='auto')
            ax.set_title(title)

            # 更改采样点样式和箭头样式
            for ref, samp in zip(reference_points, sampled_points):
                ref_y, ref_x = ref
                samp_y, samp_x = samp

                # 参考点用淡蓝色
                ax.scatter(ref_x, ref_y, c='skyblue', s=70, marker='o', edgecolor='black', linewidth=2)  # 较大、黑色边框
                # 目标点用橙色
                ax.scatter(samp_x, samp_y, c='orange', s=70, marker='x', linewidth=12)  # 橙色 "x" 标记

                # 箭头颜色为半透明绿色
                ax.arrow(ref_x, ref_y, samp_x - ref_x, samp_y - ref_y, color='limegreen', alpha=0.7,
                         head_width=4, head_length=6, linewidth=6, length_includes_head=True)

            # 绘制 patch 分隔线
            patch_height, patch_width = patch_size
            for y in range(0, H_orig, patch_height):
                ax.plot([0, W_orig], [y, y], color='white', linewidth=1.5, linestyle='--')  # 水平线
            for x in range(0, W_orig, patch_width):
                ax.plot([x, x], [0, H_orig], color='white', linewidth=1.5, linestyle='--')  # 垂直线

            ax.set_xlim(-1, W_orig)
            ax.set_ylim(H_orig, -1)  # y 轴反转

            # 保存到 writer
            if writer is not None:
                writer.add_figure(f"{title}", fig, global_step=epoch)
            plt.savefig(
                f'../off_vis/{modality[pattern]}/{img_path.split("/")[-1].split(".")[0]}.png')
            # plt.show()
            plt.close(fig)

    def show_cam_on_image(self, img: np.ndarray,
                          mask: np.ndarray,
                          use_rgb: bool = False,
                          colormap: int = cv2.COLORMAP_HOT,
                          image_weight: float = 0.3) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
        :returns: The default image with the cam overlay.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        if np.max(img) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")

        if image_weight < 0 or image_weight > 1:
            raise Exception(
                f"image_weight should be in the range [0, 1].\
                    Got: {image_weight}")

        cam = (1 - image_weight) * heatmap + image_weight * img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    @torch.no_grad()
    def visualize_attention_with_distribution(self, attn_map, img_paths, index, reference_pointss, sampled_pointss,
                                              writer=None, epoch=0, title='Attention Heatmap with Distribution',
                                              patch_size=(16, 16), text=''):
        """
        在原始图像上根据 attn_map 选择一个注意力分布，生成热图并覆盖，并显示描述文本
        """
        modality = ['v_RGB', 'v_NIR', 'v_TIR', 't_RGB', 't_NIR', 't_TIR']
        if index == 0 or index == 3:
            prefix = '../RGBNT201/test/RGB/'
            text = text['rgb_text']
        elif index == 1 or index == 4:
            prefix = '../RGBNT201/test/NI/'
            text = text['ni_text']
        elif index == 2 or index == 5:
            prefix = '../RGBNT201/test/TI/'
            text = text['ti_text']

        grid_height = 7  # 高度方向的块数
        grid_width = 3  # 宽度方向的块数

        for i in range(len(img_paths)):
            img_path = prefix + img_paths[i]
            original_image = Image.open(img_path).convert('RGB').resize((128, 256))
            original_image = np.float32(original_image) / 255  # 将图像转换为 [0, 1] 范围的浮点数

            H_orig, W_orig = original_image.shape[:2]  # 获取原图的高度和宽度
            grid_height_size = H_orig // grid_height  # 计算每个网格的高度
            grid_width_size = W_orig // grid_width  # 计算每个网格的宽度

            # 根据 index 选择对应的注意力区域
            if index == 0 or index == 3:
                selected_attn = attn_map[i, index, :21]
            elif index == 1 or index == 4:
                selected_attn = attn_map[i, index, 21:42]
            else:
                selected_attn = attn_map[i, index, 42:]

            selected_attn = selected_attn * 1000 * 2  # 放大权重
            selected_attn = torch.softmax(selected_attn, dim=0)  # 应用 softmax
            selected_attn = selected_attn.detach().cpu().numpy()  # 转换为 NumPy 数组

            # 初始化一个全为零的热图
            heatmap = np.zeros((H_orig, W_orig))

            # 根据网格和注意力权重构建热图
            for row in range(grid_height):
                for col in range(grid_width):
                    # 计算网格区域的起始和结束位置
                    y_start = row * grid_height_size
                    x_start = col * grid_width_size
                    y_end = (row + 1) * grid_height_size if row < grid_height - 1 else H_orig
                    x_end = (col + 1) * grid_width_size if col < grid_width - 1 else W_orig

                    # 获取当前网格的注意力权重
                    weight = selected_attn[row * grid_width + col]

                    # 将该权重值应用到对应的网格区域
                    heatmap[y_start:y_end, x_start:x_end] += weight

            # 将热图与原图叠加
            overlay = self.show_cam_on_image(original_image, heatmap, use_rgb=True, image_weight=0.5)

            # 创建可视化图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(overlay, aspect='auto')
            ax.set_title(f"{title} - Image {i + 1}")

            # 在图像中添加描述文本

            ax.text(
                0.5,  # X 位置 (0到1之间的比例坐标)
                -0.12,  # Y 位置，设置为负值将文本放置在图像下方
                textwrap.fill(text[i], width=80),  # 要显示的文本
                transform=ax.transAxes,  # 使用轴的坐标系统
                color='black',  # 字体颜色
                fontsize=10,  # 字体大小
                ha='center',  # 水平居中
                va='bottom',  # 垂直对齐方式，设置为'底部'对齐
                weight='bold'  # 字体加粗
            )

            # 设置图像的坐标轴
            ax.set_xlim(-1, W_orig)
            ax.set_ylim(H_orig, -1)

            # 如果提供了 writer，则保存到 TensorBoard
            if writer is not None:
                writer.add_figure(f"{title}", fig, global_step=epoch)

            # 保存结果图像
            output_path = f'../attn_vis/{modality[index]}/{img_path.split("/")[-1].split(".")[0]}.png'
            plt.savefig(output_path)
            # plt.show()  # 如果你需要在屏幕上显示图像
            plt.close(fig)

    def off_set_shared(self, data, reference):
        data = einops.rearrange(data, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=3 * self.n_group_channels)
        offset = self.conv_offset(data)
        Hk, Wk = offset.size(2), offset.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        pos_x = (offset + reference).clamp(-1., +1.)
        pos_y = (offset + reference).clamp(-1., +1.)
        pos_z = (offset + reference).clamp(-1., +1.)
        return pos_x, pos_y, pos_z, Hk, Wk

    def off_set_unshared(self, data, reference):
        x, y, z = data.chunk(3, dim=1)
        x = einops.rearrange(x, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        y = einops.rearrange(y, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        z = einops.rearrange(z, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset_r = self.conv_offset_r(x)
        offset_n = self.conv_offset_n(y)
        offset_t = self.conv_offset_t(z)
        Hk, Wk = offset_r.size(2), offset_r.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset_r = offset_r.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_n = offset_n.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_t = offset_t.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset_r = einops.rearrange(offset_r, 'b p h w -> b h w p')
        offset_n = einops.rearrange(offset_n, 'b p h w -> b h w p')
        offset_t = einops.rearrange(offset_t, 'b p h w -> b h w p')
        pos_x = (offset_r + reference).clamp(-1., +1.)
        pos_y = (offset_n + reference).clamp(-1., +1.)
        pos_z = (offset_t + reference).clamp(-1., +1.)
        return pos_x, pos_y, pos_z, Hk, Wk

    def forward(self, x, y, z, writer=None, epoch=None, img_path=None, text=''):
        B, C, H, W = x.size()
        #b_, c_, h_, w_ = query.size()
        dtype, device = x.dtype, x.device
        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device) #这里好像是论文中的local mixer
        if self.share_offset: # shared
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_shared(data, reference)
        else:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_unshared(data, reference)
        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)

        sampled_x, sampled_y, sampled_z = [
            t.reshape(B, C, 1, n_sample).squeeze(2).permute(2, 0, 1)
            for t in [sampled_x, sampled_y, sampled_z]
        ]

        # sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        #
        # q = self.proj_q(query)
        # q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        # k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        # v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        # attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        # attn = attn.mul(self.scale)
        # attn = F.softmax(attn, dim=2)
        #
        # attn = self.attn_drop(attn)
        # out = torch.einsum('b m n, b c n -> b c m', attn, v)
        # out = out.reshape(B, C, 1, h_ * w_)
        # out = self.proj_drop(self.proj_out(out))
        # out = query + out
        return sampled_x,sampled_y,sampled_z


    def forwardOld(self, query, x, y, z, writer=None, epoch=None, img_path=None, text=''):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        dtype, device = x.dtype, x.device
        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device) #这里好像是论文中的local mixer
        if self.share_offset:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_shared(data, reference)
        else:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_unshared(data, reference)
        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, 1, n_sample)
        sampled_y = sampled_y.reshape(B, C, 1, n_sample)
        sampled_z = sampled_z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)

        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)

        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def forward_woCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)

        if self.share_offset:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_shared(data, reference)
        else:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_unshared(data, reference)
        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, 1, n_sample)
        sampled_y = sampled_y.reshape(B, C, 1, n_sample)
        sampled_z = sampled_z.reshape(B, C, 1, n_sample)
        input = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(input)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        k = self.proj_k(input).reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        v = self.proj_v(input).reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, 3 * Hk * Wk)
        out = self.proj_drop(self.proj_out(out))
        out = input + out
        sampled_x, sampled_y, sampled_z = out.chunk(3, dim=-1)

        sampled_x = torch.mean(sampled_x, dim=-1, keepdim=True)
        sampled_y = torch.mean(sampled_y, dim=-1, keepdim=True)
        sampled_z = torch.mean(sampled_z, dim=-1, keepdim=True)

        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        sampled_2 = torch.cat([sampled, sampled], dim=-1)
        return sampled_2.squeeze(2)

    def forward_woSample_wCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        n_sample = H * W
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def forward_woSample_woCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        n_sample = H * W
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        input = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(input)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        k = self.proj_k(input).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(input).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, 3 * n_sample)
        out = self.proj_drop(self.proj_out(out))
        out = input + out
        sampled_x, sampled_y, sampled_z = out.chunk(3, dim=-1)

        sampled_x = torch.mean(sampled_x, dim=-1, keepdim=True)
        sampled_y = torch.mean(sampled_y, dim=-1, keepdim=True)
        sampled_z = torch.mean(sampled_z, dim=-1, keepdim=True)

        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        sampled_2 = torch.cat([sampled, sampled], dim=-1)
        return sampled_2.squeeze(2)

    def forward_woOffset(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        data = torch.cat([x, y, z], dim=1)
        x = self.conv_v(data)
        y = self.conv_n(data)
        z = self.conv_t(data)
        h_new, w_new = x.size(2), x.size(3)
        n_sample = h_new * w_new
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)


import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import textwrap


class DAttentionEnhanced(nn.Module):
    """
    Enhanced Deformable Attention with:
    1. Adaptive Modal Weighting (AMW)
    2. Multi-Scale Offset Fusion (MSOF)
    3. Learnable Temperature Scaling (LTS)
    4. Residual Offset Connection (ROC)
    """

    def __init__(
            self, q_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, ksize, share
    ):

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor

        self.ksize = ksize
        self.stride = stride
        kk = self.ksize
        pad_size = 0
        self.share_offset = share

        # ===== 创新点1: Adaptive Modal Weighting (AMW) =====
        self.modal_weights = nn.Parameter(torch.ones(3))  # 为RGB, NIR, TIR三个模态学习权重
        self.modal_gate = nn.Sequential(
            nn.Conv2d(3 * self.n_group_channels, self.n_group_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_group_channels // 4, 3, 1),
            nn.Sigmoid()
        )

        # ===== 创新点2: Multi-Scale Offset Fusion (MSOF) =====
        self.multi_scale_levels = 3
        self.scale_weights = nn.Parameter(torch.ones(self.multi_scale_levels))

        # ===== 创新点3: Learnable Temperature Scaling (LTS) =====
        self.temperature = nn.Parameter(torch.ones(1))

        # ===== 创新点4: Residual Offset Connection (ROC) =====
        self.offset_residual_weight = nn.Parameter(torch.tensor(0.1))

        if self.share_offset:
            # 主要偏移网络（原有的）
            self.conv_offset = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            )

            # 多尺度偏移网络
            self.conv_offset_coarse = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk + 2, stride, pad_size + 1,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            )

            self.conv_offset_fine = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, max(kk - 2, 1), stride, 0,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            )
        else:
            # 原有的非共享偏移网络
            self.conv_offset_r = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_n = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_t = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )

            # 多尺度版本 (简化实现)
            self.conv_offset_r_coarse = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk + 2, stride, pad_size + 1,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            # 为简化起见，只为R通道添加多尺度，实际使用中可以为所有通道添加

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)  # 需要导入trunc_normal_
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def _get_ref_points(self, H_in, W_in, B, kernel_size, stride, dtype, device):
        """
        生成参考点在每个卷积块的中心位置。

        :param H_in: 输入特征图的高度 (如 16)
        :param W_in: 输入特征图的宽度 (如 8)
        :param B: 批次大小
        :param kernel_size: 卷积核大小
        :param stride: 卷积步幅
        :param dtype: 数据类型
        :param device: 设备类型
        :return: 参考点张量，形状为 (B * n_groups, H_out, W_out, 2)
        """

        # 计算输出特征图的高度和宽度
        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1

        # 计算每个卷积位置的中心点在原图坐标上的位置
        center_y = torch.arange(H_out, dtype=dtype, device=device) * stride + (kernel_size // 2)
        center_x = torch.arange(W_out, dtype=dtype, device=device) * stride + (kernel_size // 2)

        # 生成网格
        ref_y, ref_x = torch.meshgrid(center_y, center_x, indexing='ij')
        ref = torch.stack((ref_y, ref_x), dim=-1)  # Shape: (H_out, W_out, 2)

        # 归一化到 [-1, 1]
        ref[..., 1].div_(W_in - 1.0).mul_(2.0).sub_(1.0)  # x 坐标归一化
        ref[..., 0].div_(H_in - 1.0).mul_(2.0).sub_(1.0)  # y 坐标归一化

        # 扩展批次和组维度
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # Shape: (B * n_groups, H_out, W_out, 2)

        return ref

    def adaptive_modal_weighting(self, x, y, z):
        """
        创新点1: 自适应模态权重
        """
        # 计算全局模态权重
        modal_weights = F.softmax(self.modal_weights, dim=0)

        # 计算局部门控权重
        concat_features = torch.cat([x, y, z], dim=1)
        B, _, H, W = concat_features.shape
        avg_pool = F.adaptive_avg_pool2d(concat_features, 1)
        gate_weights = self.modal_gate(avg_pool)  # [B, 3, 1, 1]

        # 结合全局和局部权重
        combined_weights = modal_weights.view(1, 3, 1, 1) * gate_weights
        combined_weights = F.softmax(combined_weights, dim=1)

        # 应用权重
        weighted_x = x * combined_weights[:, 0:1]
        weighted_y = y * combined_weights[:, 1:2]
        weighted_z = z * combined_weights[:, 2:3]

        return weighted_x, weighted_y, weighted_z

    def multi_scale_offset_fusion(self, data, reference):
        """
        创新点2: 多尺度偏移融合
        """
        if not self.share_offset:
            # 对于非共享情况，使用增强版
            return self.off_set_unshared_enhanced(data, reference)

        data = einops.rearrange(data, 'b (g c) h w -> (b g) c h w',
                                g=self.n_groups, c=3 * self.n_group_channels)

        # 计算不同尺度的偏移
        offset_main = self.conv_offset(data)
        offset_coarse = self.conv_offset_coarse(data)
        offset_fine = self.conv_offset_fine(data)

        # 将fine偏移调整到与main相同的尺寸（如果需要）
        if offset_fine.shape != offset_main.shape:
            offset_fine = F.interpolate(offset_fine, size=offset_main.shape[2:],
                                        mode='bilinear', align_corners=True)
        if offset_coarse.shape != offset_main.shape:
            offset_coarse = F.interpolate(offset_coarse, size=offset_main.shape[2:],
                                          mode='bilinear', align_corners=True)

        # 融合多尺度偏移
        scale_weights = F.softmax(self.scale_weights, dim=0)
        offset = (scale_weights[0] * offset_main +
                  scale_weights[1] * offset_coarse +
                  scale_weights[2] * offset_fine)

        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)],
                                        device=data.device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # 创新点4: 残差偏移连接
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        residual_offset = offset * self.offset_residual_weight
        final_offset = offset + residual_offset

        pos_x = (final_offset + reference).clamp(-1., +1.)
        pos_y = (final_offset + reference).clamp(-1., +1.)
        pos_z = (final_offset + reference).clamp(-1., +1.)

        return pos_x, pos_y, pos_z, Hk, Wk

    def off_set_unshared_enhanced(self, data, reference):
        """增强版非共享偏移计算"""
        x, y, z = data.chunk(3, dim=1)
        x = einops.rearrange(x, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        y = einops.rearrange(y, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        z = einops.rearrange(z, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)

        offset_r = self.conv_offset_r(x)
        offset_n = self.conv_offset_n(y)
        offset_t = self.conv_offset_t(z)

        # 添加多尺度 (简化版本，只为R通道)
        if hasattr(self, 'conv_offset_r_coarse'):
            offset_r_coarse = self.conv_offset_r_coarse(x)
            if offset_r_coarse.shape != offset_r.shape:
                offset_r_coarse = F.interpolate(offset_r_coarse, size=offset_r.shape[2:],
                                                mode='bilinear', align_corners=True)
            offset_r = 0.7 * offset_r + 0.3 * offset_r_coarse

        Hk, Wk = offset_r.size(2), offset_r.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset_r = offset_r.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_n = offset_n.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_t = offset_t.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset_r = einops.rearrange(offset_r, 'b p h w -> b h w p')
        offset_n = einops.rearrange(offset_n, 'b p h w -> b h w p')
        offset_t = einops.rearrange(offset_t, 'b p h w -> b h w p')

        # 应用残差连接
        offset_r = offset_r + offset_r * self.offset_residual_weight
        offset_n = offset_n + offset_n * self.offset_residual_weight
        offset_t = offset_t + offset_t * self.offset_residual_weight

        pos_x = (offset_r + reference).clamp(-1., +1.)
        pos_y = (offset_n + reference).clamp(-1., +1.)
        pos_z = (offset_t + reference).clamp(-1., +1.)
        return pos_x, pos_y, pos_z, Hk, Wk

    def off_set_shared(self, data, reference):
        """原有的共享偏移计算 (保持兼容性)"""
        data = einops.rearrange(data, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=3 * self.n_group_channels)
        offset = self.conv_offset(data)
        Hk, Wk = offset.size(2), offset.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        pos_x = (offset + reference).clamp(-1., +1.)
        pos_y = (offset + reference).clamp(-1., +1.)
        pos_z = (offset + reference).clamp(-1., +1.)
        return pos_x, pos_y, pos_z, Hk, Wk

    def off_set_unshared(self, data, reference):
        """原有的非共享偏移计算 (保持兼容性)"""
        x, y, z = data.chunk(3, dim=1)
        x = einops.rearrange(x, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        y = einops.rearrange(y, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        z = einops.rearrange(z, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset_r = self.conv_offset_r(x)
        offset_n = self.conv_offset_n(y)
        offset_t = self.conv_offset_t(z)
        Hk, Wk = offset_r.size(2), offset_r.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset_r = offset_r.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_n = offset_n.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_t = offset_t.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset_r = einops.rearrange(offset_r, 'b p h w -> b h w p')
        offset_n = einops.rearrange(offset_n, 'b p h w -> b h w p')
        offset_t = einops.rearrange(offset_t, 'b p h w -> b h w p')
        pos_x = (offset_r + reference).clamp(-1., +1.)
        pos_y = (offset_n + reference).clamp(-1., +1.)
        pos_z = (offset_t + reference).clamp(-1., +1.)
        return pos_x, pos_y, pos_z, Hk, Wk

    @torch.no_grad()
    def visualize_sampling_with_offset(self, feature_maps, sampled_pointss, img_paths, reference_pointss, writer=None,
                                       epoch=0, title='Sampling Points with Offset', pattern=0, patch_size=(16, 16)):
        """
        在原始图像上标记采样点的偏移效果并保存
        """
        # 根据模式设置图像路径前缀
        modality = ['RGB', 'NI', 'TI']
        if pattern == 0:
            prefix = '../RGBNT201/test/RGB/'
        elif pattern == 1:
            prefix = '../RGBNT201/test/NI/'
        elif pattern == 2:
            prefix = '../RGBNT201/test/TI/'
        for i in range(len(img_paths)):
            img_path = prefix + img_paths[i]

            # 加载并调整图像大小
            original_image = Image.open(img_path).resize((128, 256))
            original_image = np.array(original_image)

            # 处理特征图
            feature_map = torch.mean(feature_maps[i], dim=0, keepdim=True)
            feature_map = feature_map.detach().cpu().numpy()
            feature_map = (feature_map - np.min(feature_map)) / np.ptp(feature_map)

            # 转换采样点和参考点为 numpy 格式
            sampled_points = sampled_pointss[i].detach().cpu().numpy()
            reference_points = reference_pointss[i].detach().cpu().numpy()

            # 获取特征图和原始图像的尺寸
            H_feat, W_feat = feature_map.shape[1:]
            H_orig, W_orig = original_image.shape[:2]

            # 计算下采样比例
            scale_x = W_orig / W_feat
            scale_y = H_orig / H_feat

            # 转换坐标到原图系
            sampled_points[:, 1] = (sampled_points[:, 1] + 1) / 2 * (W_feat - 1) * scale_x
            sampled_points[:, 0] = (sampled_points[:, 0] + 1) / 2 * (H_feat - 1) * scale_y
            reference_points[:, 1] = (reference_points[:, 1] + 1) / 2 * (W_feat - 1) * scale_x
            reference_points[:, 0] = (reference_points[:, 0] + 1) / 2 * (H_feat - 1) * scale_y

            # 绘制原图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(original_image, aspect='auto')
            ax.set_title(title)

            # 更改采样点样式和箭头样式
            for ref, samp in zip(reference_points, sampled_points):
                ref_y, ref_x = ref
                samp_y, samp_x = samp

                # 参考点用淡蓝色
                ax.scatter(ref_x, ref_y, c='skyblue', s=70, marker='o', edgecolor='black', linewidth=2)
                # 目标点用橙色
                ax.scatter(samp_x, samp_y, c='orange', s=70, marker='x', linewidth=12)

                # 箭头颜色为半透明绿色
                ax.arrow(ref_x, ref_y, samp_x - ref_x, samp_y - ref_y, color='limegreen', alpha=0.7,
                         head_width=4, head_length=6, linewidth=6, length_includes_head=True)

            # 绘制 patch 分隔线
            patch_height, patch_width = patch_size
            for y in range(0, H_orig, patch_height):
                ax.plot([0, W_orig], [y, y], color='white', linewidth=1.5, linestyle='--')
            for x in range(0, W_orig, patch_width):
                ax.plot([x, x], [0, H_orig], color='white', linewidth=1.5, linestyle='--')

            ax.set_xlim(-1, W_orig)
            ax.set_ylim(H_orig, -1)

            # 保存到 writer
            if writer is not None:
                writer.add_figure(f"{title}", fig, global_step=epoch)
            plt.savefig(
                f'../off_vis/{modality[pattern]}/{img_path.split("/")[-1].split(".")[0]}.png')
            plt.close(fig)

    def show_cam_on_image(self, img: np.ndarray,
                          mask: np.ndarray,
                          use_rgb: bool = False,
                          colormap: int = cv2.COLORMAP_HOT,
                          image_weight: float = 0.3) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
        :returns: The default image with the cam overlay.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        if np.max(img) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")

        if image_weight < 0 or image_weight > 1:
            raise Exception(
                f"image_weight should be in the range [0, 1].\
                    Got: {image_weight}")

        cam = (1 - image_weight) * heatmap + image_weight * img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    @torch.no_grad()
    def visualize_attention_with_distribution(self, attn_map, img_paths, index, reference_pointss, sampled_pointss,
                                              writer=None, epoch=0, title='Attention Heatmap with Distribution',
                                              patch_size=(16, 16), text=''):
        """
        在原始图像上根据 attn_map 选择一个注意力分布，生成热图并覆盖，并显示描述文本
        """
        modality = ['v_RGB', 'v_NIR', 'v_TIR', 't_RGB', 't_NIR', 't_TIR']
        if index == 0 or index == 3:
            prefix = '../RGBNT201/test/RGB/'
            text = text['rgb_text']
        elif index == 1 or index == 4:
            prefix = '../RGBNT201/test/NI/'
            text = text['ni_text']
        elif index == 2 or index == 5:
            prefix = '../RGBNT201/test/TI/'
            text = text['ti_text']

        grid_height = 7
        grid_width = 3

        for i in range(len(img_paths)):
            img_path = prefix + img_paths[i]
            original_image = Image.open(img_path).convert('RGB').resize((128, 256))
            original_image = np.float32(original_image) / 255

            H_orig, W_orig = original_image.shape[:2]
            grid_height_size = H_orig // grid_height
            grid_width_size = W_orig // grid_width

            # 根据 index 选择对应的注意力区域
            if index == 0 or index == 3:
                selected_attn = attn_map[i, index, :21]
            elif index == 1 or index == 4:
                selected_attn = attn_map[i, index, 21:42]
            else:
                selected_attn = attn_map[i, index, 42:]

            selected_attn = selected_attn * 1000 * 2
            selected_attn = torch.softmax(selected_attn, dim=0)
            selected_attn = selected_attn.detach().cpu().numpy()

            # 初始化一个全为零的热图
            heatmap = np.zeros((H_orig, W_orig))

            # 根据网格和注意力权重构建热图
            for row in range(grid_height):
                for col in range(grid_width):
                    y_start = row * grid_height_size
                    x_start = col * grid_width_size
                    y_end = (row + 1) * grid_height_size if row < grid_height - 1 else H_orig
                    x_end = (col + 1) * grid_width_size if col < grid_width - 1 else W_orig

                    weight = selected_attn[row * grid_width + col]
                    heatmap[y_start:y_end, x_start:x_end] += weight

            # 将热图与原图叠加
            overlay = self.show_cam_on_image(original_image, heatmap, use_rgb=True, image_weight=0.5)

            # 创建可视化图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(overlay, aspect='auto')
            ax.set_title(f"{title} - Image {i + 1}")

            # 在图像中添加描述文本
            ax.text(
                0.5, -0.12,
                textwrap.fill(text[i], width=80),
                transform=ax.transAxes,
                color='black',
                fontsize=10,
                ha='center',
                va='bottom',
                weight='bold'
            )

            ax.set_xlim(-1, W_orig)
            ax.set_ylim(H_orig, -1)

            if writer is not None:
                writer.add_figure(f"{title}", fig, global_step=epoch)

            output_path = f'../attn_vis/{modality[index]}/{img_path.split("/")[-1].split(".")[0]}.png'
            plt.savefig(output_path)
            plt.close(fig)

    def forward(self, x, y, z, writer=None, epoch=None, img_path=None, text=''):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # 创新点1: 应用自适应模态权重
        x, y, z = self.adaptive_modal_weighting(x, y, z)

        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)

        # 创新点2&4: 多尺度偏移融合 + 残差连接
        pos_x, pos_y, pos_z, Hk, Wk = self.multi_scale_offset_fusion(data, reference)

        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],
            mode='bilinear', align_corners=True)

        sampled_x, sampled_y, sampled_z = [
            t.reshape(B, C, 1, n_sample).squeeze(2).permute(2, 0, 1)
            for t in [sampled_x, sampled_y, sampled_z]
        ]

        return sampled_x, sampled_y, sampled_z

    def forwardOld(self, query, x, y, z, writer=None, epoch=None, img_path=None, text=''):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        dtype, device = x.dtype, x.device

        # 创新点1: 应用自适应模态权重
        x, y, z = self.adaptive_modal_weighting(x, y, z)

        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)

        # 使用增强的偏移计算或原有的
        if self.share_offset:
            pos_x, pos_y, pos_z, Hk, Wk = self.multi_scale_offset_fusion(data, reference)
        else:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_unshared(data, reference)

        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],
            mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, 1, n_sample)
        sampled_y = sampled_y.reshape(B, C, 1, n_sample)
        sampled_z = sampled_z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)

        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)

        # 创新点3: 可学习温度缩放
        attn = attn.mul(self.scale * self.temperature)
        attn = F.softmax(attn, dim=2)

        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def forward_woCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # 应用自适应模态权重
        x, y, z = self.adaptive_modal_weighting(x, y, z)

        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)

        if self.share_offset:
            pos_x, pos_y, pos_z, Hk, Wk = self.multi_scale_offset_fusion(data, reference)
        else:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_unshared(data, reference)

        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],
            mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, 1, n_sample)
        sampled_y = sampled_y.reshape(B, C, 1, n_sample)
        sampled_z = sampled_z.reshape(B, C, 1, n_sample)
        input = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)

        q = self.proj_q(input)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        k = self.proj_k(input).reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        v = self.proj_v(input).reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)

        # 应用温度缩放
        attn = attn.mul(self.scale * self.temperature)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, 3 * Hk * Wk)
        out = self.proj_drop(self.proj_out(out))
        out = input + out
        sampled_x, sampled_y, sampled_z = out.chunk(3, dim=-1)

        sampled_x = torch.mean(sampled_x, dim=-1, keepdim=True)
        sampled_y = torch.mean(sampled_y, dim=-1, keepdim=True)
        sampled_z = torch.mean(sampled_z, dim=-1, keepdim=True)

        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        sampled_2 = torch.cat([sampled, sampled], dim=-1)
        return sampled_2.squeeze(2)

    def forward_woSample_wCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()

        # 应用自适应模态权重
        x, y, z = self.adaptive_modal_weighting(x, y, z)

        n_sample = H * W
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)

        # 应用温度缩放
        attn = attn.mul(self.scale * self.temperature)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def forward_woSample_woCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()

        # 应用自适应模态权重
        x, y, z = self.adaptive_modal_weighting(x, y, z)

        n_sample = H * W
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        input = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(input)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        k = self.proj_k(input).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(input).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)

        # 应用温度缩放
        attn = attn.mul(self.scale * self.temperature)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, 3 * n_sample)
        out = self.proj_drop(self.proj_out(out))
        out = input + out
        sampled_x, sampled_y, sampled_z = out.chunk(3, dim=-1)

        sampled_x = torch.mean(sampled_x, dim=-1, keepdim=True)
        sampled_y = torch.mean(sampled_y, dim=-1, keepdim=True)
        sampled_z = torch.mean(sampled_z, dim=-1, keepdim=True)

        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        sampled_2 = torch.cat([sampled, sampled], dim=-1)
        return sampled_2.squeeze(2)

    def forward_woOffset(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()

        # 应用自适应模态权重
        x, y, z = self.adaptive_modal_weighting(x, y, z)

        data = torch.cat([x, y, z], dim=1)
        # 注意：这里原代码使用了conv_v, conv_n, conv_t，但在原始代码中没有定义
        # 为了保持兼容性，我们直接使用输入
        h_new, w_new = x.size(2), x.size(3)
        n_sample = h_new * w_new
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)

        # 应用温度缩放
        attn = attn.mul(self.scale * self.temperature)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)



class RWKV_CrossAttention(nn.Module):
    def __init__(self, dim, n_query=1):
        super().__init__()
        self.dim = dim
        self.n_query = n_query

        # 可学习的 query
        self.query = nn.Parameter(torch.randn(1, n_query, dim))

        # 用于计算 key/value/receptance/decay
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.time_decay = nn.Linear(dim, dim, bias=False)

    def forward(self, kv_feats):
        """
        kv_feats: [B, T, C] 来自另一个结构的特征（比如 CNN 或 ViT）
        return: [B, n_query, C]
        """
        B, T, C = kv_feats.shape

        # 扩展 query 为 batch 尺寸
        q = self.query.expand(B, -1, -1)  # [B, n_query, C]

        # 得到 key / value / decay
        k = self.key(kv_feats)            # [B, T, C]
        v = self.value(kv_feats)          # [B, T, C]
        w = self.time_decay(kv_feats)     # [B, T, C]
        ew = torch.exp(-torch.relu(w))    # 控制每个 token 的衰减

        # 计算门控 r（由 query 控制）
        r = torch.sigmoid(self.receptance(q))  # [B, n_query, C]

        # 衰减加权的 key * value（RWKV 样式）
        weighted_kv = k * v * ew  # [B, T, C]

        # 总加权：每个 query 对所有 kv 进行 sum（或你可以用 dot-product kernel）
        out = r * torch.sum(weighted_kv, dim=1, keepdim=True)  # [B, n_query, C]

        return out


class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

import torch
import torch.nn as nn

class TokenSE(nn.Module):
    def __init__(self, token_dim, reduction=4, use_residual=True):
        """
        Args:
            token_dim (int): 输入 token 的数量 T
            reduction (int): 通道缩减比例
            use_residual (bool): 是否使用残差连接
        """
        super().__init__()
        self.use_residual = use_residual

        self.fc = nn.Sequential(
            nn.Linear(token_dim, token_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim // reduction, token_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, T, N] - B=batch, T=token数, N=每个token的特征维度
        """
        B, T, N = x.shape
        # squeeze: 在特征维度上聚合
        s = x.mean(dim=2)           # [B, T]
        weights = self.fc(s)        # [B, T]
        weights = weights.unsqueeze(2)  # [B, T, 1]

        # scale: token加权
        if self.use_residual:
            return x * weights + x
        else:
            return x * weights


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class FreMLP(nn.Module):

    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq) #分离出频域数据的 幅度（magnitude）
        pha = torch.angle(x_freq) #分离出频域数据的 相位（phase）
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out

#cv fenghe  EBblock104
class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, dilations=[1, 4, 9], extra_depth_wise=True):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        y = inp
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
        x_step2 = self.norm2(y)  # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2)  # size [B, C, H, W]
        x = y * x_freq
        x = y + x * self.gamma

        return x


#################SE PART ############################################
class MultiModalTokenSE(nn.Module):
    """
    多模态交互式TokenSE模块 - 稳定版
    修复所有维度问题，确保稳定运行
    """

    def __init__(self, token_dim, feature_dim, reduction=4, use_residual=True,
                 interaction_mode='fusion'):
        """
        Args:
            token_dim (int): Token数量 T
            feature_dim (int): 每个Token的特征维度 N
            reduction (int): 通道缩减比例
            use_residual (bool): 是否使用残差连接
            interaction_mode (str): 交互模式 ['fusion', 'adaptive_weight', 'simple']
        """
        super().__init__()
        self.token_dim = token_dim
        self.feature_dim = feature_dim
        self.use_residual = use_residual
        self.interaction_mode = interaction_mode

        # 🚀 创新1: 单模态特征聚合器
        self.rgb_aggregator = self._build_aggregator(reduction)
        self.nir_aggregator = self._build_aggregator(reduction)
        self.tir_aggregator = self._build_aggregator(reduction)

        # 🚀 创新2: 跨模态交互机制
        if interaction_mode == 'fusion':
            self.modal_fusion = ModalFusion(token_dim, reduction)
        elif interaction_mode == 'adaptive_weight':
            self.adaptive_weighting = AdaptiveWeighting(token_dim, reduction)
        elif interaction_mode == 'simple':
            self.simple_interaction = SimpleInteraction(feature_dim)

        # 🚀 创新3: 全局上下文整合
        self.global_context_net = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid()
        )

        # 🚀 创新4: 自适应模态平衡 - 简化版
        self.modal_balance = nn.Sequential(
            nn.Linear(3, 8),  # 输入3个标量
            nn.ReLU(inplace=True),
            nn.Linear(8, 3),  # 输出3个权重
            nn.Softmax(dim=-1)
        )

        # 🚀 创新5: 增强权重学习
        self.enhancement_net = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid()
        )

    def _build_aggregator(self, reduction):
        """构建单模态特征聚合器"""
        return nn.Sequential(
            nn.Linear(self.token_dim, max(1, self.token_dim // reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, self.token_dim // reduction), self.token_dim),
            nn.Sigmoid()
        )

    def forward(self, rgb_tokens, nir_tokens, tir_tokens):
        """
        Args:
            rgb_tokens: [B, T, N] RGB模态token特征
            nir_tokens: [B, T, N] NIR模态token特征
            tir_tokens: [B, T, N] TIR模态token特征
        Returns:
            enhanced_rgb: [B, T, N] 增强的RGB特征
            enhanced_nir: [B, T, N] 增强的NIR特征
            enhanced_tir: [B, T, N] 增强的TIR特征
        """
        B, T, N = rgb_tokens.shape

        # ========== Step 1: 单模态特征聚合 ==========
        # 在特征维度上聚合，获得每个Token的全局描述
        rgb_global = rgb_tokens.mean(dim=2)  # [B, T]
        nir_global = nir_tokens.mean(dim=2)  # [B, T]
        tir_global = tir_tokens.mean(dim=2)  # [B, T]

        # 学习单模态Token权重
        rgb_weights = self.rgb_aggregator(rgb_global)  # [B, T]
        nir_weights = self.nir_aggregator(nir_global)  # [B, T]
        tir_weights = self.tir_aggregator(tir_global)  # [B, T]

        # ========== Step 2: 跨模态交互增强 ==========
        if self.interaction_mode == 'fusion':
            rgb_enhanced, nir_enhanced, tir_enhanced = self.modal_fusion(
                rgb_global, nir_global, tir_global, rgb_tokens, nir_tokens, tir_tokens
            )
        elif self.interaction_mode == 'adaptive_weight':
            rgb_enhanced, nir_enhanced, tir_enhanced = self.adaptive_weighting(
                rgb_global, nir_global, tir_global, rgb_tokens, nir_tokens, tir_tokens
            )
        elif self.interaction_mode == 'simple':
            rgb_enhanced, nir_enhanced, tir_enhanced = self.simple_interaction(
                rgb_tokens, nir_tokens, tir_tokens
            )
        else:
            rgb_enhanced, nir_enhanced, tir_enhanced = rgb_tokens, nir_tokens, tir_tokens

        # ========== Step 3: 全局上下文整合 ==========
        # 融合三模态的全局信息
        global_context = torch.cat([rgb_global, nir_global, tir_global], dim=-1)  # [B, T*3]
        global_weights = self.global_context_net(global_context)  # [B, T]

        # ========== Step 4: 自适应模态平衡 ==========
        # 计算每个模态的全局特征强度（单个标量）
        rgb_strength = rgb_global.mean(dim=1)  # [B]
        nir_strength = nir_global.mean(dim=1)  # [B]
        tir_strength = tir_global.mean(dim=1)  # [B]

        # 组合为 [B, 3] 张量
        modal_strengths = torch.stack([rgb_strength, nir_strength, tir_strength], dim=1)  # [B, 3]
        modal_weights = self.modal_balance(modal_strengths)  # [B, 3]

        # 提取各模态权重并扩展到token维度
        w_rgb = modal_weights[:, 0].unsqueeze(1).repeat(1, T)  # [B, T]
        w_nir = modal_weights[:, 1].unsqueeze(1).repeat(1, T)  # [B, T]
        w_tir = modal_weights[:, 2].unsqueeze(1).repeat(1, T)  # [B, T]

        # ========== Step 5: 最终权重计算与应用 ==========
        # 综合各种权重
        final_rgb_weights = (rgb_weights * global_weights * w_rgb).unsqueeze(-1)  # [B, T, 1]
        final_nir_weights = (nir_weights * global_weights * w_nir).unsqueeze(-1)  # [B, T, 1]
        final_tir_weights = (tir_weights * global_weights * w_tir).unsqueeze(-1)  # [B, T, 1]

        # 增强权重学习
        enhancement_weights = self.enhancement_net(global_context).unsqueeze(-1)  # [B, T, 1]

        # 应用权重增强特征
        if self.use_residual:
            final_rgb = rgb_enhanced * final_rgb_weights * enhancement_weights + rgb_tokens
            final_nir = nir_enhanced * final_nir_weights * enhancement_weights + nir_tokens
            final_tir = tir_enhanced * final_tir_weights * enhancement_weights + tir_tokens
        else:
            final_rgb = rgb_enhanced * final_rgb_weights * enhancement_weights
            final_nir = nir_enhanced * final_nir_weights * enhancement_weights
            final_tir = tir_enhanced * final_tir_weights * enhancement_weights

        return final_rgb, final_nir, final_tir


class SimpleInteraction(nn.Module):
    """简单但稳定的跨模态交互"""

    def __init__(self, feature_dim):
        super().__init__()
        self.rgb_proj = nn.Linear(feature_dim, feature_dim)
        self.nir_proj = nn.Linear(feature_dim, feature_dim)
        self.tir_proj = nn.Linear(feature_dim, feature_dim)

        self.interaction_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, rgb_tokens, nir_tokens, tir_tokens):
        # 简单的线性交互
        rgb_enhanced = rgb_tokens + self.interaction_weight * (
                self.nir_proj(nir_tokens) + self.tir_proj(tir_tokens)
        ) / 2

        nir_enhanced = nir_tokens + self.interaction_weight * (
                self.rgb_proj(rgb_tokens) + self.tir_proj(tir_tokens)
        ) / 2

        tir_enhanced = tir_tokens + self.interaction_weight * (
                self.rgb_proj(rgb_tokens) + self.nir_proj(nir_tokens)
        ) / 2

        return rgb_enhanced, nir_enhanced, tir_enhanced


class ModalFusion(nn.Module):
    """模态融合交互机制"""

    def __init__(self, token_dim, reduction=4):
        super().__init__()
        self.fusion_net = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim * 2, token_dim * 3),
            nn.Sigmoid()
        )

    def forward(self, rgb_global, nir_global, tir_global,
                rgb_tokens, nir_tokens, tir_tokens):
        # 融合全局描述符
        fused_global = torch.cat([rgb_global, nir_global, tir_global], dim=-1)  # [B, T*3]
        fusion_weights = self.fusion_net(fused_global)  # [B, T*3]

        # 分离权重
        w_rgb, w_nir, w_tir = fusion_weights.chunk(3, dim=-1)  # [B, T] each

        # 应用权重进行交互
        rgb_enhanced = rgb_tokens * w_rgb.unsqueeze(-1) + 0.1 * (
                nir_tokens * w_nir.unsqueeze(-1) + tir_tokens * w_tir.unsqueeze(-1)
        ) / 2

        nir_enhanced = nir_tokens * w_nir.unsqueeze(-1) + 0.1 * (
                rgb_tokens * w_rgb.unsqueeze(-1) + tir_tokens * w_tir.unsqueeze(-1)
        ) / 2

        tir_enhanced = tir_tokens * w_tir.unsqueeze(-1) + 0.1 * (
                rgb_tokens * w_rgb.unsqueeze(-1) + nir_tokens * w_nir.unsqueeze(-1)
        ) / 2

        return rgb_enhanced, nir_enhanced, tir_enhanced


class AdaptiveWeighting(nn.Module):
    """自适应权重交互机制"""

    def __init__(self, token_dim, reduction=4):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(token_dim * 3, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim * 3),
            nn.Softmax(dim=-1)
        )

        self.interaction_strength = nn.Parameter(torch.tensor(0.2))

    def forward(self, rgb_global, nir_global, tir_global,
                rgb_tokens, nir_tokens, tir_tokens):
        # 计算自适应权重
        combined_global = torch.cat([rgb_global, nir_global, tir_global], dim=-1)  # [B, T*3]
        adaptive_weights = self.weight_net(combined_global)  # [B, T*3]

        w_rgb, w_nir, w_tir = adaptive_weights.chunk(3, dim=-1)  # [B, T] each

        # 交互式增强：每个模态都能看到其他模态的信息
        rgb_enhanced = rgb_tokens + self.interaction_strength * (
                w_nir.unsqueeze(-1) * nir_tokens + w_tir.unsqueeze(-1) * tir_tokens
        ) / 2

        nir_enhanced = nir_tokens + self.interaction_strength * (
                w_rgb.unsqueeze(-1) * rgb_tokens + w_tir.unsqueeze(-1) * tir_tokens
        ) / 2

        tir_enhanced = tir_tokens + self.interaction_strength * (
                w_rgb.unsqueeze(-1) * rgb_tokens + w_nir.unsqueeze(-1) * nir_tokens
        ) / 2

        return rgb_enhanced, nir_enhanced, tir_enhanced



# ==================== 修复版特征区分度增强融合系统 ====================
class FeatureDiversityEnhancedFusion(nn.Module):
    """
    特征区分度增强的融合系统 - 修复版
    专门为MoE输入设计，最大化各模态特征的区分度
    """

    def __init__(self, feat_dim, dropout=0.1):
        super().__init__()
        self.feat_dim = feat_dim
        self.dropout = dropout

        # 保持原始token设计
        scale = feat_dim ** -0.5
        self.base_tokens = nn.ParameterDict({
            'r': nn.Parameter(scale * torch.randn(1, 1, feat_dim)),
            'n': nn.Parameter(scale * torch.randn(1, 1, feat_dim)),
            't': nn.Parameter(scale * torch.randn(1, 1, feat_dim)),
            'rn': nn.Parameter(scale * torch.randn(1, 1, feat_dim)),
            'rt': nn.Parameter(scale * torch.randn(1, 1, feat_dim)),
            'nt': nn.Parameter(scale * torch.randn(1, 1, feat_dim)),
            'rnt': nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        })

        # 🚀 创新1: 模态特异性增强器
        self.modality_specific_enhancers = nn.ModuleDict({
            'r': ModalitySpecificEnhancer(feat_dim, modality_type='rgb'),
            'n': ModalitySpecificEnhancer(feat_dim, modality_type='nir'),
            't': ModalitySpecificEnhancer(feat_dim, modality_type='tir'),
            'rn': ModalitySpecificEnhancer(feat_dim, modality_type='dual'),
            'rt': ModalitySpecificEnhancer(feat_dim, modality_type='dual'),
            'nt': ModalitySpecificEnhancer(feat_dim, modality_type='dual'),
            'rnt': ModalitySpecificEnhancer(feat_dim, modality_type='triple')
        })

        # 🚀 创新2: 特征分离器
        self.feature_separators = nn.ModuleDict({
            name: FeatureSeparator(feat_dim, separator_id=i)
            for i, name in enumerate(['r', 'n', 't', 'rn', 'rt', 'nt', 'rnt'])
        })

        # 🚀 创新3: 对比学习损失计算器
        self.contrastive_projector = ContrastiveProjector(feat_dim)

        # 🚀 创新4: 区分度量化器
        self.diversity_quantifier = DiversityQuantifier(feat_dim)

        # 原始注意力模块
        head_num_attn = feat_dim // 64
        self.attentions = nn.ModuleDict({
            'r': nn.MultiheadAttention(embed_dim=feat_dim, num_heads=head_num_attn, dropout=dropout),
            'n': nn.MultiheadAttention(embed_dim=feat_dim, num_heads=head_num_attn, dropout=dropout),
            't': nn.MultiheadAttention(embed_dim=feat_dim, num_heads=head_num_attn, dropout=dropout),
            'rn': nn.MultiheadAttention(embed_dim=feat_dim, num_heads=head_num_attn, dropout=dropout),
            'rt': nn.MultiheadAttention(embed_dim=feat_dim, num_heads=head_num_attn, dropout=dropout),
            'nt': nn.MultiheadAttention(embed_dim=feat_dim, num_heads=head_num_attn, dropout=dropout),
            'rnt': nn.MultiheadAttention(embed_dim=feat_dim, num_heads=head_num_attn, dropout=dropout)
        })

        # 🚀 创新5: 区分度损失权重
        self.diversity_loss_weight = nn.Parameter(torch.tensor(1.0))

        # 存储中间结果用于分析
        self.intermediate_features = {}

    def forward(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global, return_loss=True):
        """
        Args:
            return_loss: 是否返回区分度损失（训练时True，推理时False）
        Returns:
            features: 7个具有高区分度的特征 [B, 512]
            diversity_loss: 用于训练的区分度损失（可选）
        """
        batch = RGB_cash.size(1)

        # Step 1: 构建特征图（保持原始逻辑）
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)

        RGB = torch.cat([r_global, RGB_cash], dim=0)
        NI = torch.cat([n_global, NI_cash], dim=0)
        TI = torch.cat([t_global, TI_cash], dim=0)
        RGB_NI = torch.cat([RGB, NI], dim=0)
        RGB_TI = torch.cat([RGB, TI], dim=0)
        NI_TI = torch.cat([NI, TI], dim=0)
        RGB_NI_TI = torch.cat([RGB, NI, TI], dim=0)

        feature_maps = [RGB, NI, TI, RGB_NI, RGB_TI, NI_TI, RGB_NI_TI]
        feature_names = ['r', 'n', 't', 'rn', 'rt', 'nt', 'rnt']

        # Step 2: 原始注意力提取
        raw_features = []
        for name, feature_map in zip(feature_names, feature_maps):
            token = self.base_tokens[name].repeat(1, batch, 1)
            attn_output = self.attentions[name](token, feature_map, feature_map)[0]
            feature = attn_output.permute(1, 2, 0).squeeze()
            raw_features.append(feature)

        # Step 3: 模态特异性增强
        specific_features = []
        for name, feature in zip(feature_names, raw_features):
            enhanced_feature = self.modality_specific_enhancers[name](feature)
            specific_features.append(enhanced_feature)

        # Step 4: 特征分离
        separated_features = []
        for i, (name, feature) in enumerate(zip(feature_names, specific_features)):
            separated_feature = self.feature_separators[name](feature, specific_features, i)
            separated_features.append(separated_feature)

        # 存储中间特征用于分析
        self.intermediate_features = {
            'raw': raw_features,
            'specific': specific_features,
            'separated': separated_features
        }

        if return_loss:
            # Step 5: 计算区分度损失
            diversity_loss = self._compute_diversity_loss(separated_features)
            return separated_features, diversity_loss
        else:
            return separated_features

    def _compute_diversity_loss(self, features):
        """计算特征区分度损失"""
        # 对比学习损失
        contrastive_loss = self.contrastive_projector(features)

        # 区分度量化损失
        diversity_loss = self.diversity_quantifier(features)

        # 总损失
        total_loss = contrastive_loss + diversity_loss

        return total_loss * self.diversity_loss_weight

    def get_diversity_metrics(self):
        """获取区分度指标用于监控"""
        if not self.intermediate_features:
            return {}

        metrics = {}
        for stage_name, features in self.intermediate_features.items():
            # 计算特征间相似度
            similarities = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    sim = F.cosine_similarity(features[i], features[j], dim=-1).mean()
                    similarities.append(sim.item())

            avg_similarity = sum(similarities) / len(similarities) if similarities else 0
            metrics[f'{stage_name}_avg_similarity'] = avg_similarity
            metrics[f'{stage_name}_diversity_score'] = 1.0 - avg_similarity

        return metrics


class ModalitySpecificEnhancer(nn.Module):
    """模态特异性增强器"""

    def __init__(self, feat_dim, modality_type):
        super().__init__()
        self.modality_type = modality_type
        self.feat_dim = feat_dim

        # 根据模态类型设计不同的增强策略
        if modality_type == 'rgb':
            # RGB模态：强调颜色和纹理特征，使用ReLU
            self.enhancer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.LayerNorm(feat_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim)
            )
        elif modality_type == 'nir':
            # NIR模态：强调植被和结构，使用GELU
            self.enhancer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.LayerNorm(feat_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim)
            )
        elif modality_type == 'tir':
            # TIR模态：强调温度和热特征，使用自定义激活
            self.enhancer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.LayerNorm(feat_dim * 2),
                nn.SiLU(),  # 使用SiLU替代Mish
                nn.Dropout(0.1),
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim)
            )
        elif modality_type == 'dual':
            # 双模态：强调互补性，使用ELU
            self.enhancer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.LayerNorm(feat_dim * 2),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim)
            )
        else:  # triple
            # 三模态：强调综合性，使用LeakyReLU
            self.enhancer = nn.Sequential(
                nn.Linear(feat_dim, feat_dim * 2),
                nn.LayerNorm(feat_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
                nn.Linear(feat_dim * 2, feat_dim),
                nn.LayerNorm(feat_dim)
            )

        # 模态身份编码
        self.modality_embedding = nn.Parameter(torch.randn(feat_dim) * 0.02)

    def forward(self, feature):
        """模态特异性增强"""
        # 特异性增强
        enhanced = self.enhancer(feature)

        # 添加模态身份信息
        modality_enhanced = enhanced + self.modality_embedding.unsqueeze(0)

        # 残差连接，但权重较小以突出特异性
        final_feature = 0.7 * modality_enhanced + 0.3 * feature

        return final_feature


class FeatureSeparator(nn.Module):
    """特征分离器 - 最大化与其他特征的区别"""

    def __init__(self, feat_dim, separator_id):
        super().__init__()
        self.feat_dim = feat_dim
        self.separator_id = separator_id

        # 分离网络
        self.separator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        # 正交化投影器
        self.orthogonal_projector = OrthogonalProjector(feat_dim)

        # 分离强度控制
        self.separation_strength = nn.Parameter(torch.tensor(0.3))

    def forward(self, target_feature, all_features, target_id):
        """
        Args:
            target_feature: 当前特征 [B, feat_dim]
            all_features: 所有特征列表
            target_id: 当前特征的ID
        """
        # 基础分离
        separated = self.separator(target_feature)

        # 正交化处理
        orthogonalized = self.orthogonal_projector(separated, all_features, target_id)

        # 控制分离强度
        final_feature = (1 - self.separation_strength) * target_feature + \
                        self.separation_strength * orthogonalized

        return final_feature


class OrthogonalProjector(nn.Module):
    """正交化投影器"""

    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

        # 正交化权重
        self.orthogonal_weight = nn.Parameter(torch.eye(feat_dim) * 0.05)

    def forward(self, target_feature, all_features, target_id):
        """正交化投影"""
        # 计算与其他特征的相似度
        similarities = []
        for i, other_feature in enumerate(all_features):
            if i != target_id and other_feature is not None:
                sim = F.cosine_similarity(target_feature, other_feature, dim=-1, eps=1e-8)
                similarities.append(sim.unsqueeze(-1))

        if not similarities:
            return target_feature

        # 平均相似度
        avg_similarity = torch.cat(similarities, dim=-1).mean(dim=-1, keepdim=True)

        # 正交化投影
        orthogonal_component = torch.matmul(target_feature, self.orthogonal_weight)

        # 根据相似度调整正交化强度
        orthogonalized = target_feature - avg_similarity * orthogonal_component

        return orthogonalized


class ContrastiveProjector(nn.Module):
    """对比学习投影器 - 修复版"""

    def __init__(self, feat_dim, projection_dim=128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, projection_dim)
        )

        self.temperature = nn.Parameter(torch.tensor(0.1))

    def forward(self, features):
        """计算对比学习损失"""
        # 投影到对比空间并L2标准化
        projected_features = []
        for f in features:
            proj = self.projector(f)
            # 使用F.normalize替代nn.L2Norm
            normalized = F.normalize(proj, p=2, dim=-1)
            projected_features.append(normalized)

        # 计算对比损失
        contrastive_loss = 0
        num_pairs = 0

        for i in range(len(projected_features)):
            for j in range(i + 1, len(projected_features)):
                # 计算相似度
                similarity = torch.sum(projected_features[i] * projected_features[j], dim=-1)
                similarity = similarity / self.temperature

                # 对比损失：希望不同模态特征尽可能不相似
                loss = torch.mean(torch.exp(similarity))
                contrastive_loss += loss
                num_pairs += 1

        return contrastive_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=features[0].device)


class DiversityQuantifier(nn.Module):
    """区分度量化器"""

    def __init__(self, feat_dim):
        super().__init__()
        self.feat_dim = feat_dim

        # 多样性度量网络
        self.diversity_net = nn.Sequential(
            nn.Linear(feat_dim * 7, feat_dim * 2),
            nn.ReLU(),
            nn.Linear(feat_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """量化特征区分度"""
        # 拼接所有特征
        all_features = torch.cat(features, dim=-1)  # [B, 7*feat_dim]

        # 计算多样性得分
        diversity_score = self.diversity_net(all_features)  # [B, 1]

        # 多样性损失：希望多样性得分尽可能高
        diversity_loss = torch.mean(1.0 - diversity_score)

        return diversity_loss


# ==================== 简化版MoE友好融合 ====================

class FeatureDiversifier(nn.Module):
    """简化的特征区分器"""

    def __init__(self, feat_dim, diversifier_id):
        super().__init__()
        self.diversifier_id = diversifier_id

        # 区分网络
        self.diversify_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        # 区分强度
        self.diversity_strength = nn.Parameter(torch.tensor(0.2))

    def forward(self, target_feature, all_features, target_id):
        """特征区分化"""
        # 基础区分
        diversified = self.diversify_net(target_feature)

        # 计算与其他特征的相似度并减少相似性
        other_features = [f for i, f in enumerate(all_features) if i != target_id]
        if other_features:
            avg_other = torch.stack(other_features).mean(dim=0)
            # 朝远离平均值的方向调整
            direction = diversified - avg_other
            adjusted = diversified + self.diversity_strength * direction
        else:
            adjusted = diversified

        # 残差连接
        final_feature = 0.8 * adjusted + 0.2 * target_feature

        return final_feature


class BoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8):
        super(BoQBlock, self).__init__()

        self.encoder = torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=nheads, dim_feedforward=4 * in_dim,
                                                        batch_first=True, dropout=0.)
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim))

        # the following two lines are used during training only, you can cache their output in eval.
        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)
        #####

        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)
        return x, out, attn.detach()


class BoQ(torch.nn.Module):
    def __init__(self, in_dim=512, num_queries=32, num_layers=4, row_dim=32):
        super().__init__()
        self.norm_input = torch.nn.LayerNorm(in_dim)

        self.boqs = torch.nn.ModuleList([
            BoQBlock(in_dim, num_queries, nheads=in_dim // 64) for _ in range(num_layers)])

        self.fc = torch.nn.Linear(num_layers * num_queries, row_dim)

    def forward(self, x):
        # x shape: [B, seq_len, dim]
        x = self.norm_input(x)

        outs = []
        attns = []
        for i in range(len(self.boqs)):
            x, out, attn = self.boqs[i](x)
            outs.append(out)
            attns.append(attn)

        out = torch.cat(outs, dim=1)  # [B, num_layers*num_queries, dim]
        out = self.fc(out.permute(0, 2, 1))  # [B, dim, row_dim]
        out = out.flatten(1)  # [B, dim*row_dim]
        out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out, attns


class MultiModalBoQ(nn.Module):
    def __init__(self, in_dim=512, num_queries=32, num_layers=4, row_dim=32):
        super().__init__()

        # 为不同的特征组合创建BoQ网络
        # 由于输入序列长度不同，可以共享参数或分别创建
        self.boq_single = BoQ(in_dim, num_queries, num_layers, row_dim)  # 单模态 (129)
        self.boq_dual = BoQ(in_dim, num_queries, num_layers, row_dim)  # 双模态 (258)
        self.boq_triple = BoQ(in_dim, num_queries, num_layers, row_dim)  # 三模态 (387)

    def forward(self, features_dict):
        """
        features_dict: 包含所有特征的字典
        """
        results = {}
        attentions = {}

        # 处理单模态特征 (129, 64, 512)
        for modality in ['RGB', 'NI', 'TI']:
            if modality in features_dict:
                feat = features_dict[modality]  # [129, 64, 512]
                feat = feat.permute(1, 0, 2)  # [64, 129, 512] -> [B, seq_len, dim]

                out, attn = self.boq_single(feat)
                results[modality] = out
                attentions[modality] = attn

        # 处理双模态特征 (258, 64, 512)
        for modality in ['RGB_NI', 'RGB_TI', 'NI_TI']:
            if modality in features_dict:
                feat = features_dict[modality]  # [258, 64, 512]
                feat = feat.permute(1, 0, 2)  # [64, 258, 512] -> [B, seq_len, dim]

                out, attn = self.boq_dual(feat)
                results[modality] = out
                attentions[modality] = attn

        # 处理三模态特征 (387, 64, 512)
        if 'RGB_NI_TI' in features_dict:
            feat = features_dict['RGB_NI_TI']  # [387, 64, 512]
            feat = feat.permute(1, 0, 2)  # [64, 387, 512] -> [B, seq_len, dim]

            out, attn = self.boq_triple(feat)
            results['RGB_NI_TI'] = out
            attentions['RGB_NI_TI'] = attn

        return results, attentions



############gaijin boq~~~~~~~~~~~~~~
import math

class ImprovedBoQBlock(torch.nn.Module):
    def __init__(self, in_dim, num_queries, nheads=8, layer_idx=0, total_layers=2, prev_num_queries=None):
        super(ImprovedBoQBlock, self).__init__()

        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.num_queries = num_queries

        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=nheads, dim_feedforward=4 * in_dim,
            batch_first=True, dropout=0.)

        # 改进1: 分层查询初始化策略
        # 浅层学习局部特征，深层学习全局特征
        init_scale = 1 * (1 + layer_idx * 0.5)  # 深层查询初始化scale更大
        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim) * init_scale)

        # 改进2: 查询专门化 - 不同层有不同的注意力模式
        self.query_projection = nn.Linear(in_dim, in_dim)

        # 查询尺寸适配器 - 解决不同层查询数量不匹配问题
        if prev_num_queries is not None and prev_num_queries != num_queries:
            self.query_adapter = nn.Linear(prev_num_queries, num_queries)
        else:
            self.query_adapter = None

        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)

        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

        # 改进3: 残差门控机制
        self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x, prev_queries=None):
        B = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)

        # 改进4: 层间查询传递 - 处理尺寸不匹配
        if prev_queries is not None and self.layer_idx > 0:
            # 如果查询数量不匹配，使用适配器
            if self.query_adapter is not None:
                # 转置 -> 适配 -> 转置回来
                adapted_queries = self.query_adapter(prev_queries.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                adapted_queries = prev_queries

            # 融合前一层的查询信息
            alpha = torch.sigmoid(self.gate)
            q = alpha * q + (1 - alpha) * adapted_queries

        # 查询专门化投影
        q = self.query_projection(q)

        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)

        return x, out, attn.detach(), q  # 返回处理后的查询供下一层使用


class AdaptiveBoQ(torch.nn.Module):
    """改进的BoQ网络"""

    def __init__(self, input_dim=512, num_queries=32, num_layers=2, row_dim=32,
                 use_positional_encoding=True):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding
        self.norm_input = torch.nn.LayerNorm(input_dim)

        # 改进5: 可选的位置编码
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(input_dim)

        in_dim = input_dim

        # 改进6: 层特异性的查询数量
        layer_queries = self._get_layer_queries(num_queries, num_layers)

        self.boqs = torch.nn.ModuleList()
        for i in range(num_layers):
            prev_queries = layer_queries[i - 1] if i > 0 else None
            self.boqs.append(
                ImprovedBoQBlock(in_dim, layer_queries[i],
                                 nheads=max(1, in_dim // 64),
                                 layer_idx=i, total_layers=num_layers,
                                 prev_num_queries=prev_queries)
            )

        # 改进7: 自适应特征融合
        total_query_outputs = sum(layer_queries)
        self.adaptive_fusion = AdaptiveFusion(input_dim, total_query_outputs, row_dim)

    def _get_layer_queries(self, base_queries, num_layers):
        """为不同层分配不同数量的查询"""
        if num_layers == 1:
            return [base_queries]

        # 浅层更多查询（细节），深层较少查询（抽象）
        queries_per_layer = []
        for i in range(num_layers):
            ratio = 1.0 - (i / (num_layers - 1)) * 0.3  # 30%递减
            layer_q = max(8, int(base_queries * ratio))
            queries_per_layer.append(layer_q)

        return queries_per_layer

    def forward(self, x):
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        x = self.norm_input(x)

        outs = []
        attns = []
        prev_queries = None

        for i, boq_layer in enumerate(self.boqs):
            x, out, attn, queries = boq_layer(x, prev_queries)
            outs.append(out)
            attns.append(attn)
            prev_queries = queries  # 传递查询到下一层

        # 自适应融合
        final_out = self.adaptive_fusion(outs)

        return final_out, attns


class PositionalEncoding(nn.Module):
    """改进8: 可学习的位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 传统sin/cos位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        # 可学习的位置权重
        self.pos_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pe[:seq_len].unsqueeze(0)
        return x + self.pos_weight * pos_enc


class AdaptiveFusion(nn.Module):
    """改进9: 自适应特征融合模块"""

    def __init__(self, feat_dim, total_queries, output_dim):
        super().__init__()

        self.feat_dim = feat_dim
        self.total_queries = total_queries

        # 注意力权重网络
        self.attention_net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ReLU(),
            nn.Linear(feat_dim // 4, 1)
        )

        # 最终投影
        self.final_proj = nn.Linear(total_queries, output_dim)

    def forward(self, layer_outputs):
        # layer_outputs: list of [B, num_queries_i, feat_dim]

        # 计算每层的重要性权重
        layer_weights = []
        for out in layer_outputs:
            # 全局平均池化 + 注意力权重
            global_feat = torch.mean(out, dim=1)  # [B, feat_dim]
            weight = torch.sigmoid(self.attention_net(global_feat))  # [B, 1]
            layer_weights.append(weight)

        # 加权融合
        weighted_outputs = []
        for out, weight in zip(layer_outputs, layer_weights):
            weighted_out = out * weight.unsqueeze(1)  # [B, num_queries, feat_dim]
            weighted_outputs.append(weighted_out)

        # 拼接所有层的输出
        concat_out = torch.cat(weighted_outputs, dim=1)  # [B, total_queries, feat_dim]

        # 最终投影和归一化
        final_out = self.final_proj(concat_out.permute(0, 2, 1))  # [B, feat_dim, output_dim]
        final_out = final_out.flatten(1)  # [B, feat_dim * output_dim]
        final_out = torch.nn.functional.normalize(final_out, p=2, dim=-1)

        return final_out


# 多模态专用改进
class ModalitySpecificBoQ(nn.Module):
    """改进10: 模态特异性BoQ"""

    def __init__(self, input_dim=512, num_queries=32, num_layers=2, row_dim=32):
        super().__init__()

        # 为不同模态组合设计特异性参数
        self.single_modal_boq = AdaptiveBoQ(input_dim, num_queries, num_layers, row_dim)
        self.dual_modal_boq = AdaptiveBoQ(input_dim, int(num_queries * 1.2), num_layers, row_dim)
        self.triple_modal_boq = AdaptiveBoQ(input_dim, int(num_queries * 1.5), num_layers, row_dim)

        # 模态特异性的查询初始化
        self._init_modality_specific_queries()

    def _init_modality_specific_queries(self):
        """为不同模态初始化特异性查询"""
        # 这里可以加入先验知识，比如RGB关注纹理，TIR关注温度等
        pass

    def forward(self, features_dict):
        results = {}
        attentions = {}

        for modality, feat in features_dict.items():
            feat = feat.permute(1, 0, 2)  # [N, B, D] -> [B, N, D]

            if modality in ['RGB', 'NI', 'TI']:
                out, attn = self.single_modal_boq(feat)
            elif modality in ['RGB_NI', 'RGB_TI', 'NI_TI']:
                out, attn = self.dual_modal_boq(feat)
            else:  # RGB_NI_TI
                out, attn = self.triple_modal_boq(feat)

            results[modality] = out
            attentions[modality] = attn

        return results, attentions




###################多模态xiaorong###########
class HAQNConfig:
    """
    HAQN消融实验配置类 - 控制各组件的开启/关闭
    """

    def __init__(self,
                 enable_hierarchical_queries=True,  # Progressive query learning (层次化查询学习)
                 enable_query_propagation=True,  # Inter-layer query propagation (跨层查询传播)
                 enable_modality_specific=True,  # Modality-specific architectures (模态特异性架构)
                 enable_adaptive_fusion=True):  # Adaptive feature fusion (自适应特征融合)
        self.enable_hierarchical_queries = enable_hierarchical_queries
        self.enable_query_propagation = enable_query_propagation
        self.enable_modality_specific = enable_modality_specific
        self.enable_adaptive_fusion = enable_adaptive_fusion

    def get_ablation_name(self):
        """生成消融实验的名称标识"""
        components = []
        if self.enable_hierarchical_queries: components.append("HQ")
        if self.enable_query_propagation: components.append("QP")
        if self.enable_modality_specific: components.append("MS")
        if self.enable_adaptive_fusion: components.append("AF")
        return "_".join(components) if components else "Baseline"


class ImprovedBoQBlockAblation(torch.nn.Module):
    """BoQ Block with Ablation Interface"""

    def __init__(self, in_dim, num_queries, nheads=8, layer_idx=0, total_layers=2,
                 prev_num_queries=None, ablation_config=None):
        super(ImprovedBoQBlockAblation, self).__init__()

        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.num_queries = num_queries
        self.config = ablation_config if ablation_config is not None else HAQNConfig()

        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=nheads, dim_feedforward=4 * in_dim,
            batch_first=True, dropout=0.)

        # 组件1: Progressive Query Learning (层次化查询学习)
        if self.config.enable_hierarchical_queries:
            # 分层查询初始化策略 - 浅层学习局部特征，深层学习全局特征
            init_scale = 1 * (1 + layer_idx * 0.5)  # 深层查询初始化scale更大
            print(f"[HAQN Ablation] ✓ Layer {layer_idx}: Hierarchical queries enabled with scale {init_scale:.3f}")
        else:
            # 禁用时使用固定初始化
            init_scale = 1
            print(
                f"[HAQN Ablation] ✗ Layer {layer_idx}: Hierarchical queries disabled, using fixed scale {init_scale:.3f}")

        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim) * init_scale)

        # 查询专门化投影
        self.query_projection = nn.Linear(in_dim, in_dim)

        # 组件2: Inter-layer Query Propagation (跨层查询传播)
        if self.config.enable_query_propagation and prev_num_queries is not None and prev_num_queries != num_queries:
            self.query_adapter = nn.Linear(prev_num_queries, num_queries)
            print(
                f"[HAQN Ablation] ✓ Layer {layer_idx}: Query propagation enabled with adapter ({prev_num_queries}->{num_queries})")
        elif self.config.enable_query_propagation:
            self.query_adapter = None
            print(f"[HAQN Ablation] ✓ Layer {layer_idx}: Query propagation enabled (no adapter needed)")
        else:
            self.query_adapter = None
            print(f"[HAQN Ablation] ✗ Layer {layer_idx}: Query propagation disabled")

        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)

        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

        # 残差门控机制 (仅在查询传播启用时使用)
        if self.config.enable_query_propagation:
            self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x, prev_queries=None):
        B = x.size(0)
        x = self.encoder(x)

        q = self.queries.repeat(B, 1, 1)

        # 组件2: 层间查询传递
        if self.config.enable_query_propagation and prev_queries is not None and self.layer_idx > 0:
            # 如果查询数量不匹配，使用适配器
            if self.query_adapter is not None:
                adapted_queries = self.query_adapter(prev_queries.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                adapted_queries = prev_queries

            # 融合前一层的查询信息
            alpha = torch.sigmoid(self.gate)
            q = alpha * q + (1 - alpha) * adapted_queries

        # 查询专门化投影
        q = self.query_projection(q)

        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)

        out, attn = self.cross_attn(q, x, x)
        out = self.norm_out(out)

        return x, out, attn.detach(), q


class PositionalEncodingAblation(nn.Module):
    """可学习的位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)
        self.pos_weight = nn.Parameter(torch.ones(1))

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pe[:seq_len].unsqueeze(0)
        return x + self.pos_weight * pos_enc


class AdaptiveFusionAblation(nn.Module):
    """自适应特征融合模块 (可消融)"""

    def __init__(self, feat_dim, total_queries, output_dim, ablation_config=None):
        super().__init__()

        self.feat_dim = feat_dim
        self.total_queries = total_queries
        self.config = ablation_config if ablation_config is not None else HAQNConfig()

        if self.config.enable_adaptive_fusion:
            # 启用自适应融合：注意力权重网络
            self.attention_net = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
            print("[HAQN Ablation] ✓ Adaptive fusion enabled")
        else:
            print("[HAQN Ablation] ✗ Adaptive fusion disabled (using simple concatenation)")

        # 最终投影
        self.final_proj = nn.Linear(total_queries, output_dim)

    def forward(self, layer_outputs):
        # layer_outputs: list of [B, num_queries_i, feat_dim]

        if self.config.enable_adaptive_fusion:
            # 启用：计算每层的重要性权重
            layer_weights = []
            for out in layer_outputs:
                global_feat = torch.mean(out, dim=1)  # [B, feat_dim]
                weight = torch.sigmoid(self.attention_net(global_feat))  # [B, 1]
                layer_weights.append(weight)

            # 加权融合
            weighted_outputs = []
            for out, weight in zip(layer_outputs, layer_weights):
                weighted_out = out * weight.unsqueeze(1)  # [B, num_queries, feat_dim]
                weighted_outputs.append(weighted_out)
        else:
            # 禁用：简单等权重融合
            weighted_outputs = layer_outputs

        # 拼接所有层的输出
        concat_out = torch.cat(weighted_outputs, dim=1)  # [B, total_queries, feat_dim]

        # 最终投影和归一化
        final_out = self.final_proj(concat_out.permute(0, 2, 1))  # [B, feat_dim, output_dim]
        final_out = final_out.flatten(1)  # [B, feat_dim * output_dim]
        final_out = torch.nn.functional.normalize(final_out, p=2, dim=-1)

        return final_out


class AdaptiveBoQAblation(torch.nn.Module):
    """AdaptiveBoQ with Ablation Study Interface"""

    def __init__(self, input_dim=512, num_queries=32, num_layers=2, row_dim=32,
                 use_positional_encoding=True, ablation_config=None):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding
        self.norm_input = torch.nn.LayerNorm(input_dim)
        self.config = ablation_config if ablation_config is not None else HAQNConfig()

        print(f"[HAQN Ablation] Using configuration: {self.config.get_ablation_name()}")

        if use_positional_encoding:
            self.pos_encoding = PositionalEncodingAblation(input_dim)

        in_dim = input_dim

        # 组件1: 层特异性的查询数量 (层次化查询学习)
        if self.config.enable_hierarchical_queries:
            layer_queries = self._get_layer_queries_hierarchical(num_queries, num_layers)
            print(f"[HAQN Ablation] ✓ Hierarchical queries: {layer_queries}")
        else:
            layer_queries = [num_queries] * num_layers  # 所有层使用相同查询数量
            print(f"[HAQN Ablation] ✗ Hierarchical queries disabled: {layer_queries}")

        self.layer_queries = layer_queries
        self.boqs = torch.nn.ModuleList()
        for i in range(num_layers):
            prev_queries = layer_queries[i - 1] if i > 0 else None
            self.boqs.append(
                ImprovedBoQBlockAblation(in_dim, layer_queries[i],
                                         nheads=max(1, in_dim // 64),
                                         layer_idx=i, total_layers=num_layers,
                                         prev_num_queries=prev_queries,
                                         ablation_config=self.config)
            )

        # 组件4: 自适应特征融合
        total_query_outputs = sum(layer_queries)
        self.adaptive_fusion = AdaptiveFusionAblation(input_dim, total_query_outputs, row_dim, self.config)

    def _get_layer_queries_hierarchical(self, base_queries, num_layers):
        """层次化查询分配 (组件1)"""
        if num_layers == 1:
            return [base_queries]

        # 浅层更多查询（细节），深层较少查询（抽象）
        queries_per_layer = []
        for i in range(num_layers):
            ratio = 1.0 - (i / (num_layers - 1)) * 0.3  # 30%递减
            layer_q = max(8, int(base_queries * ratio))
            queries_per_layer.append(layer_q)

        return queries_per_layer

    def forward(self, x):
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        x = self.norm_input(x)

        outs = []
        attns = []
        prev_queries = None

        for i, boq_layer in enumerate(self.boqs):
            x, out, attn, queries = boq_layer(x, prev_queries)
            outs.append(out)
            attns.append(attn)

            # 组件2: 查询传播控制
            if self.config.enable_query_propagation:
                prev_queries = queries  # 传递查询到下一层
            else:
                prev_queries = None  # 不传递查询信息

        # 组件4: 自适应融合
        final_out = self.adaptive_fusion(outs)

        return final_out, attns

    def get_ablation_summary(self):
        """获取当前消融配置的总结"""
        enabled = []
        disabled = []

        components = [
            ("HQ", "Hierarchical Queries", self.config.enable_hierarchical_queries),
            ("QP", "Query Propagation", self.config.enable_query_propagation),
            ("MS", "Modality Specific", self.config.enable_modality_specific),
            ("AF", "Adaptive Fusion", self.config.enable_adaptive_fusion)
        ]

        for abbr, full_name, enabled_flag in components:
            if enabled_flag:
                enabled.append(f"{abbr} ({full_name})")
            else:
                disabled.append(f"{abbr} ({full_name})")

        summary = f"""
=== HAQN Ablation Configuration ===
Configuration Name: {self.config.get_ablation_name()}
Layer Queries: {self.layer_queries}
Enabled Components: {', '.join(enabled) if enabled else 'None'}
Disabled Components: {', '.join(disabled) if disabled else 'None'}
===================================
        """
        return summary.strip()


class ModalitySpecificBoQAblation(nn.Module):
    """模态特异性BoQ (可消融)"""

    def __init__(self, input_dim=512, num_queries=32, num_layers=2, row_dim=32, ablation_config=None):
        super().__init__()

        self.config = ablation_config if ablation_config is not None else HAQNConfig()

        if self.config.enable_modality_specific:
            # 启用模态特异性：为不同模态组合设计特异性参数
            self.single_modal_boq = AdaptiveBoQAblation(input_dim, num_queries, num_layers, row_dim, True, self.config)
            self.dual_modal_boq = AdaptiveBoQAblation(input_dim, int(num_queries * 1.2), num_layers, row_dim, True,
                                                      self.config)
            self.triple_modal_boq = AdaptiveBoQAblation(input_dim, int(num_queries * 1.5), num_layers, row_dim, True,
                                                        self.config)
            print("[HAQN Ablation] ✓ Modality-specific architectures enabled")
        else:
            # 禁用模态特异性：所有模态使用相同架构
            shared_boq = AdaptiveBoQAblation(input_dim, num_queries, num_layers, row_dim, True, self.config)
            self.single_modal_boq = shared_boq
            self.dual_modal_boq = shared_boq
            self.triple_modal_boq = shared_boq
            print("[HAQN Ablation] ✗ Modality-specific architectures disabled (using shared architecture)")

    def forward(self, features_dict):
        results = {}
        attentions = {}

        for modality, feat in features_dict.items():
            feat = feat.permute(1, 0, 2)  # [N, B, D] -> [B, N, D]

            if self.config.enable_modality_specific:
                # 启用模态特异性：使用专门的架构
                if modality in ['RGB', 'NI', 'TI']:
                    out, attn = self.single_modal_boq(feat)
                elif modality in ['RGB_NI', 'RGB_TI', 'NI_TI']:
                    out, attn = self.dual_modal_boq(feat)
                else:  # RGB_NI_TI
                    out, attn = self.triple_modal_boq(feat)
            else:
                # 禁用模态特异性：所有模态使用相同架构
                out, attn = self.single_modal_boq(feat)

            results[modality] = out
            attentions[modality] = attn

        return results, attentions


# 创建所有可能的HAQN消融配置
def create_haqn_ablation_configs():
    """创建所有可能的HAQN消融配置"""
    configs = {}

    # 完整模型
    configs['Full'] = HAQNConfig(True, True, True, True)

    # 单组件消融
    configs['w/o_HQ'] = HAQNConfig(False, True, True, True)  # 移除层次化查询
    configs['w/o_QP'] = HAQNConfig(True, False, True, True)  # 移除查询传播
    configs['w/o_MS'] = HAQNConfig(True, True, False, True)  # 移除模态特异性
    configs['w/o_AF'] = HAQNConfig(True, True, True, False)  # 移除自适应融合

    # 双组件消融
    configs['w/o_HQ_QP'] = HAQNConfig(False, False, True, True)
    configs['w/o_HQ_MS'] = HAQNConfig(False, True, False, True)
    configs['w/o_HQ_AF'] = HAQNConfig(False, True, True, False)
    configs['w/o_QP_MS'] = HAQNConfig(True, False, False, True)
    configs['w/o_QP_AF'] = HAQNConfig(True, False, True, False)
    configs['w/o_MS_AF'] = HAQNConfig(True, True, False, False)

    # 基线模型
    configs['Baseline'] = HAQNConfig(False, False, False, False)

    return configs


# 使用示例
# if __name__ == "__main__":
#     # 创建消融配置
#     ablation_configs = create_haqn_ablation_configs()
#
#     # 示例：创建移除查询传播的模型
#     config = ablation_configs['w/o_QP']
#     model = AdaptiveBoQAblation(
#         input_dim=512, num_queries=32, num_layers=3, row_dim=1,
#         ablation_config=config
#     )
#
#     print(model.get_ablation_summary())
#
#     # 测试模态特异性BoQ
#     modality_model = ModalitySpecificBoQAblation(
#         input_dim=512, num_queries=64, num_layers=4, row_dim=1,
#         ablation_config=config
#     )
#
#     print("\n" + "=" * 50)
#     print("Modality-Specific BoQ Configuration:")
#     print("=" * 50)



################## deform ablation ########


class EDAConfig:
    """
    消融实验配置类 - 控制EDA各组件的开启/关闭
    """

    def __init__(self,
                 enable_amw=True,  # Adaptive Modal Weighting
                 enable_msof=True,  # Multi-Scale Offset Fusion
                 enable_lts=True,  # Learnable Temperature Scaling
                 enable_roc=True):  # Residual Offset Connection
        self.enable_amw = enable_amw
        self.enable_msof = enable_msof
        self.enable_lts = enable_lts
        self.enable_roc = enable_roc

    def get_ablation_name(self):
        """生成消融实验的名称标识"""
        components = []
        if self.enable_amw: components.append("AMW")
        if self.enable_msof: components.append("MSOF")
        if self.enable_lts: components.append("LTS")
        if self.enable_roc: components.append("ROC")
        return "_".join(components) if components else "Baseline"


class DAttentionEnhancedAblation(nn.Module):
    """
    Enhanced Deformable Attention with Ablation Study Interface
    """

    def __init__(
            self, q_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, ksize, share,
            ablation_config=None  # 新增消融配置参数
    ):

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor

        self.ksize = ksize
        self.stride = stride
        kk = self.ksize
        pad_size = 0
        self.share_offset = share

        # 消融配置
        self.config = ablation_config if ablation_config is not None else EDAConfig()
        print(f"[EDA Ablation] Using configuration: {self.config.get_ablation_name()}")

        # ===== 组件1: Adaptive Modal Weighting (AMW) =====
        if self.config.enable_amw:
            self.modal_weights = nn.Parameter(torch.ones(3))
            self.modal_gate = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.n_group_channels // 4, 3, 1),
                nn.Sigmoid()
            )
            print("[EDA Ablation] ✓ AMW (Adaptive Modal Weighting) enabled")
        else:
            print("[EDA Ablation] ✗ AMW (Adaptive Modal Weighting) disabled")

        # ===== 组件2: Multi-Scale Offset Fusion (MSOF) =====
        if self.config.enable_msof:
            self.multi_scale_levels = 3
            self.scale_weights = nn.Parameter(torch.ones(self.multi_scale_levels))
            print("[EDA Ablation] ✓ MSOF (Multi-Scale Offset Fusion) enabled")
        else:
            print("[EDA Ablation] ✗ MSOF (Multi-Scale Offset Fusion) disabled")

        # ===== 组件3: Learnable Temperature Scaling (LTS) =====
        if self.config.enable_lts:
            self.temperature = nn.Parameter(torch.ones(1))
            print("[EDA Ablation] ✓ LTS (Learnable Temperature Scaling) enabled")
        else:
            print("[EDA Ablation] ✗ LTS (Learnable Temperature Scaling) disabled")

        # ===== 组件4: Residual Offset Connection (ROC) =====
        if self.config.enable_roc:
            self.offset_residual_weight = nn.Parameter(torch.tensor(0.1))
            print("[EDA Ablation] ✓ ROC (Residual Offset Connection) enabled")
        else:
            print("[EDA Ablation] ✗ ROC (Residual Offset Connection) disabled")

        # 基础偏移网络
        if self.share_offset:
            # 主要偏移网络
            self.conv_offset = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            )

            # 多尺度偏移网络 (仅在MSOF启用时创建)
            if self.config.enable_msof:
                self.conv_offset_coarse = nn.Sequential(
                    nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                    nn.GELU(),
                    nn.Conv2d(self.n_group_channels, self.n_group_channels, kk + 2, stride, pad_size + 1,
                              groups=self.n_group_channels),
                    nn.GELU(),
                    nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
                )

                self.conv_offset_fine = nn.Sequential(
                    nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                    nn.GELU(),
                    nn.Conv2d(self.n_group_channels, self.n_group_channels, max(kk - 2, 1), stride, 0,
                              groups=self.n_group_channels),
                    nn.GELU(),
                    nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
                )
        else:
            # 非共享偏移网络
            self.conv_offset_r = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            # ... 其他模态的网络

        # 基础投影层
        self.proj_q = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def adaptive_modal_weighting(self, x, y, z):
        """
        组件1: 自适应模态权重 (可消融)
        """
        if not self.config.enable_amw:
            # AMW禁用时，使用均等权重
            return x, y, z

        # 计算全局模态权重
        modal_weights = F.softmax(self.modal_weights, dim=0)

        # 计算局部门控权重
        concat_features = torch.cat([x, y, z], dim=1)
        B, _, H, W = concat_features.shape
        avg_pool = F.adaptive_avg_pool2d(concat_features, 1)
        gate_weights = self.modal_gate(avg_pool)  # [B, 3, 1, 1]

        # 结合全局和局部权重
        combined_weights = modal_weights.view(1, 3, 1, 1) * gate_weights
        combined_weights = F.softmax(combined_weights, dim=1)

        # 应用权重
        weighted_x = x * combined_weights[:, 0:1]
        weighted_y = y * combined_weights[:, 1:2]
        weighted_z = z * combined_weights[:, 2:3]

        return weighted_x, weighted_y, weighted_z

    def multi_scale_offset_fusion(self, data, reference):
        """
        组件2: 多尺度偏移融合 (可消融)
        """
        if not self.share_offset:
            return self.off_set_unshared_enhanced(data, reference)

        data = einops.rearrange(data, 'b (g c) h w -> (b g) c h w',
                                g=self.n_groups, c=3 * self.n_group_channels)

        # 计算主要偏移
        offset_main = self.conv_offset(data)

        if self.config.enable_msof:
            # MSOF启用：多尺度偏移融合
            offset_coarse = self.conv_offset_coarse(data)
            offset_fine = self.conv_offset_fine(data)

            # 调整尺寸
            if offset_fine.shape != offset_main.shape:
                offset_fine = F.interpolate(offset_fine, size=offset_main.shape[2:],
                                            mode='bilinear', align_corners=True)
            if offset_coarse.shape != offset_main.shape:
                offset_coarse = F.interpolate(offset_coarse, size=offset_main.shape[2:],
                                              mode='bilinear', align_corners=True)

            # 融合多尺度偏移
            scale_weights = F.softmax(self.scale_weights, dim=0)
            offset = (scale_weights[0] * offset_main +
                      scale_weights[1] * offset_coarse +
                      scale_weights[2] * offset_fine)
        else:
            # MSOF禁用：仅使用主要偏移
            offset = offset_main

        Hk, Wk = offset.size(2), offset.size(3)

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)],
                                        device=data.device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')

        if self.config.enable_roc:
            # ROC启用：残差偏移连接
            residual_offset = offset * self.offset_residual_weight
            final_offset = offset + residual_offset
        else:
            # ROC禁用：直接使用偏移
            final_offset = offset

        pos_x = (final_offset + reference).clamp(-1., +1.)
        pos_y = (final_offset + reference).clamp(-1., +1.)
        pos_z = (final_offset + reference).clamp(-1., +1.)

        return pos_x, pos_y, pos_z, Hk, Wk

    def apply_temperature_scaling(self, attn):
        """
        组件3: 可学习温度缩放 (可消融)
        """
        if self.config.enable_lts:
            # LTS启用：使用可学习温度
            return attn.mul(self.scale * self.temperature)
        else:
            # LTS禁用：使用固定缩放
            return attn.mul(self.scale)

    @torch.no_grad()
    def _get_ref_points(self, H_in, W_in, B, kernel_size, stride, dtype, device):
        """生成参考点"""
        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1

        center_y = torch.arange(H_out, dtype=dtype, device=device) * stride + (kernel_size // 2)
        center_x = torch.arange(W_out, dtype=dtype, device=device) * stride + (kernel_size // 2)

        ref_y, ref_x = torch.meshgrid(center_y, center_x, indexing='ij')
        ref = torch.stack((ref_y, ref_x), dim=-1)

        ref[..., 1].div_(W_in - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_in - 1.0).mul_(2.0).sub_(1.0)

        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def off_set_unshared_enhanced(self, data, reference):
        """增强版非共享偏移计算"""
        x, y, z = data.chunk(3, dim=1)
        x = einops.rearrange(x, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)

        offset_r = self.conv_offset_r(x)
        # 简化处理其他模态
        Hk, Wk = offset_r.size(2), offset_r.size(3)

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset_r = offset_r.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset_r = einops.rearrange(offset_r, 'b p h w -> b h w p')

        if self.config.enable_roc:
            offset_r = offset_r + offset_r * self.offset_residual_weight

        pos_x = (offset_r + reference).clamp(-1., +1.)
        pos_y = (offset_r + reference).clamp(-1., +1.)
        pos_z = (offset_r + reference).clamp(-1., +1.)
        return pos_x, pos_y, pos_z, Hk, Wk

    def forward(self, x, y, z, writer=None, epoch=None, img_path=None, text=''):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # 组件1: 应用自适应模态权重
        x, y, z = self.adaptive_modal_weighting(x, y, z)

        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)

        # 组件2&4: 多尺度偏移融合 + 残差连接
        pos_x, pos_y, pos_z, Hk, Wk = self.multi_scale_offset_fusion(data, reference)

        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],
            mode='bilinear', align_corners=True)

        sampled_x, sampled_y, sampled_z = [
            t.reshape(B, C, 1, n_sample).squeeze(2).permute(2, 0, 1)
            for t in [sampled_x, sampled_y, sampled_z]
        ]

        return sampled_x, sampled_y, sampled_z

    def forwardOld(self, query, x, y, z, writer=None, epoch=None, img_path=None, text=''):
        """带交叉注意力的前向传播（用于消融LTS组件）"""
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        dtype, device = x.dtype, x.device

        # 应用自适应模态权重
        x, y, z = self.adaptive_modal_weighting(x, y, z)

        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)

        pos_x, pos_y, pos_z, Hk, Wk = self.multi_scale_offset_fusion(data, reference)

        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],
            mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, 1, n_sample)
        sampled_y = sampled_y.reshape(B, C, 1, n_sample)
        sampled_z = sampled_z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)

        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)

        # 组件3: 可学习温度缩放
        attn = self.apply_temperature_scaling(attn)
        attn = F.softmax(attn, dim=2)

        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def get_ablation_summary(self):
        """获取当前消融配置的总结"""
        enabled = []
        disabled = []

        components = [
            ("AMW", "Adaptive Modal Weighting", self.config.enable_amw),
            ("MSOF", "Multi-Scale Offset Fusion", self.config.enable_msof),
            ("LTS", "Learnable Temperature Scaling", self.config.enable_lts),
            ("ROC", "Residual Offset Connection", self.config.enable_roc)
        ]

        for abbr, full_name, enabled_flag in components:
            if enabled_flag:
                enabled.append(f"{abbr} ({full_name})")
            else:
                disabled.append(f"{abbr} ({full_name})")

        summary = f"""
=== EDA Ablation Configuration ===
Configuration Name: {self.config.get_ablation_name()}
Enabled Components: {', '.join(enabled) if enabled else 'None'}
Disabled Components: {', '.join(disabled) if disabled else 'None'}
================================
        """
        return summary.strip()


# 使用示例
def create_eda_ablation_configs():
    """创建所有可能的消融配置"""
    configs = {}

    # 完整模型
    configs['Full'] = EDAConfig(True, True, True, True)

    # 单组件消融 (移除一个组件)
    configs['w/o_AMW'] = EDAConfig(False, True, True, True)
    configs['w/o_MSOF'] = EDAConfig(True, False, True, True)
    configs['w/o_LTS'] = EDAConfig(True, True, False, True)
    configs['w/o_ROC'] = EDAConfig(True, True, True, False)

    # 双组件消融 (移除两个组件)
    configs['w/o_AMW_MSOF'] = EDAConfig(False, False, True, True)
    configs['w/o_AMW_LTS'] = EDAConfig(False, True, False, True)
    configs['w/o_AMW_ROC'] = EDAConfig(False, True, True, False)
    configs['w/o_MSOF_LTS'] = EDAConfig(True, False, False, True)
    configs['w/o_MSOF_ROC'] = EDAConfig(True, False, True, False)
    configs['w/o_LTS_ROC'] = EDAConfig(True, True, False, False)

    # 基线模型 (移除所有组件)
    configs['Baseline'] = EDAConfig(False, False, False, False)

    return configs


# 使用示例
# if __name__ == "__main__":
#     # 创建消融配置
#     ablation_configs = create_eda_ablation_configs()
#
#     # 示例：创建移除AMW组件的模型
#     config = ablation_configs['w/o_AMW']
#     model = DAttentionEnhancedAblation(
#         q_size=(16, 8), n_heads=1, n_head_channels=512, n_groups=1,
#         attn_drop=0.0, proj_drop=0.0, stride=2,
#         offset_range_factor=5.0, ksize=4, share=True,
#         ablation_config=config
#     )
#
#     print(model.get_ablation_summary())

#################new ablation ##############

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


# ===== 新的配置类 =====
class SimplifiedEDAConfig:
    """简化的EDA消融配置"""

    def __init__(self,
                 enable_msof=True,  # Multi-Scale Offset Fusion (保留)
                 enable_cma=True,  # Channel-wise Modal Attention (新)
                 enable_sal=True):  # Spatial-aware Adaptive Learning (新)
        self.enable_msof = enable_msof
        self.enable_cma = enable_cma
        self.enable_sal = enable_sal

    def get_ablation_name(self):
        components = []
        if self.enable_msof: components.append("MSOF")
        if self.enable_cma: components.append("CMA")
        if self.enable_sal: components.append("SAL")
        return "_".join(components) if components else "Baseline"


class SimplifiedHAQNConfig:
    """简化的HAQN消融配置"""

    def __init__(self,
                 enable_hq=True,  # Hierarchical Queries (保留简化版)
                 enable_af=True,  # Adaptive Fusion (保留)
                 enable_qrm=True):  # Query Refinement Mechanism (新)
        self.enable_hq = enable_hq
        self.enable_af = enable_af
        self.enable_qrm = enable_qrm

    def get_ablation_name(self):
        components = []
        if self.enable_hq: components.append("HQ")
        if self.enable_af: components.append("AF")
        if self.enable_qrm: components.append("QRM")
        return "_".join(components) if components else "Baseline"


# ===== 简化的EDA模块 (保持不变) =====
class SimplifiedEDA(nn.Module):
    """简化的增强可变形注意力模块"""

    def __init__(self, q_size, n_heads, n_head_channels, n_groups,
                 attn_drop, proj_drop, stride, offset_range_factor, ksize, share,
                 ablation_config=None):
        super().__init__()

        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.stride = stride
        self.share_offset = share

        self.config = ablation_config if ablation_config is not None else SimplifiedEDAConfig()
        print(f"[Simplified EDA] Configuration: {self.config.get_ablation_name()}")

        # 基础偏移网络
        self.conv_offset_main = nn.Sequential(
            nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, self.n_group_channels, ksize, stride, 0,
                      groups=self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
        )

        # 创新1: Multi-Scale Offset Fusion (MSOF) - 保留
        if self.config.enable_msof:
            self.conv_offset_coarse = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, ksize + 2, stride, 1,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, bias=False),
            )
            self.conv_offset_fine = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, max(ksize - 2, 1), stride, 0,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, bias=False),
            )
            # 简化的尺度权重
            self.scale_weights = nn.Parameter(torch.tensor([0.5, 1.0, 0.3]))
            print("[Simplified EDA] ✓ MSOF enabled")
        else:
            print("[Simplified EDA] ✗ MSOF disabled")

        # 创新2: Channel-wise Modal Attention (CMA) - 新设计
        if self.config.enable_cma:
            self.channel_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.n_group_channels // 4, 3 * self.n_group_channels, 1),
                nn.Sigmoid()
            )
            print("[Simplified EDA] ✓ CMA enabled")
        else:
            print("[Simplified EDA] ✗ CMA disabled")

        # 创新3: Spatial-aware Adaptive Learning (SAL) - 新设计
        if self.config.enable_sal:
            self.spatial_adapter = nn.Conv2d(2, 2, 3, 1, 1)  # 对偏移进行空间自适应
            self.spatial_gate = nn.Parameter(torch.tensor(0.1))  # 可学习的空间权重
            print("[Simplified EDA] ✓ SAL enabled")
        else:
            print("[Simplified EDA] ✗ SAL disabled")

        # 基础网络层
        self.proj_q = nn.Conv2d(self.nc, self.nc, 1)
        self.proj_k = nn.Conv2d(self.nc, self.nc, 1)
        self.proj_v = nn.Conv2d(self.nc, self.nc, 1)
        self.proj_out = nn.Conv2d(self.nc, self.nc, 1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def channel_wise_modal_attention(self, x, y, z):
        """创新2: 通道级模态注意力 - 简化版AMW"""
        if not self.config.enable_cma:
            return torch.cat([x, y, z], dim=1)

        concat_features = torch.cat([x, y, z], dim=1)
        channel_weights = self.channel_attention(concat_features)

        # 直接应用通道注意力
        weighted_features = concat_features * channel_weights
        return weighted_features

    def multi_scale_offset_fusion(self, data):
        """创新1: 多尺度偏移融合 - 保留并简化"""
        data = einops.rearrange(data, 'b (g c) h w -> (b g) c h w',
                                g=self.n_groups, c=3 * self.n_group_channels)

        offset_main = self.conv_offset_main(data)

        if self.config.enable_msof:
            offset_coarse = self.conv_offset_coarse(data)
            offset_fine = self.conv_offset_fine(data)

            # 尺寸对齐
            if offset_coarse.shape != offset_main.shape:
                offset_coarse = F.interpolate(offset_coarse, size=offset_main.shape[2:],
                                              mode='bilinear', align_corners=True)
            if offset_fine.shape != offset_main.shape:
                offset_fine = F.interpolate(offset_fine, size=offset_main.shape[2:],
                                            mode='bilinear', align_corners=True)

            # 简化的权重融合
            weights = F.softmax(self.scale_weights, dim=0)
            offset = weights[0] * offset_coarse + weights[1] * offset_main + weights[2] * offset_fine
        else:
            offset = offset_main

        return offset

    def spatial_aware_learning(self, offset):
        """创新3: 空间感知自适应学习"""
        if not self.config.enable_sal:
            return offset

        # 空间自适应调整
        adapted_offset = self.spatial_adapter(offset)

        # 门控融合
        final_offset = offset + self.spatial_gate * adapted_offset
        return final_offset

    @torch.no_grad()
    def _get_ref_points(self, H_in, W_in, B, kernel_size, stride, dtype, device):
        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1

        center_y = torch.arange(H_out, dtype=dtype, device=device) * stride + (kernel_size // 2)
        center_x = torch.arange(W_out, dtype=dtype, device=device) * stride + (kernel_size // 2)

        ref_y, ref_x = torch.meshgrid(center_y, center_x, indexing='ij')
        ref = torch.stack((ref_y, ref_x), dim=-1)

        ref[..., 1].div_(W_in - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_in - 1.0).mul_(2.0).sub_(1.0)

        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, x, y, z):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        # 创新2: 通道级模态注意力
        data = self.channel_wise_modal_attention(x, y, z)

        # 创新1: 多尺度偏移融合
        offset = self.multi_scale_offset_fusion(data)

        # 创新3: 空间感知学习
        offset = self.spatial_aware_learning(offset)

        # 标准化偏移
        Hk, Wk = offset.size(2), offset.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)],
                                        device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        # 生成参考点
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')

        # 可变形采样
        pos = (offset + reference).clamp(-1., +1.)
        n_sample = Hk * Wk

        sampled_x = F.grid_sample(x.reshape(B * self.n_groups, self.n_group_channels, H, W),
                                  pos[..., (1, 0)], mode='bilinear', align_corners=True)
        sampled_y = F.grid_sample(y.reshape(B * self.n_groups, self.n_group_channels, H, W),
                                  pos[..., (1, 0)], mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(z.reshape(B * self.n_groups, self.n_group_channels, H, W),
                                  pos[..., (1, 0)], mode='bilinear', align_corners=True)

        sampled_x, sampled_y, sampled_z = [
            t.reshape(B, C, 1, n_sample).squeeze(2).permute(2, 0, 1)
            for t in [sampled_x, sampled_y, sampled_z]
        ]

        return sampled_x, sampled_y, sampled_z

    def get_ablation_summary(self):
        enabled = []
        disabled = []

        components = [
            ("MSOF", "Multi-Scale Offset Fusion", self.config.enable_msof),
            ("CMA", "Channel-wise Modal Attention", self.config.enable_cma),
            ("SAL", "Spatial-aware Adaptive Learning", self.config.enable_sal)
        ]

        for abbr, full_name, enabled_flag in components:
            if enabled_flag:
                enabled.append(f"{abbr} ({full_name})")
            else:
                disabled.append(f"{abbr} ({full_name})")

        return f"""
=== Simplified EDA Configuration ===
Configuration: {self.config.get_ablation_name()}
Enabled: {', '.join(enabled) if enabled else 'None'}
Disabled: {', '.join(disabled) if disabled else 'None'}
================================
        """.strip()


# ===== 修改后的HAQN模块 - 适应你的接口 =====
class SimplifiedHAQN(nn.Module):
    """简化的层次化自适应查询网络 - 修改以适应特征字典输入"""

    def __init__(self, input_dim=512, num_queries=32, num_layers=3, row_dim=1,
                 ablation_config=None):
        super().__init__()

        self.config = ablation_config if ablation_config is not None else SimplifiedHAQNConfig()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.row_dim = row_dim

        print(f"[Simplified HAQN] Configuration: {self.config.get_ablation_name()}")

        # 创新1: Hierarchical Queries (HQ) - 简化版
        if self.config.enable_hq:
            # 简化的层次查询分配
            base_queries = [num_queries, max(16, num_queries // 2), max(8, num_queries // 4)]
            self.layer_queries = base_queries[:num_layers]
            print(f"[Simplified HAQN] ✓ HQ enabled: {self.layer_queries}")
        else:
            self.layer_queries = [num_queries] * num_layers
            print(f"[Simplified HAQN] ✗ HQ disabled: {self.layer_queries}")

        # 为每种模态组合创建独立的处理分支
        self.modality_keys = ['RGB', 'NI', 'TI', 'RGB_NI', 'RGB_TI', 'NI_TI', 'RGB_NI_TI']

        # 为每种模态组合创建查询和注意力层
        self.modality_modules = nn.ModuleDict()

        # 创新2: Adaptive Fusion (AF) - 为每个模态创建层权重参数
        if self.config.enable_af:
            self.layer_weights = nn.ParameterDict({
                key: nn.Parameter(torch.ones(num_layers))
                for key in self.modality_keys
            })
            print("[Simplified HAQN] ✓ AF enabled")
        else:
            print("[Simplified HAQN] ✗ AF disabled")

        # 创新3: Query Refinement Mechanism (QRM)
        if self.config.enable_qrm:
            print("[Simplified HAQN] ✓ QRM enabled")
        else:
            print("[Simplified HAQN] ✗ QRM disabled")

        for key in self.modality_keys:
            modules = nn.ModuleDict()

            # 查询参数
            modules['queries'] = nn.ParameterList([
                nn.Parameter(torch.randn(1, nq, input_dim) * 0.02)
                for nq in self.layer_queries
            ])

            # 自注意力层
            modules['self_attns'] = nn.ModuleList([
                nn.MultiheadAttention(input_dim, 8, batch_first=True)
                for _ in self.layer_queries
            ])

            # 交叉注意力层
            modules['cross_attns'] = nn.ModuleList([
                nn.MultiheadAttention(input_dim, 8, batch_first=True)
                for _ in self.layer_queries
            ])

            # 归一化层
            modules['norms1'] = nn.ModuleList([
                nn.LayerNorm(input_dim) for _ in self.layer_queries
            ])

            modules['norms2'] = nn.ModuleList([
                nn.LayerNorm(input_dim) for _ in self.layer_queries
            ])

            # 创新3: Query Refinement Mechanism (QRM)
            if self.config.enable_qrm:
                modules['query_refinement'] = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, input_dim // 2),
                        nn.ReLU(),
                        nn.Linear(input_dim // 2, input_dim),
                        nn.Dropout(0.1)
                    ) for _ in range(num_layers)
                ])

            # 最终投影层
            total_queries = sum(self.layer_queries)
            modules['final_proj'] = nn.Linear(total_queries, row_dim)

            self.modality_modules[key] = modules

        print("[Simplified HAQN] ✓ Created modules for all modality combinations")

    def query_refinement_mechanism(self, queries, modules, layer_idx):
        """创新3: 查询精炼机制"""
        if not self.config.enable_qrm:
            return queries

        # 通过MLP精炼查询表示
        refined = modules['query_refinement'][layer_idx](queries)
        # 残差连接
        return queries + 0.1 * refined

    def adaptive_fusion(self, layer_outputs, modules, modality_key):
        """创新2: 自适应特征融合"""
        if self.config.enable_af:
            # 学习每层的重要性权重
            layer_weights = F.softmax(self.layer_weights[modality_key], dim=0)

            # 加权融合
            weighted_outputs = []
            for i, out in enumerate(layer_outputs):
                weighted = out * layer_weights[i]
                weighted_outputs.append(weighted)
        else:
            # 简单等权重
            weighted_outputs = layer_outputs

        # 拼接并投影
        concat_out = torch.cat(weighted_outputs, dim=1)
        final_out = modules['final_proj'](concat_out.permute(0, 2, 1))
        final_out = final_out.flatten(1)
        final_out = F.normalize(final_out, p=2, dim=-1)

        return final_out

    def process_modality(self, features, key):
        """处理单个模态组合"""
        # features: [seq_len, batch, dim]
        features = features.permute(1, 0, 2)  # [batch, seq_len, dim]
        B = features.size(0)

        modules = self.modality_modules[key]
        layer_outputs = []
        attentions = []

        for i in range(len(self.layer_queries)):
            # 获取查询
            queries = modules['queries'][i].repeat(B, 1, 1)

            # 创新3: 查询精炼
            if self.config.enable_qrm:
                queries = self.query_refinement_mechanism(queries, modules, i)

            # 自注意力
            queries_refined, self_attn = modules['self_attns'][i](
                queries, queries, queries, need_weights=True
            )
            queries_refined = queries + queries_refined
            queries_refined = modules['norms1'][i](queries_refined)

            # 交叉注意力
            output, cross_attn = modules['cross_attns'][i](
                queries_refined, features, features, need_weights=True
            )
            output = queries_refined + output
            output = modules['norms2'][i](output)

            layer_outputs.append(output)
            attentions.append({
                'self_attn': self_attn,
                'cross_attn': cross_attn
            })

        # 创新2: 自适应融合
        final_output = self.adaptive_fusion(layer_outputs, modules, key)

        return final_output, attentions

    def forward(self, features_dict):
        """
        输入:
            features_dict: 包含不同模态组合的字典
                - 'RGB': RGB特征 [seq_len, batch, dim]
                - 'NI': NI特征 [seq_len, batch, dim]
                - 'TI': TI特征 [seq_len, batch, dim]
                - 'RGB_NI': RGB+NI组合特征
                - 'RGB_TI': RGB+TI组合特征
                - 'NI_TI': NI+TI组合特征
                - 'RGB_NI_TI': RGB+NI+TI组合特征

        输出:
            results: 处理后的特征字典
            attentions: 注意力权重字典
        """
        results = {}
        attentions = {}

        # 处理每种模态组合
        for key in self.modality_keys:
            if key in features_dict:
                result, attn = self.process_modality(features_dict[key], key)
                results[key] = result
                attentions[key] = attn

        return results, attentions

    def get_ablation_summary(self):
        enabled = []
        disabled = []

        components = [
            ("HQ", "Hierarchical Queries", self.config.enable_hq),
            ("AF", "Adaptive Fusion", self.config.enable_af),
            ("QRM", "Query Refinement Mechanism", self.config.enable_qrm)
        ]

        for abbr, full_name, enabled_flag in components:
            if enabled_flag:
                enabled.append(f"{abbr} ({full_name})")
            else:
                disabled.append(f"{abbr} ({full_name})")

        return f"""
=== Simplified HAQN Configuration ===
Configuration: {self.config.get_ablation_name()}
Layer Queries: {self.layer_queries}
Enabled: {', '.join(enabled) if enabled else 'None'}
Disabled: {', '.join(disabled) if disabled else 'None'}
==================================
        """.strip()


# ===== 创建消融配置 =====
def create_simplified_ablation_configs():
    """创建简化的消融实验配置"""
    eda_configs = {}
    haqn_configs = {}
    # ./runrgbnt201.sh 0 newablation_womsof_full


    # EDA配置
    eda_configs['Full'] = SimplifiedEDAConfig(True, True, True)
    eda_configs['w/o_MSOF'] = SimplifiedEDAConfig(False, True, True)
    eda_configs['w/o_CMA'] = SimplifiedEDAConfig(True, False, True)
    eda_configs['w/o_SAL'] = SimplifiedEDAConfig(True, True, False)
    eda_configs['MSOF_only'] = SimplifiedEDAConfig(True, False, False)
    eda_configs['Baseline'] = SimplifiedEDAConfig(False, False, False)

    # HAQN配置
    haqn_configs['Full'] = SimplifiedHAQNConfig(True, True, True)
    haqn_configs['w/o_HQ'] = SimplifiedHAQNConfig(False, True, True)
    haqn_configs['w/o_AF'] = SimplifiedHAQNConfig(True, False, True)
    haqn_configs['w/o_QRM'] = SimplifiedHAQNConfig(True, True, False)
    haqn_configs['AF_only'] = SimplifiedHAQNConfig(False, True, False)
    haqn_configs['Baseline'] = SimplifiedHAQNConfig(False, False, False)

    return eda_configs, haqn_configs
# ===== 使用示例 =====
# if __name__ == "__main__":
#     # 创建消融配置
#     eda_configs, haqn_configs = create_simplified_ablation_configs()
#
#     # 测试EDA模块
#     print("=" * 60)
#     print("Testing Simplified EDA Module")
#     print("=" * 60)
#
#     eda_config = eda_configs['Full']
#     eda_model = SimplifiedEDA(
#         q_size=(16, 8), n_heads=1, n_head_channels=512, n_groups=1,
#         attn_drop=0.0, proj_drop=0.0, stride=2,
#         offset_range_factor=5.0, ksize=4, share=True,
#         ablation_config=eda_config
#     )
#     print(eda_model.get_ablation_summary())
#
#     # 测试HAQN模块
#     print("\n" + "=" * 60)
#     print("Testing Simplified HAQN Module")
#     print("=" * 60)
#
#     haqn_config = haqn_configs['Full']
#     haqn_model = SimplifiedHAQN(
#         input_dim=512, num_queries=32, num_layers=3, row_dim=1,
#         ablation_config=haqn_config
#     )
#     print(haqn_model.get_ablation_summary())
#
#     print("\n✅ Simplified ablation modules ready!")
#     print("📋 Recommended experiment plan:")
#     print("   1. Test EDA components: MSOF vs CMA vs SAL")
#     print("   2. Test HAQN components: HQ vs AF vs QRM")
#     print("   3. Test key combinations for synergy effects")
#     print("   4. Compare with original baseline (87.3/86.8)")


class GeneralFusion(nn.Module):
    def __init__(self, feat_dim, num_experts, head, reg_weight=0.1, dropout=0.1, cfg=None):
        super(GeneralFusion, self).__init__()
        self.reg_weight = reg_weight
        self.feat_dim = feat_dim
        self.datasetsname = cfg.DATASETS.NAMES

        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM

        self.combineway = 'newablation'
        print('combineway:', self.combineway,'mxa')
        logger = logging.getLogger("DeMo")
        logger.info(f'combineway: {self.combineway}')

        self.newdeform = cfg.MODEL.NEWDEFORM
        print('newdeform:', self.newdeform,'mxa')


        if self.HDM and self.combineway == 'normal':
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




        self.UsingDiversity = False
        print('UsingDiversity:', self.UsingDiversity,'mxa')

        # loggernew = logging.getLogger("DeMo")
        # loggernew.info(f'combineway: {self.combineway}')

        if self.combineway == 'deform':
            if self.datasetsname == 'RGBNT201':
                q_size = (16,8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)

            if self.newdeform:
                # 直接替换
                self.deformselect = DAttentionEnhanced(
                    q_size, 1, 512, 1, 0.0, 0.0, 2,
                    5.0, 4, True
                )
            else:
                self.deformselect = DAttentionBaseline(
                    q_size, 1, 512, 1, 0.0, 0.0, 2,
                    5.0, 4, True
                )
        elif self.combineway == 'boq':
            # 初始化模型
            self.multimodalboq = MultiModalBoQ(in_dim= self.feat_dim, num_queries=64, num_layers=4, row_dim=1)
        elif self.combineway == 'adaptiveboq':
            self.modalityboq = ModalitySpecificBoQ(input_dim=self.feat_dim, num_queries=64, num_layers=4, row_dim=1)
        elif self.combineway == 'adaptiveboqdeform':
            self.modalityboq = ModalitySpecificBoQ(input_dim=self.feat_dim, num_queries=64, num_layers=4, row_dim=1)
            if self.datasetsname == 'RGBNT201':
                q_size = (16,8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)


            if self.newdeform:
                # 直接替换
                self.deformselect = DAttentionEnhanced(
                    q_size, 1, 512, 1, 0.0, 0.0, 2,
                    5.0, 4, True
                )
            else:
                self.deformselect = DAttentionBaseline(
                    q_size, 1, 512, 1, 0.0, 0.0, 2,
                    5.0, 4, True
                )

        elif self.combineway == 'adaptiveboqdeformablation':
            # ./runrgbnt201.sh 3 aboqnewdeform_ablation_full_no_msof
            # ./runrgbnt201.sh 0 aboqnewdeform_ablation_no_HQ_full

            ablation_configs_boq = create_haqn_ablation_configs()

            # 示例：创建移除查询传播的模型
            configboq = ablation_configs_boq['Full']

            # 测试模态特异性BoQ
            self.modalityboqablation = ModalitySpecificBoQAblation(
                input_dim=self.feat_dim, num_queries=64, num_layers=4, row_dim=1,
                ablation_config=configboq
            )


            if self.datasetsname == 'RGBNT201':
                q_size = (16,8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)

            ablation_configs_deform = create_eda_ablation_configs()

            # 示例：创建移除AMW组件的模型
            configdeform = ablation_configs_deform['Full']
            self.deformselectablation = DAttentionEnhancedAblation(
                q_size=q_size, n_heads=1, n_head_channels=512, n_groups=1,
                attn_drop=0.0, proj_drop=0.0, stride=2,
                offset_range_factor=5.0, ksize=4, share=True,
                ablation_config=configdeform
            )

            # 直接替换
            # self.deformselect = DAttentionEnhanced(
            #     q_size, 1, 512, 1, 0.0, 0.0, 2,
            #     5.0, 4, True
            # )

        elif self.combineway == 'newablation':
            # ./runrgbnt201.sh 3 aboqnewdeform_ablation_full_no_msof
            # ./runrgbnt201.sh 0 aboqnewdeform_ablation_no_HQ_full

            if self.datasetsname == 'RGBNT201':
                q_size = (16, 8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)

            eda_configs, haqn_configs = create_simplified_ablation_configs()

            # 测试EDA模块
            print("=" * 60)
            print("Testing Simplified EDA Module")
            print("=" * 60)

            eda_config = eda_configs['Baseline']
            self.eda_model = SimplifiedEDA(
                q_size=q_size, n_heads=1, n_head_channels=512, n_groups=1,
                attn_drop=0.0, proj_drop=0.0, stride=2,
                offset_range_factor=5.0, ksize=4, share=True,
                ablation_config=eda_config
            )
            print(self.eda_model.get_ablation_summary())
            # 测试HAQN模块
            print("\n" + "=" * 60)
            print("Testing Simplified HAQN Module")
            print("=" * 60)

            haqn_config = haqn_configs['Baseline']
            self.haqn_model = SimplifiedHAQN(
                input_dim=self.feat_dim, num_queries=64, num_layers=4, row_dim=1,
                ablation_config=haqn_config
            )
            print(self.haqn_model.get_ablation_summary())

            print("\n✅ Simplified ablation modules ready!")
            print("📋 Recommended experiment plan:")
            print("   1. Test EDA components: MSOF vs CMA vs SAL")
            print("   2. Test HAQN components: HQ vs AF vs QRM")
            print("   3. Test key combinations for synergy effects")
            print("   4. Compare with original baseline (87.3/86.8)")



        elif self.combineway == 'ebblockdeform':
            if self.datasetsname == 'RGBNT201':
                q_size = (16,8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)
            self.ebblock_r = EBlock(c=self.feat_dim)
            self.ebblock_n = EBlock(c=self.feat_dim)
            self.ebblock_t = EBlock(c=self.feat_dim)
            if self.newdeform:
                # 直接替换
                self.deformselect = DAttentionEnhanced(
                    q_size, 1, 512, 1, 0.0, 0.0, 2,
                    5.0, 4, True
                )
            else:
                self.deformselect = DAttentionBaseline(
                    q_size, 1, 512, 1, 0.0, 0.0, 2,
                    5.0, 4, True
                )
        elif self.combineway == 'multimodelse':
            if self.datasetsname == 'RGBNT201':
                q_size = (16, 8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)
            self.usemultimodal_token_se = True
            print('usemultimodal_token_se:', self.usemultimodal_token_se, 'mxa')
            if self.usemultimodal_token_se:
                self.multimodal_token_se = MultiModalTokenSE(
                    token_dim= q_size[0] * q_size[1],
                    feature_dim=self.feat_dim,
                    reduction=4,
                    use_residual=True,
                    interaction_mode='adaptive_weight'
                )
            if self.UsingDiversity:
                # 🚀 简化的特征区分增强器
                self.feature_diversifiers = nn.ModuleList([
                    FeatureDiversifier(feat_dim, diversifier_id=i)
                    for i in range(7)
                ])

                # 区分度损失权重
                self.diversity_weight = 0.1

            # self.deformselect = DAttentionBaseline(
            #     q_size, 1, 512, 1, 0.0, 0.0, 2,
            #     5.0, 4, True
            # )

            
        elif self.combineway == 'sedeform':
            if self.datasetsname == 'RGBNT201':
                q_size = (16, 8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)

            if self.newdeform:
                # 直接替换
                self.deformselect = DAttentionEnhanced(
                    q_size, 1, 512, 1, 0.0, 0.0, 2,
                    5.0, 4, True
                )
            else:
                self.deformselect = DAttentionBaseline(
                    q_size, 1, 512, 1, 0.0, 0.0, 2,
                    5.0, 4, True
                )
            self.tokense_r = TokenSE(token_dim=q_size[0] * q_size[1], reduction=4, use_residual=True)
            self.tokense_n = TokenSE(token_dim=q_size[0] * q_size[1], reduction=4, use_residual=True)
            self.tokense_t = TokenSE(token_dim=q_size[0] * q_size[1], reduction=4, use_residual=True)
        elif self.combineway == 'multimodeltokense':
            if self.datasetsname == 'RGBNT201':
                q_size = (16, 8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)
            self.tokense_r = TokenSE(token_dim=q_size[0] * q_size[1], reduction=4, use_residual=True)
            self.tokense_n = TokenSE(token_dim=q_size[0] * q_size[1], reduction=4, use_residual=True)
            self.tokense_t = TokenSE(token_dim=q_size[0] * q_size[1], reduction=4, use_residual=True)
            if self.UsingDiversity:
                # 🚀 简化的特征区分增强器
                self.feature_diversifiers = nn.ModuleList([
                    FeatureDiversifier(feat_dim, diversifier_id=i)
                    for i in range(7)
                ])

                # 区分度损失权重
                self.diversity_weight = 0.1

            
        elif self.combineway == 'rwkvadd' or self.combineway == 'rwkvaddlinear':
            rwkv_cfg = dict(
                #n_embd=feat_dim,
                #n_head=12,
                n_layer=12,
                layer_id=0,
                shift_mode='q_shift_multihead',
                shift_pixel=1,
                drop_path=0,
                hidden_rate=4,
                init_mode='fancy',
                key_norm=False,
                with_cls_token=False,
                with_cp=False,

                ########
                n_embd=feat_dim,
                n_head=8,
                init_values=1e-5,
                post_norm=True,

            )

            self.rwkvblock_r = Block(**rwkv_cfg)
            self.rwkvblock_n = Block(**rwkv_cfg)
            self.rwkvblock_t = Block(**rwkv_cfg)
            self.rwkvblock_rn = Block(**rwkv_cfg)
            self.rwkvblock_rt = Block(**rwkv_cfg)
            self.rwkvblock_nt = Block(**rwkv_cfg)
            self.rwkvblock_rnt = Block(**rwkv_cfg)
            #in_features = 128 +128 +128 + 256 +256 +256 + 384
            self.linearrwkv = nn.Linear(feat_dim, feat_dim)
        elif self.combineway == 'rwkvcross':
            self.rwkvcross_r = RWKV_CrossAttention(feat_dim, n_query=1)
            self.rwkvcross_n = RWKV_CrossAttention(feat_dim, n_query=1)
            self.rwkvcross_t = RWKV_CrossAttention(feat_dim, n_query=1)
            self.rwkvcross_rn = RWKV_CrossAttention(feat_dim, n_query=1)
            self.rwkvcross_rt = RWKV_CrossAttention(feat_dim, n_query=1)
            self.rwkvcross_nt = RWKV_CrossAttention(feat_dim, n_query=1)
            self.rwkvcross_rnt = RWKV_CrossAttention(feat_dim, n_query=1)
        elif self.combineway == 'EMA':
            self.ema = EMA(feat_dim, factor=8)
            self.ema_r = EMA(feat_dim, factor=8)
            self.ema_n = EMA(feat_dim, factor=8)
            self.ema_t = EMA(feat_dim, factor=8)
        elif self.combineway == 'deema':
            if self.datasetsname == 'RGBNT201':
                q_size = (16,8)
            elif self.datasetsname == 'RGBNT100':
                q_size = (8, 16)
            else:
                q_size = (8, 16)

            self.deformselect = DAttentionBaseline(
                q_size, 1, 512, 1, 0.0, 0.0, 2,
                5.0, 4, True
            )
            self.ema = EMA(feat_dim, factor=8)
            self.ema_r = EMA(feat_dim, factor=8)
            self.ema_n = EMA(feat_dim, factor=8)
            self.ema_t = EMA(feat_dim, factor=8)





    def forward_HDMDeform(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input
        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)


        # token selectect 用可变哪个东西
        RGB_cash = RGB_cash.permute(1, 2, 0)
        NI_cash = NI_cash.permute(1, 2, 0)
        TI_cash = TI_cash.permute(1, 2, 0)

        if self.datasetsname == 'RGBNT100':
            q_size = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            q_size = (16, 8)
        else:
            q_size = (8, 16)

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), RGB_cash.size(1), q_size[0], q_size[1])
        NI_cash = NI_cash.reshape(NI_cash.size(0), NI_cash.size(1), q_size[0], q_size[1])
        TI_cash = TI_cash.reshape(TI_cash.size(0), TI_cash.size(1), q_size[0], q_size[1])

        # B, C, H, W = RGB_cash.size()
        # dtype, device = RGB_cash.dtype, RGB_cash.device
        # data = torch.cat([RGB_cash, NI_cash, TI_cash], dim=1)
        RGB_cash,NI_cash,TI_cash = self.deformselect(RGB_cash, NI_cash, TI_cash)



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
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared

    def forward_HDMseDeform(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input

        RGB_cash = self.tokense_r(RGB_cash)
        NI_cash = self.tokense_n(NI_cash)
        TI_cash = self.tokense_t(TI_cash)

        RGB_cash = RGB_cash.permute(0, 2, 1)  # [B, T, N] → [B, N, T]
        NI_cash = NI_cash.permute(0, 2, 1)
        TI_cash = TI_cash.permute(0, 2, 1)

        if self.datasetsname == 'RGBNT100':
            q_size = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            q_size = (16, 8)
        else:
            q_size = (8, 16)

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), RGB_cash.size(1), q_size[0], q_size[1])
        NI_cash = NI_cash.reshape(NI_cash.size(0), NI_cash.size(1), q_size[0], q_size[1])
        TI_cash = TI_cash.reshape(TI_cash.size(0), TI_cash.size(1), q_size[0], q_size[1])

        # B, C, H, W = RGB_cash.size()
        # dtype, device = RGB_cash.dtype, RGB_cash.device
        # data = torch.cat([RGB_cash, NI_cash, TI_cash], dim=1)
        RGB_cash,NI_cash,TI_cash = self.deformselect(RGB_cash, NI_cash, TI_cash)



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
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared
    
    def forward_HDMmultimodeltokense(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input

        RGB_cash = self.tokense_r(RGB_cash)
        NI_cash = self.tokense_n(NI_cash)
        TI_cash = self.tokense_t(TI_cash)



        RGB_cash = RGB_cash.permute(1, 0, 2) # token batch dim
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
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()


        if not self.UsingDiversity:
            return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared
        else:
            raw_features = [RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared]
            # 🚀 特征区分增强
            diverse_features = []
            for i, (feature, diversifier) in enumerate(zip(raw_features, self.feature_diversifiers)):
                diverse_feature = diversifier(feature, raw_features, i)
                diverse_features.append(diverse_feature)
            # 计算简化的区分度损失
            diversity_loss = self._compute_simple_diversity_loss(diverse_features)
            return diverse_features[0], diverse_features[1], diverse_features[2], diverse_features[3], diverse_features[4], diverse_features[5], diverse_features[6], diversity_loss


    def forward_HDMmultimodelse(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input

        # RGB_cash = self.tokense_r(RGB_cash)
        # NI_cash = self.tokense_n(NI_cash)
        # TI_cash = self.tokense_t(TI_cash)
        if self.usemultimodal_token_se:
            RGB_cash, NI_cash, TI_cash = self.multimodal_token_se(RGB_cash, NI_cash, TI_cash)


        RGB_cash = RGB_cash.permute(1, 0, 2) # token batch dim
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)


        # xiamian shi deform
        # RGB_cash = RGB_cash.permute(0, 2, 1)  # [B, T, N] → [B, N, T]
        # NI_cash = NI_cash.permute(0, 2, 1)
        # TI_cash = TI_cash.permute(0, 2, 1)
        #
        # if self.datasetsname == 'RGBNT100':
        #     q_size = (8, 16)
        # elif self.datasetsname == 'RGBNT201':
        #     q_size = (16, 8)
        # else:
        #     q_size = (8, 16)
        #
        # RGB_cash = RGB_cash.reshape(RGB_cash.size(0), RGB_cash.size(1), q_size[0], q_size[1])
        # NI_cash = NI_cash.reshape(NI_cash.size(0), NI_cash.size(1), q_size[0], q_size[1])
        # TI_cash = TI_cash.reshape(TI_cash.size(0), TI_cash.size(1), q_size[0], q_size[1])
        #
        # RGB_cash,NI_cash,TI_cash = self.deformselect(RGB_cash, NI_cash, TI_cash)

        ############




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
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()


        if not self.UsingDiversity:
            return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared
        else:
            raw_features = [RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared]
            # 🚀 特征区分增强
            diverse_features = []
            for i, (feature, diversifier) in enumerate(zip(raw_features, self.feature_diversifiers)):
                diverse_feature = diversifier(feature, raw_features, i)
                diverse_features.append(diverse_feature)
            # 计算简化的区分度损失
            diversity_loss = self._compute_simple_diversity_loss(diverse_features)
            return diverse_features[0], diverse_features[1], diverse_features[2], diverse_features[3], diverse_features[4], diverse_features[5], diverse_features[6], diversity_loss

    def _compute_simple_diversity_loss(self, features):
        """计算区分度损失"""
        total_loss = 0
        num_pairs = 0

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                similarity = F.cosine_similarity(features[i], features[j], dim=-1).mean()
                loss = torch.exp(similarity * 5)  # 惩罚高相似度
                total_loss += loss
                num_pairs += 1

        return (total_loss / num_pairs) * self.diversity_weight if num_pairs > 0 else torch.tensor(0.0)

    def forward_HDMebblockDeform(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input


        RGB_cash = RGB_cash.permute(0, 2, 1)  # [B, T, N] → [B, N, T]
        NI_cash = NI_cash.permute(0, 2, 1)
        TI_cash = TI_cash.permute(0, 2, 1)

        if self.datasetsname == 'RGBNT100':
            q_size = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            q_size = (16, 8)
        else:
            q_size = (8, 16)

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), RGB_cash.size(1), q_size[0], q_size[1])
        NI_cash = NI_cash.reshape(NI_cash.size(0), NI_cash.size(1), q_size[0], q_size[1])
        TI_cash = TI_cash.reshape(TI_cash.size(0), TI_cash.size(1), q_size[0], q_size[1])

        RGB_cash = self.ebblock_r(RGB_cash)
        NI_cash = self.ebblock_n(NI_cash)
        TI_cash = self.ebblock_t(TI_cash)

        # B, C, H, W = RGB_cash.size()
        # dtype, device = RGB_cash.dtype, RGB_cash.device
        # data = torch.cat([RGB_cash, NI_cash, TI_cash], dim=1)
        RGB_cash,NI_cash,TI_cash = self.deformselect(RGB_cash, NI_cash, TI_cash)



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
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared


    def forward_HDMDeema(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)

        # ema module-------------------------------------------
        if self.datasetsname == 'RGBNT100':
            height = 8
        elif self.datasetsname == 'RGBNT201':
            height = 16
        else:
            height = 8

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), height,-1,RGB_cash.size(2))
        RGB_cash = RGB_cash.permute(0, 3, 1, 2)
        RGB_cash = self.ema_r(RGB_cash)
        RGB_cash = RGB_cash.permute(0, 2, 3, 1)
        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), -1, RGB_cash.size(3))

        NI_cash = NI_cash.reshape(NI_cash.size(0), height,-1,NI_cash.size(2))
        NI_cash = NI_cash.permute(0, 3, 1, 2)
        NI_cash = self.ema_n(NI_cash)
        NI_cash = NI_cash.permute(0, 2, 3, 1)
        NI_cash = NI_cash.reshape(NI_cash.size(0), -1, NI_cash.size(3))

        TI_cash = TI_cash.reshape(TI_cash.size(0), height,-1,TI_cash.size(2))
        TI_cash = TI_cash.permute(0, 3, 1, 2)
        TI_cash = self.ema_t(TI_cash)
        TI_cash = TI_cash.permute(0, 2, 3, 1)
        TI_cash = TI_cash.reshape(TI_cash.size(0), -1, TI_cash.size(3))
        #-------------------------------------------

        # permute for the cross attn input
        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)

        # token selectect 用可变哪个东西
        RGB_cash = RGB_cash.permute(1, 2, 0)
        NI_cash = NI_cash.permute(1, 2, 0)
        TI_cash = TI_cash.permute(1, 2, 0)

        if self.datasetsname == 'RGBNT100':
            q_size = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            q_size = (16, 8)
        else:
            q_size = (8, 16)

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), RGB_cash.size(1), q_size[0], q_size[1])
        NI_cash = NI_cash.reshape(NI_cash.size(0), NI_cash.size(1), q_size[0], q_size[1])
        TI_cash = TI_cash.reshape(TI_cash.size(0), TI_cash.size(1), q_size[0], q_size[1])

        # B, C, H, W = RGB_cash.size()
        # dtype, device = RGB_cash.dtype, RGB_cash.device
        # data = torch.cat([RGB_cash, NI_cash, TI_cash], dim=1)
        RGB_cash, NI_cash, TI_cash = self.deformselect(RGB_cash, NI_cash, TI_cash)

        # get the embedding
        withglobal = False
        if withglobal:
            RGB = torch.cat([r_global, RGB_cash], dim=0)
            NI = torch.cat([n_global, NI_cash], dim=0)
            TI = torch.cat([t_global, TI_cash], dim=0)
        else:
            RGB = RGB_cash
            NI = NI_cash
            TI = TI_cash

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
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2,
                                                                 0).squeeze()  # r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared

    def forward_HDMrw(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input

        RGB_cash = RGB_cash.permute(1, 0, 2) # tokenamount BATCH DIM
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)



        RGB_NI_cash =  torch.cat([RGB_cash, NI_cash], dim=0)
        RGB_TI_cash =  torch.cat([RGB_cash, TI_cash], dim=0)
        NI_TI_cash =  torch.cat([NI_cash, TI_cash], dim=0)
        RGB_NI_TI_cash =  torch.cat([RGB_cash, NI_cash, TI_cash], dim=0)

        if self.datasetsname == 'RGBNT100':
            patch_resolution = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            patch_resolution = (16, 8)
        else:
            patch_resolution = (8, 16)

        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)
        RGB_NI_cash = RGB_NI_cash.permute(1, 0, 2)
        RGB_TI_cash = RGB_TI_cash.permute(1, 0, 2)
        NI_TI_cash = NI_TI_cash.permute(1, 0, 2)
        RGB_NI_TI_cash = RGB_NI_TI_cash.permute(1, 0, 2)

        #patch_resolution = (8, 8)
        RGB_cash = self.rwkvblock_r(RGB_cash, patch_resolution)
        NI_cash = self.rwkvblock_n(NI_cash, patch_resolution)
        TI_cash = self.rwkvblock_t(TI_cash, patch_resolution)
        RGB_NI_cash = self.rwkvblock_rn(RGB_NI_cash, patch_resolution)
        RGB_TI_cash = self.rwkvblock_rt(RGB_TI_cash, patch_resolution)
        NI_TI_cash = self.rwkvblock_nt(NI_TI_cash, patch_resolution)
        RGB_NI_TI_cash = self.rwkvblock_rnt(RGB_NI_TI_cash, patch_resolution)

        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)
        RGB_NI_cash = RGB_NI_cash.permute(1, 0, 2)
        RGB_TI_cash = RGB_TI_cash.permute(1, 0, 2)
        NI_TI_cash = NI_TI_cash.permute(1, 0, 2)
        RGB_NI_TI_cash = RGB_NI_TI_cash.permute(1, 0, 2)

        # RGB = torch.cat([r_global, RGB_cash], dim=0)
        # NI = torch.cat([n_global, NI_cash], dim=0)
        # TI = torch.cat([t_global, TI_cash], dim=0)
        RGB = RGB_cash
        NI = NI_cash
        TI = TI_cash
        RGB_NI = RGB_NI_cash
        RGB_TI = RGB_TI_cash
        NI_TI = NI_TI_cash
        RGB_NI_TI = RGB_NI_TI_cash

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
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared


    def forward_HDMrwpluslinear(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input

        RGB_cash = RGB_cash.permute(1, 0, 2) # tokenamount BATCH DIM
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)



        RGB_NI_cash =  torch.cat([RGB_cash, NI_cash], dim=0)
        RGB_TI_cash =  torch.cat([RGB_cash, TI_cash], dim=0)
        NI_TI_cash =  torch.cat([NI_cash, TI_cash], dim=0)
        RGB_NI_TI_cash =  torch.cat([RGB_cash, NI_cash, TI_cash], dim=0)

        if self.datasetsname == 'RGBNT100':
            patch_resolution = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            patch_resolution = (16, 8)
        else:
            patch_resolution = (8, 16)

        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)
        RGB_NI_cash = RGB_NI_cash.permute(1, 0, 2)
        RGB_TI_cash = RGB_TI_cash.permute(1, 0, 2)
        NI_TI_cash = NI_TI_cash.permute(1, 0, 2)
        RGB_NI_TI_cash = RGB_NI_TI_cash.permute(1, 0, 2)

        #patch_resolution = (8, 8)
        RGB_cash = self.rwkvblock_r(RGB_cash, patch_resolution)  # input : B T dim
        NI_cash = self.rwkvblock_n(NI_cash, patch_resolution)
        TI_cash = self.rwkvblock_t(TI_cash, patch_resolution)
        RGB_NI_cash = self.rwkvblock_rn(RGB_NI_cash, patch_resolution)
        RGB_TI_cash = self.rwkvblock_rt(RGB_TI_cash, patch_resolution)
        NI_TI_cash = self.rwkvblock_nt(NI_TI_cash, patch_resolution)
        RGB_NI_TI_cash = self.rwkvblock_rnt(RGB_NI_TI_cash, patch_resolution)

        T_list = [RGB_cash.shape[1], NI_cash.shape[1], TI_cash.shape[1],
                  RGB_NI_cash.shape[1], RGB_TI_cash.shape[1],
                  NI_TI_cash.shape[1], RGB_NI_TI_cash.shape[1]]
        concat_tensor = torch.cat([
            RGB_cash, NI_cash, TI_cash,
            RGB_NI_cash, RGB_TI_cash,
            NI_TI_cash, RGB_NI_TI_cash
        ], dim=1)

        projected = self.linearrwkv(concat_tensor)
        splits = torch.split(projected, T_list, dim=1)
        RGB_cash, NI_cash, TI_cash, RGB_NI_cash, RGB_TI_cash, NI_TI_cash, RGB_NI_TI_cash = splits







        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)
        RGB_NI_cash = RGB_NI_cash.permute(1, 0, 2)
        RGB_TI_cash = RGB_TI_cash.permute(1, 0, 2)
        NI_TI_cash = NI_TI_cash.permute(1, 0, 2)
        RGB_NI_TI_cash = RGB_NI_TI_cash.permute(1, 0, 2)

        # RGB = torch.cat([r_global, RGB_cash], dim=0)
        # NI = torch.cat([n_global, NI_cash], dim=0)
        # TI = torch.cat([t_global, TI_cash], dim=0)
        RGB = RGB_cash
        NI = NI_cash
        TI = TI_cash
        RGB_NI = RGB_NI_cash
        RGB_TI = RGB_TI_cash
        NI_TI = NI_TI_cash
        RGB_NI_TI = RGB_NI_TI_cash

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
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared

    def forward_HDMcrossrwkv(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1)
        n_global = NI_global.unsqueeze(1)
        t_global = TI_global.unsqueeze(1)

        # get the embedding
        RGB = torch.cat([r_global, RGB_cash], dim=1)
        NI = torch.cat([n_global, NI_cash], dim=1)
        TI = torch.cat([t_global, TI_cash], dim=1)
        RGB_NI = torch.cat([RGB, NI], dim=1)
        RGB_TI = torch.cat([RGB, TI], dim=1)
        NI_TI = torch.cat([NI, TI], dim=1)
        RGB_NI_TI = torch.cat([RGB, NI, TI], dim=1)
        batch = RGB.size(0)

        RGB_special = self.rwkvcross_r(RGB).squeeze()
        NI_special = self.rwkvcross_n(NI).squeeze()
        TI_special = self.rwkvcross_t(TI).squeeze()
        RN_shared = self.rwkvcross_rn(RGB_NI).squeeze()
        RT_shared = self.rwkvcross_rt(RGB_TI).squeeze()
        NT_shared = self.rwkvcross_nt(NI_TI).squeeze()
        RNT_shared = self.rwkvcross_rnt(RGB_NI_TI).squeeze()



        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared
    
    
    def forward_HDMema(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)

        if self.datasetsname == 'RGBNT100':
            height = 8
        elif self.datasetsname == 'RGBNT201':
            height = 16
        else:
            height = 8

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), height,-1,RGB_cash.size(2))
        RGB_cash = RGB_cash.permute(0, 3, 1, 2)
        RGB_cash = self.ema_r(RGB_cash)
        RGB_cash = RGB_cash.permute(0, 2, 3, 1)
        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), -1, RGB_cash.size(3))

        NI_cash = NI_cash.reshape(NI_cash.size(0), height,-1,NI_cash.size(2))
        NI_cash = NI_cash.permute(0, 3, 1, 2)
        NI_cash = self.ema_n(NI_cash)
        NI_cash = NI_cash.permute(0, 2, 3, 1)
        NI_cash = NI_cash.reshape(NI_cash.size(0), -1, NI_cash.size(3))

        TI_cash = TI_cash.reshape(TI_cash.size(0), height,-1,TI_cash.size(2))
        TI_cash = TI_cash.permute(0, 3, 1, 2)
        TI_cash = self.ema_t(TI_cash)
        TI_cash = TI_cash.permute(0, 2, 3, 1)
        TI_cash = TI_cash.reshape(TI_cash.size(0), -1, TI_cash.size(3))



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

        #从这里开始拿到的都是 BS 512 的特征也就是  B dIM
        # for single modality
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # for double modality
        RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # for triple modality
        RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared


    def forward_HDMboq(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
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

        # 创建特征字典
        features_dict = {
            'RGB': RGB,
            'NI': NI,
            'TI': TI,
            'RGB_NI': RGB_NI,
            'RGB_TI': RGB_TI,
            'NI_TI': NI_TI,
            'RGB_NI_TI': RGB_NI_TI
        }

        results, attentions = self.multimodalboq(features_dict)
        RGB_special = results['RGB']
        NI_special = results['NI']
        TI_special = results['TI']
        RN_shared = results['RGB_NI']
        RT_shared = results['RGB_TI']
        NT_shared = results['NI_TI']
        RNT_shared = results['RGB_NI_TI']



        #
        # # get the learnable token
        # r_embedding = self.r_token.repeat(1, batch, 1)
        # n_embedding = self.n_token.repeat(1, batch, 1)
        # t_embedding = self.t_token.repeat(1, batch, 1)
        # rn_embedding = self.rn_token.repeat(1, batch, 1)
        # rt_embedding = self.rt_token.repeat(1, batch, 1)
        # nt_embedding = self.nt_token.repeat(1, batch, 1)
        # rnt_embedding = self.rnt_token.repeat(1, batch, 1)
        #
        # #从这里开始拿到的都是 BS 512 的特征也就是  B dIM
        # # for single modality
        # RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
        # NI_special = (self.n(n_embedding, NI, NI)[0]).permute(1, 2, 0).squeeze()
        # TI_special = (self.t(t_embedding, TI, TI)[0]).permute(1, 2, 0).squeeze()
        # # for double modality
        # RN_shared = (self.rn(rn_embedding, RGB_NI, RGB_NI)[0]).permute(1, 2, 0).squeeze()
        # RT_shared = (self.rt(rt_embedding, RGB_TI, RGB_TI)[0]).permute(1, 2, 0).squeeze()
        # NT_shared = (self.nt(nt_embedding, NI_TI, NI_TI)[0]).permute(1, 2, 0).squeeze()
        # # for triple modality
        # RNT_shared = (self.rnt(rnt_embedding, RGB_NI_TI, RGB_NI_TI)[0]).permute(1, 2, 0).squeeze()

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared


    def forward_HDMadaptiveboq(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
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

        # 创建特征字典
        features_dict = {
            'RGB': RGB,
            'NI': NI,
            'TI': TI,
            'RGB_NI': RGB_NI,
            'RGB_TI': RGB_TI,
            'NI_TI': NI_TI,
            'RGB_NI_TI': RGB_NI_TI
        }

        results, attentions = self.modalityboq(features_dict)
        RGB_special = results['RGB']
        NI_special = results['NI']
        TI_special = results['TI']
        RN_shared = results['RGB_NI']
        RT_shared = results['RGB_TI']
        NT_shared = results['NI_TI']
        RNT_shared = results['RGB_NI_TI']

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared


    def forward_HDMadaptiveboqDeform(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input
        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)


        # token selectect 用可变哪个东西
        RGB_cash = RGB_cash.permute(1, 2, 0)
        NI_cash = NI_cash.permute(1, 2, 0)
        TI_cash = TI_cash.permute(1, 2, 0)

        if self.datasetsname == 'RGBNT100':
            q_size = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            q_size = (16, 8)
        else:
            q_size = (8, 16)

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), RGB_cash.size(1), q_size[0], q_size[1])
        NI_cash = NI_cash.reshape(NI_cash.size(0), NI_cash.size(1), q_size[0], q_size[1])
        TI_cash = TI_cash.reshape(TI_cash.size(0), TI_cash.size(1), q_size[0], q_size[1])

        # B, C, H, W = RGB_cash.size()
        # dtype, device = RGB_cash.dtype, RGB_cash.device
        # data = torch.cat([RGB_cash, NI_cash, TI_cash], dim=1)
        RGB_cash,NI_cash,TI_cash = self.deformselect(RGB_cash, NI_cash, TI_cash)

        # get the embedding
        RGB = torch.cat([r_global, RGB_cash], dim=0)
        NI = torch.cat([n_global, NI_cash], dim=0)
        TI = torch.cat([t_global, TI_cash], dim=0)
        RGB_NI = torch.cat([RGB, NI], dim=0)
        RGB_TI = torch.cat([RGB, TI], dim=0)
        NI_TI = torch.cat([NI, TI], dim=0)
        RGB_NI_TI = torch.cat([RGB, NI, TI], dim=0)

        # 创建特征字典
        features_dict = {
            'RGB': RGB,
            'NI': NI,
            'TI': TI,
            'RGB_NI': RGB_NI,
            'RGB_TI': RGB_TI,
            'NI_TI': NI_TI,
            'RGB_NI_TI': RGB_NI_TI
        }

        results, attentions = self.modalityboq(features_dict)
        RGB_special = results['RGB']
        NI_special = results['NI']
        TI_special = results['TI']
        RN_shared = results['RGB_NI']
        RT_shared = results['RGB_TI']
        NT_shared = results['NI_TI']
        RNT_shared = results['RGB_NI_TI']

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared


    def forward_HDMadaptiveboqDeformAblation(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input
        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)


        # token selectect 用可变哪个东西
        RGB_cash = RGB_cash.permute(1, 2, 0)
        NI_cash = NI_cash.permute(1, 2, 0)
        TI_cash = TI_cash.permute(1, 2, 0)

        if self.datasetsname == 'RGBNT100':
            q_size = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            q_size = (16, 8)
        else:
            q_size = (8, 16)

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), RGB_cash.size(1), q_size[0], q_size[1])
        NI_cash = NI_cash.reshape(NI_cash.size(0), NI_cash.size(1), q_size[0], q_size[1])
        TI_cash = TI_cash.reshape(TI_cash.size(0), TI_cash.size(1), q_size[0], q_size[1])

        # B, C, H, W = RGB_cash.size()
        # dtype, device = RGB_cash.dtype, RGB_cash.device
        # data = torch.cat([RGB_cash, NI_cash, TI_cash], dim=1)
        RGB_cash,NI_cash,TI_cash = self.deformselectablation(RGB_cash, NI_cash, TI_cash)

        # get the embedding
        RGB = torch.cat([r_global, RGB_cash], dim=0)
        NI = torch.cat([n_global, NI_cash], dim=0)
        TI = torch.cat([t_global, TI_cash], dim=0)
        RGB_NI = torch.cat([RGB, NI], dim=0)
        RGB_TI = torch.cat([RGB, TI], dim=0)
        NI_TI = torch.cat([NI, TI], dim=0)
        RGB_NI_TI = torch.cat([RGB, NI, TI], dim=0)

        # 创建特征字典
        features_dict = {
            'RGB': RGB,
            'NI': NI,
            'TI': TI,
            'RGB_NI': RGB_NI,
            'RGB_TI': RGB_TI,
            'NI_TI': NI_TI,
            'RGB_NI_TI': RGB_NI_TI
        }

        results, attentions = self.modalityboqablation(features_dict)
        RGB_special = results['RGB']
        NI_special = results['NI']
        TI_special = results['TI']
        RN_shared = results['RGB_NI']
        RT_shared = results['RGB_TI']
        NT_shared = results['NI_TI']
        RNT_shared = results['RGB_NI_TI']

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared


    def forward_HDMnewablation(self, RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global):
        # get the global feature
        r_global = RGB_global.unsqueeze(1).permute(1, 0, 2)
        n_global = NI_global.unsqueeze(1).permute(1, 0, 2)
        t_global = TI_global.unsqueeze(1).permute(1, 0, 2)
        # permute for the cross attn input
        RGB_cash = RGB_cash.permute(1, 0, 2)
        NI_cash = NI_cash.permute(1, 0, 2)
        TI_cash = TI_cash.permute(1, 0, 2)


        # token selectect 用可变哪个东西
        RGB_cash = RGB_cash.permute(1, 2, 0)
        NI_cash = NI_cash.permute(1, 2, 0)
        TI_cash = TI_cash.permute(1, 2, 0)

        if self.datasetsname == 'RGBNT100':
            q_size = (8, 16)
        elif self.datasetsname == 'RGBNT201':
            q_size = (16, 8)
        else:
            q_size = (8, 16)

        RGB_cash = RGB_cash.reshape(RGB_cash.size(0), RGB_cash.size(1), q_size[0], q_size[1])
        NI_cash = NI_cash.reshape(NI_cash.size(0), NI_cash.size(1), q_size[0], q_size[1])
        TI_cash = TI_cash.reshape(TI_cash.size(0), TI_cash.size(1), q_size[0], q_size[1])

        # B, C, H, W = RGB_cash.size()
        # dtype, device = RGB_cash.dtype, RGB_cash.device
        # data = torch.cat([RGB_cash, NI_cash, TI_cash], dim=1)
        RGB_cash,NI_cash,TI_cash = self.eda_model(RGB_cash, NI_cash, TI_cash)

        # get the embedding
        RGB = torch.cat([r_global, RGB_cash], dim=0)
        NI = torch.cat([n_global, NI_cash], dim=0)
        TI = torch.cat([t_global, TI_cash], dim=0)
        RGB_NI = torch.cat([RGB, NI], dim=0)
        RGB_TI = torch.cat([RGB, TI], dim=0)
        NI_TI = torch.cat([NI, TI], dim=0)
        RGB_NI_TI = torch.cat([RGB, NI, TI], dim=0)

        # # 创建特征字典
        features_dict = {
            'RGB': RGB,
            'NI': NI,
            'TI': TI,
            'RGB_NI': RGB_NI,
            'RGB_TI': RGB_TI,
            'NI_TI': NI_TI,
            'RGB_NI_TI': RGB_NI_TI
        }

        results, attentions = self.haqn_model(features_dict)
        RGB_special = results['RGB']
        NI_special = results['NI']
        TI_special = results['TI']
        RN_shared = results['RGB_NI']
        RT_shared = results['RGB_TI']
        NT_shared = results['NI_TI']
        RNT_shared = results['RGB_NI_TI']
        # RGB_special = self.haqn_model(RGB)
        # NI_special = self.haqn_model(NI)
        # TI_special = self.haqn_model(TI)
        # RN_shared = self.haqn_model(RGB_NI)
        # RT_shared = self.haqn_model(RGB_TI)
        # NT_shared = self.haqn_model(NI_TI)
        # RNT_shared = self.haqn_model(RGB_NI_TI)

        return RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared




    
    

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

        #从这里开始拿到的都是 BS 512 的特征也就是  B dIM
        # for single modality
        RGB_special = (self.r(r_embedding, RGB, RGB)[0]).permute(1, 2, 0).squeeze() #r_embedding, RGB, RGB 是 query, key, value, [0] 是 attn_output, 通用做法， permute(1, 2, 0) 是将 batch_size 放到最前面
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
        if self.combineway == 'deform':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMDeform(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'boq':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMboq(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'adaptiveboq':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMadaptiveboq(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'adaptiveboqdeform':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMadaptiveboqDeform(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'adaptiveboqdeformablation':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMadaptiveboqDeformAblation(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'newablation':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMnewablation(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'sedeform':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMseDeform(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'multimodelse':
            if not self.UsingDiversity:
                RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMmultimodelse(
                    RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
            else:
                RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared, lossdiversity = self.forward_HDMmultimodelse(
                    RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'multimodeltokense':
            if not self.UsingDiversity:
                RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMmultimodeltokense(
                    RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
            else:
                RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared, lossdiversity = self.forward_HDMmultimodeltokense(
                    RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)

        elif self.combineway == 'ebblockdeform':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMebblockDeform(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'rwkvadd':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMrw(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'rwkvaddlinear':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMrwpluslinear(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'rwkvcross':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMcrossrwkv(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'EMA':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMema(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
        elif self.combineway == 'deema':
            RGB_special, NI_special, TI_special, RN_shared, RT_shared, NT_shared, RNT_shared = self.forward_HDMDeema(
                RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
            
        else:
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
                if not self.UsingDiversity:
                    return moe_feat, loss_reg
                else:
                    return moe_feat, loss_reg+lossdiversity
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
