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
import textwrap

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp
import os
from mmcv.runner.base_module import BaseModule, ModuleList


# logger = logging.getLogger(__name__)

T_MAX = 256
HEAD_SIZE = 64
userwkvblock = False
if userwkvblock:
    from torch.utils.cpp_extension import load

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
                      in range(self.head)] #按head 分块，每个head 有7个输入
        query = torch.cat([x1, x2, x3, x4, x5, x6, x7], dim=-1)
        key = torch.stack([x1, x2, x3, x4, x5, x6, x7], dim=1)
        gate_heads = self.gating_network(query, key) # Multi-HeadAttentionGating 似乎思这个，就是拿 query 和key 生成的attn作为门控
        expert_outputs = [expert(head_input[i], gate_heads[:, i]) for i, expert in enumerate(self.experts)]
        outputs = torch.cat(expert_outputs, dim=-1).flatten(start_dim=1, end_dim=-1)
        loss = 0
        if self.training:
            return outputs, loss
        return outputs




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


class GeneralFusion(nn.Module):
    def __init__(self, feat_dim, num_experts, head, reg_weight=0.1, dropout=0.1, cfg=None):
        super(GeneralFusion, self).__init__()
        self.reg_weight = reg_weight
        self.feat_dim = feat_dim
        self.datasetsname = cfg.DATASETS.NAMES

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



        self.combineway = 'rwkvcross'
        print('combineway:', self.combineway)
        logger = logging.getLogger("DeMo")
        logger.info(f'combineway: {self.combineway}')
        # loggernew = logging.getLogger("DeMo")
        # loggernew.info(f'combineway: {self.combineway}')

        if self.combineway == 'deform':
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
