from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from modeling.backbones.vit_pytorch import trunc_normal_


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0,
                                                                               1)  # NCHW -> (HW)NC  #32,2048,7,7 ->49, 32, 2048
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  50,32,2048
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1)
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        xproj = self.attnpool(x4)

        return x3, x4, xproj


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, pattern=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.begin = -1
        if pattern is not None:
            if "prompt" in pattern:
                self.k = 4
                dropout = 0.0
                self.adapter_prompt_rgb = nn.Parameter(torch.zeros(self.k, d_model))
                self.adapter_prompt_nir = nn.Parameter(torch.zeros(self.k, d_model))
                self.adapter_prompt_tir = nn.Parameter(torch.zeros(self.k, d_model))
                self.adapter_transfer = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                                      QuickGELU(),
                                                      nn.Dropout(dropout),
                                                      nn.Linear(int(d_model // 2), int(d_model)))
                self.adapter_r = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                               QuickGELU(),
                                               nn.Dropout(dropout),
                                               nn.Linear(int(d_model // 2), int(d_model)))
                self.adapter_n = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                               QuickGELU(),
                                               nn.Dropout(dropout),
                                               nn.Linear(int(d_model // 2), int(d_model)))
                self.adapter_t = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                               QuickGELU(),
                                               nn.Dropout(dropout),
                                               nn.Linear(int(d_model // 2), int(d_model)))
            if "adapter" in pattern:
                self.adapter_ffn = nn.Sequential(nn.Linear(d_model, int(d_model // 2)),
                                                 QuickGELU(),
                                                 nn.Linear(int(d_model // 2), int(d_model)))


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward_ori(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_adapter(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        adapter_ffn = self.adapter_ffn(x)
        x = x + self.mlp(self.ln_2(x)) + adapter_ffn
        return x

    def forward_with_prompt_only_first_layer(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            if index == 0:
                n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                    self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
                t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                    self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                r = last_prompt + self.adapter_transfer(last_prompt)
                n2r = last_prompt
                t2r = last_prompt
            else:
                r = last_prompt
                n2r = last_prompt
                t2r = last_prompt
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            if index == 0:
                r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                    self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
                t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                    self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                n = last_prompt + self.adapter_transfer(last_prompt)
                r2n = last_prompt
                t2n = last_prompt
            else:
                n = last_prompt
                r2n = last_prompt
                t2n = last_prompt
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            if index == 0:
                r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                    self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
                n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                    self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            elif index == 1:
                t = last_prompt + self.adapter_transfer(last_prompt)
                r2t = last_prompt
                n2t = last_prompt
            else:
                t = last_prompt
                r2t = last_prompt
                n2t = last_prompt
            x = torch.cat([x, r2t, n2t, t], dim=0)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward_with_prompt(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                r = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_rgb.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                n = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_nir.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                t = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_tir.unsqueeze(
                    1).expand(-1, x.shape[1], -1))
            else:
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2t, n2t, t], dim=0)
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward_with_prompt_adapter(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        if modality == 'rgb':
            n2r = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2r = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                r = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_rgb.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                r = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r, n2r, t2r], dim=0)
        elif modality == 'nir':
            r2n = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            t2n = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_t(
                self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                n = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_nir.unsqueeze(1).expand(
                    -1, x.shape[1], -1))
            else:
                n = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2n, n, t2n], dim=0)
        elif modality == 'tir':
            r2t = self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_r(
                self.adapter_prompt_rgb.unsqueeze(1).expand(-1, x.shape[1], -1))
            n2t = self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1) + self.adapter_n(
                self.adapter_prompt_nir.unsqueeze(1).expand(-1, x.shape[1], -1))
            if last_prompt != None:
                t = (last_prompt + self.adapter_transfer(last_prompt) + self.adapter_prompt_tir.unsqueeze(
                    1).expand(-1, x.shape[1], -1))
            else:
                t = self.adapter_prompt_tir.unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, r2t, n2t, t], dim=0)

        x = x + self.attention(self.ln_1(x))
        adapter_ffn = self.adapter_ffn(x)
        x = x + self.mlp(self.ln_2(x)) + adapter_ffn
        prompt_current = (x[-3 * self.k:-2 * self.k] + x[-2 * self.k:-1 * self.k] + x[-1 * self.k:]) / 3
        if modality == 'rgb':
            return x[:-3 * self.k], prompt_current
        elif modality == 'nir':
            return x[:-3 * self.k], prompt_current
        elif modality == 'tir':
            return x[:-3 * self.k], prompt_current

    def forward(self, x: torch.Tensor, modality=None, index=None, last_prompt=None, prompt_sign=True,
                adapter_sign=True):
        if prompt_sign and adapter_sign:
            return self.forward_with_prompt_adapter(x, modality, index, last_prompt)
        elif prompt_sign and not adapter_sign:
            if index > self.begin:
                return self.forward_with_prompt(x, modality, index, last_prompt)
            else:
                return self.forward_ori(x), None
        elif not prompt_sign and adapter_sign:
            if index > self.begin:
                return self.forward_with_adapter(x)
            else:
                return self.forward_ori(x)
        else:
            # DeMo only use this branch
            return self.forward_ori(x)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, pattern=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, pattern) for _ in range(layers)])

    def forward(self, x: torch.Tensor, modality=None, index=None, last_prompt=None):
        return self.resblocks(x, modality, index, last_prompt)


# 原始InnovativeDFF模块（修改为LND格式）
class InnovativeDFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 多尺度平均池化 - 在token维度上进行
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(2)
        self.avg_pool3 = nn.AdaptiveAvgPool1d(4)

        # 注意力全连接层，替代卷积
        self.fc_atten1 = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.Sigmoid()
        )
        self.fc_atten2 = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.Sigmoid()
        )
        self.fc_atten3 = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.Sigmoid()
        )

        # 门控机制
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.Sigmoid()
        )

        # 通道减少全连接层
        self.fc_redu = nn.Linear(dim * 2, dim, bias=False)

        # 残差分支
        self.residual_fc = nn.Linear(dim, dim, bias=False)

        # 两个全连接层用于计算注意力
        self.fc1 = nn.Linear(dim, 1, bias=True)
        self.fc2 = nn.Linear(dim, 1, bias=True)
        # Sigmoid 激活函数
        self.nonlin = nn.Sigmoid()

    def forward(self, x, skip):
        """
        Args:
            x: (L, N, D) - 第一个输入
            skip: (L, N, D) - 第二个输入
        Returns:
            output: (L, N, D) - 融合后的输出
        """
        L, N, D = x.shape

        # 沿着特征维度拼接输入特征
        output = torch.cat([x, skip], dim=-1)  # (L, N, 2*D)

        # 多尺度平均池化及注意力计算
        # 需要转置为 (N, 2*D, L) 进行池化，然后转回 (L, N, 2*D)
        output_transposed = output.permute(1, 2, 0)  # (N, 2*D, L)

        att1_pooled = self.avg_pool1(output_transposed)  # (N, 2*D, 1)
        att2_pooled = self.avg_pool2(output_transposed)  # (N, 2*D, 2)
        att3_pooled = self.avg_pool3(output_transposed)  # (N, 2*D, 4)

        # 转回 (L_pool, N, 2*D) 格式
        att1_pooled = att1_pooled.permute(2, 0, 1)  # (1, N, 2*D)
        att2_pooled = att2_pooled.permute(2, 0, 1)  # (2, N, 2*D)
        att3_pooled = att3_pooled.permute(2, 0, 1)  # (4, N, 2*D)

        # 计算注意力权重
        att1 = self.fc_atten1(att1_pooled)  # (1, N, 2*D)
        att2 = self.fc_atten2(att2_pooled)  # (2, N, 2*D)
        att3 = self.fc_atten3(att3_pooled)  # (4, N, 2*D)

        # 上采样使长度一致 - 使用插值
        att2 = F.interpolate(att2.permute(1, 2, 0), size=L, mode='linear', align_corners=True).permute(2, 0, 1)
        att3 = F.interpolate(att3.permute(1, 2, 0), size=L, mode='linear', align_corners=True).permute(2, 0, 1)
        att1 = F.interpolate(att1.permute(1, 2, 0), size=L, mode='linear', align_corners=True).permute(2, 0, 1)

        # 融合多尺度注意力
        att = att1 + att2 + att3  # (L, N, 2*D)

        # 门控机制
        gate = self.gate_fc(output)  # (L, N, 2*D)
        att = att * gate

        # 应用注意力权重
        output = output * att  # (L, N, 2*D)

        # 减少通道数量
        output = self.fc_redu(output)  # (L, N, D)

        # 残差分支
        residual = self.residual_fc(x)  # (L, N, D)
        output = output + residual

        # 计算另一个注意力权重
        att_x = self.fc1(x)  # (L, N, 1)
        att_skip = self.fc2(skip)  # (L, N, 1)
        att = att_x + att_skip  # (L, N, 1)
        att = self.nonlin(att)

        # 应用另一个注意力权重
        output = output * att  # (L, N, D)
        return output


# 三输入直接融合DFF模块（LND格式）
class TripleInputDirectDFF(nn.Module):
    """
    三输入直接融合DFF模块 - LND格式
    输入：三个 (L, N, D) 张量
    输出：一个 (L, N, D) 张量
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 多尺度平均池化（适配3倍通道）
        self.avg_pool1 = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool1d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool1d(4)

        # 注意力全连接层（适配 dim * 3 输入）
        self.fc_atten1 = nn.Sequential(
            nn.Linear(dim * 3, dim * 3, bias=False),
            nn.LayerNorm(dim * 3),
            nn.Sigmoid()
        )
        self.fc_atten2 = nn.Sequential(
            nn.Linear(dim * 3, dim * 3, bias=False),
            nn.LayerNorm(dim * 3),
            nn.Sigmoid()
        )
        self.fc_atten4 = nn.Sequential(
            nn.Linear(dim * 3, dim * 3, bias=False),
            nn.LayerNorm(dim * 3),
            nn.Sigmoid()
        )

        # 自适应权重学习网络
        self.input_weight_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1)  # 输出3个权重
        )

        # 门控机制（适配3输入）
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 3, dim * 3, bias=False),
            nn.Sigmoid()
        )

        # 额外的门控增强
        self.gate_enhancement = nn.Sequential(
            nn.Linear(dim * 3, dim * 3),
            nn.Tanh()
        )

        # 通道减少：从 dim*3 减少到 dim
        self.fc_redu = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )

        # 三分支残差机制
        self.residual_fc1 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim)
        )
        self.residual_fc2 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim)
        )
        self.residual_fc3 = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.LayerNorm(dim)
        )

        # 残差融合
        self.residual_fusion = nn.Sequential(
            nn.Linear(dim * 3, dim, bias=False),
            nn.LayerNorm(dim)
        )

        # 三路注意力计算
        self.fc1 = nn.Linear(dim, 1, bias=True)
        self.fc2 = nn.Linear(dim, 1, bias=True)
        self.fc3 = nn.Linear(dim, 1, bias=True)

        # 注意力融合
        self.attention_fusion = nn.Sequential(
            nn.Linear(3, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input1, input2, input3):
        """
        Args:
            input1, input2, input3: 三个输入特征 (L, N, D)
        Returns:
            output: 融合后的特征图 (L, N, D)
        """
        L, N, D = input1.shape

        # 直接拼接三个输入
        concat_features = torch.cat([input1, input2, input3], dim=-1)  # (L, N, 3*D)

        # 自适应权重学习 - 对每个token位置计算权重
        input_weights = self.input_weight_net(concat_features)  # (L, N, 3)
        w1, w2, w3 = input_weights[..., 0:1], input_weights[..., 1:2], input_weights[..., 2:3]

        # 应用权重到对应输入
        weighted_input1 = input1 * w1  # (L, N, D)
        weighted_input2 = input2 * w2  # (L, N, D)
        weighted_input3 = input3 * w3  # (L, N, D)
        weighted_concat = torch.cat([weighted_input1, weighted_input2, weighted_input3], dim=-1)  # (L, N, 3*D)

        # 多尺度平均池化及注意力计算
        # 转置进行池化：(L, N, 3*D) -> (N, 3*D, L)
        weighted_concat_transposed = weighted_concat.permute(1, 2, 0)

        att1_pooled = self.avg_pool1(weighted_concat_transposed).permute(2, 0, 1)  # (1, N, 3*D)
        att2_pooled = self.avg_pool2(weighted_concat_transposed).permute(2, 0, 1)  # (2, N, 3*D)
        att4_pooled = self.avg_pool4(weighted_concat_transposed).permute(2, 0, 1)  # (4, N, 3*D)

        # 计算注意力
        att1 = self.fc_atten1(att1_pooled)
        att2 = self.fc_atten2(att2_pooled)
        att4 = self.fc_atten4(att4_pooled)

        # 上采样使长度一致
        att1 = F.interpolate(att1.permute(1, 2, 0), size=L, mode='linear', align_corners=True).permute(2, 0, 1)
        att2 = F.interpolate(att2.permute(1, 2, 0), size=L, mode='linear', align_corners=True).permute(2, 0, 1)
        att4 = F.interpolate(att4.permute(1, 2, 0), size=L, mode='linear', align_corners=True).permute(2, 0, 1)

        # 融合多尺度注意力
        multi_scale_att = 0.5 * att1 + 0.3 * att2 + 0.2 * att4

        # 增强门控机制
        gate = self.gate_fc(weighted_concat)
        gate_enhancement = self.gate_enhancement(weighted_concat)
        combined_gate = gate * (1 + 0.1 * gate_enhancement)

        enhanced_att = multi_scale_att * combined_gate

        # 应用注意力权重
        attended_features = weighted_concat * enhanced_att

        # 减少通道数量
        reduced_features = self.fc_redu(attended_features)  # (L, N, D)

        # 三分支残差计算
        res1 = self.residual_fc1(input1)
        res2 = self.residual_fc2(input2)
        res3 = self.residual_fc3(input3)
        combined_residual = self.residual_fusion(torch.cat([res1, res2, res3], dim=-1))

        # 残差连接
        output = reduced_features + combined_residual

        # 三路注意力计算和融合
        att1_individual = self.fc1(input1)  # (L, N, 1)
        att2_individual = self.fc2(input2)  # (L, N, 1)
        att3_individual = self.fc3(input3)  # (L, N, 1)

        combined_attention = torch.cat([att1_individual, att2_individual, att3_individual], dim=-1)  # (L, N, 3)
        final_attention = self.attention_fusion(combined_attention)  # (L, N, 1)

        # 应用最终注意力权重
        output = output * final_attention

        return output


# 四输入层级融合DFF模块（LND格式）
class QuadInputHierarchicalDFF(nn.Module):
    """
    四输入层级融合DFF模块 - LND格式
    输入：四个 (L, N, D) 张量
    输出：一个 (L, N, D) 张量
    架构：(input1 + input2 + input3) → stage1_result，然后 stage1_result + input4 → final_output
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 第一阶段：使用三输入DFF融合前三个输入
        self.stage1_triple_dff = TripleInputDirectDFF(dim)

        # 第一阶段结果的特征增强
        self.stage1_enhancement = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            # 类似SE注意力的全局信息聚合
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

        # 第四输入的专门预处理
        self.input4_preprocessing = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

        # 跨阶段注意力机制
        self.cross_stage_attention = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 8, dim),
            nn.Sigmoid()
        )

        # 自适应权重学习
        self.adaptive_fusion_weight = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 2),
            nn.Softmax(dim=-1)
        )

        # 第二阶段：使用原始DFF融合增强后的特征和第四输入
        self.stage2_dff = InnovativeDFF(dim)

        # 多级残差记忆机制
        self.memory_input1 = nn.Linear(dim, dim, bias=False)
        self.memory_input4 = nn.Linear(dim, dim, bias=False)
        self.memory_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.LayerNorm(dim)
        )

        # 全局上下文增强
        self.global_context = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )

        # 最终输出精炼
        self.final_refinement = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, input1, input2, input3, input4):
        """
        Args:
            input1, input2, input3, input4: 四个输入特征 (L, N, D)
        Returns:
            final_output: 融合后的特征图 (L, N, D)
        """
        # 第一阶段：使用三输入DFF融合前三个输入
        stage1_result = self.stage1_triple_dff(input1, input2, input3)  # (L, N, D)

        # 对第一阶段结果进行特征增强
        # 计算全局平均作为增强权重
        global_avg = stage1_result.mean(dim=0, keepdim=True)  # (1, N, D)
        enhancement_weight = self.stage1_enhancement(global_avg)  # (1, N, D)
        enhanced_stage1 = stage1_result * enhancement_weight  # (L, N, D)

        # 第四输入预处理
        processed_input4 = self.input4_preprocessing(input4)  # (L, N, D)

        # 跨阶段注意力：让第一阶段结果指导第四输入
        cross_attention = self.cross_stage_attention(enhanced_stage1)  # (L, N, D)
        attended_input4 = processed_input4 * cross_attention  # (L, N, D)

        # 自适应权重学习
        weight_input = torch.cat([enhanced_stage1, attended_input4], dim=-1)  # (L, N, 2*D)
        fusion_weights = self.adaptive_fusion_weight(weight_input)  # (L, N, 2)
        w1, w2 = fusion_weights[..., 0:1], fusion_weights[..., 1:2]  # (L, N, 1) each

        # 应用自适应权重
        weighted_stage1 = enhanced_stage1 * w1
        weighted_input4 = attended_input4 * w2

        # 第二阶段：使用原始DFF融合
        stage2_result = self.stage2_dff(weighted_stage1, weighted_input4)  # (L, N, D)

        # 全局上下文增强
        global_context = stage2_result.mean(dim=0, keepdim=True)  # (1, N, D)
        global_att = self.global_context(global_context)  # (1, N, D)
        context_enhanced = stage2_result * global_att  # (L, N, D)

        # 多级残差记忆
        memory1 = self.memory_input1(input1)  # (L, N, D)
        memory4 = self.memory_input4(input4)  # (L, N, D)
        combined_memory = self.memory_fusion(torch.cat([memory1, memory4], dim=-1))  # (L, N, D)

        # 最终融合
        final_output = context_enhanced + combined_memory
        final_output = self.final_refinement(final_output)

        return final_output
#------------------------- menkong
class GatedModalityEnhancement(nn.Module):
    """
    门控模态增强模块
    将融合特征y分别与各个单模态特征(x_rgb, x_nir, x_tir)进行门控融合，
    生成增强的单模态特征

    创新点：
    1. 自适应门控：学习每个模态与融合特征的最优融合权重
    2. 跨模态注意力：融合特征指导单模态特征的增强
    3. 残差连接：保持原始单模态信息
    4. 多层次融合：通道级和空间级双重门控
    5. 模态特异性：为每个模态设计独立的门控机制
    """

    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio

        # 为每个模态创建独立的门控网络
        self.rgb_gating = ModalityGatingUnit(dim, reduction_ratio, modality_name="RGB")
        self.nir_gating = ModalityGatingUnit(dim, reduction_ratio, modality_name="NIR")
        self.tir_gating = ModalityGatingUnit(dim, reduction_ratio, modality_name="TIR")

        # 全局协调门控：协调三个模态的增强程度
        self.global_coordination = GlobalCoordinationGate(dim)

    def forward(self, x_rgb, x_nir, x_tir, y_fused):
        """
        Args:
            x_rgb: RGB模态特征 (L, N, D)
            x_nir: NIR模态特征 (L, N, D)
            x_tir: TIR模态特征 (L, N, D)
            y_fused: 融合特征 (L, N, D)
        Returns:
            enhanced_rgb: 增强的RGB特征 (L, N, D)
            enhanced_nir: 增强的NIR特征 (L, N, D)
            enhanced_tir: 增强的TIR特征 (L, N, D)
        """

        # 各模态独立门控增强
        enhanced_rgb = self.rgb_gating(x_rgb, y_fused)
        enhanced_nir = self.nir_gating(x_nir, y_fused)
        enhanced_tir = self.tir_gating(x_tir, y_fused)

        # 全局协调门控：平衡三个模态的增强效果
        enhanced_rgb, enhanced_nir, enhanced_tir = self.global_coordination(
            enhanced_rgb, enhanced_nir, enhanced_tir, x_rgb, x_nir, x_tir
        )

        return enhanced_rgb, enhanced_nir, enhanced_tir


class ModalityGatingUnit(nn.Module):
    """
    单模态门控单元
    为特定模态设计的门控融合机制
    """

    def __init__(self, dim, reduction_ratio=4, modality_name=""):
        super().__init__()
        self.dim = dim
        self.modality_name = modality_name

        # 1. 跨模态注意力门控
        self.cross_modal_attention = CrossModalAttentionGate(dim)

        # 2. 通道级门控网络 - 修复版本
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim),
            nn.Sigmoid()
        )

        # 3. 空间级门控网络（token级）
        self.spatial_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 融合单模态和多模态特征
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # 4. 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        # 5. 自适应权重学习
        self.adaptive_weight = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 2),  # 输出两个权重：原始特征权重 + 融合特征权重
            nn.Softmax(dim=-1)
        )

        # 6. 残差连接的权重
        self.residual_weight = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x_modal, y_fused):
        """
        Args:
            x_modal: 单模态特征 (L, N, D)
            y_fused: 融合特征 (L, N, D)
        Returns:
            enhanced_modal: 增强的单模态特征 (L, N, D)
        """
        L, N, D = x_modal.shape

        # 1. 跨模态注意力：融合特征指导单模态特征
        attended_modal = self.cross_modal_attention(x_modal, y_fused)

        # 2. 通道级门控 - 修复版本
        # 计算全局特征表示
        global_modal = x_modal.mean(dim=0, keepdim=True)  # (1, N, D)
        global_fused = y_fused.mean(dim=0, keepdim=True)  # (1, N, D)
        combined_global = global_modal + global_fused  # (1, N, D)

        # 直接通过线性层计算通道权重
        channel_weights = self.channel_gate(combined_global)  # (1, N, D)

        # 应用通道门控
        channel_gated = attended_modal * channel_weights  # (L, N, D)

        # 3. 空间级门控（token级）
        # 拼接单模态和融合特征用于空间门控
        spatial_input = torch.cat([x_modal, y_fused], dim=-1)  # (L, N, 2*D)
        spatial_weights = self.spatial_gate(spatial_input)  # (L, N, 1)

        # 应用空间门控
        spatial_gated = channel_gated * spatial_weights  # (L, N, D)

        # 4. 特征融合
        fusion_input = torch.cat([spatial_gated, y_fused], dim=-1)  # (L, N, 2*D)
        fused_features = self.feature_fusion(fusion_input)  # (L, N, D)

        # 5. 自适应权重融合
        weight_input = torch.cat([x_modal, fused_features], dim=-1)  # (L, N, 2*D)
        adaptive_weights = self.adaptive_weight(weight_input)  # (L, N, 2)
        w_original, w_fused = adaptive_weights[..., 0:1], adaptive_weights[..., 1:2]

        # 加权融合
        weighted_fusion = w_original * x_modal + w_fused * fused_features

        # 6. 残差连接
        enhanced_modal = weighted_fusion + self.residual_weight * x_modal

        return enhanced_modal


class CrossModalAttentionGate(nn.Module):
    """
    跨模态注意力门控
    使用融合特征作为Query，单模态特征作为Key和Value
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.scale = self.head_dim ** -0.5

        # Query从融合特征生成，Key和Value从单模态特征生成
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x_modal, y_fused):
        """
        Args:
            x_modal: 单模态特征 (L, N, D)
            y_fused: 融合特征 (L, N, D)
        Returns:
            attended_modal: 注意力增强的单模态特征 (L, N, D)
        """
        L, N, D = x_modal.shape

        # 生成Query, Key, Value
        Q = self.q_proj(y_fused)  # 使用融合特征作为Query
        K = self.k_proj(x_modal)  # 使用单模态特征作为Key
        V = self.v_proj(x_modal)  # 使用单模态特征作为Value

        # 重塑为多头注意力格式
        Q = Q.view(L, N, self.num_heads, self.head_dim).transpose(1, 2)  # (L, num_heads, N, head_dim)
        K = K.view(L, N, self.num_heads, self.head_dim).transpose(1, 2)  # (L, num_heads, N, head_dim)
        V = V.view(L, N, self.num_heads, self.head_dim).transpose(1, 2)  # (L, num_heads, N, head_dim)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (L, num_heads, N, N)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重
        attended = torch.matmul(attention_weights, V)  # (L, num_heads, N, head_dim)

        # 重塑回原始格式
        attended = attended.transpose(1, 2).contiguous().view(L, N, D)  # (L, N, D)

        # 输出投影
        attended_modal = self.out_proj(attended)

        return attended_modal


class GlobalCoordinationGate(nn.Module):
    """
    全局协调门控
    协调三个模态的增强效果，确保模态间的平衡
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 模态间相互作用网络
        self.interaction_net = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        # 平衡权重网络
        self.balance_net = nn.Sequential(
            nn.Linear(dim * 6, dim),  # 增强特征 + 原始特征
            nn.ReLU(inplace=True),
            nn.Linear(dim, 3),  # 三个模态的平衡权重
            nn.Softmax(dim=-1)
        )

        # 全局调制门控
        self.global_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, enhanced_rgb, enhanced_nir, enhanced_tir,
                original_rgb, original_nir, original_tir):
        """
        Args:
            enhanced_*: 增强的模态特征 (L, N, D)
            original_*: 原始的模态特征 (L, N, D)
        Returns:
            coordinated_*: 协调后的模态特征 (L, N, D)
        """

        # 1. 计算模态间相互作用
        enhanced_concat = torch.cat([enhanced_rgb, enhanced_nir, enhanced_tir], dim=-1)
        interaction_features = self.interaction_net(enhanced_concat)  # (L, N, D)

        # 2. 计算平衡权重
        all_features = torch.cat([
            enhanced_rgb, enhanced_nir, enhanced_tir,
            original_rgb, original_nir, original_tir
        ], dim=-1)  # (L, N, 6*D)

        # 全局平均池化
        global_features = all_features.mean(dim=0, keepdim=True)  # (1, N, 6*D)
        balance_weights = self.balance_net(global_features)  # (1, N, 3)
        w_rgb, w_nir, w_tir = balance_weights[..., 0:1], balance_weights[..., 1:2], balance_weights[..., 2:3]

        # 3. 全局调制
        global_modulation = self.global_gate(interaction_features)  # (L, N, 1)

        # 4. 应用协调
        coordinated_rgb = enhanced_rgb * w_rgb * global_modulation + original_rgb * (1 - global_modulation)
        coordinated_nir = enhanced_nir * w_nir * global_modulation + original_nir * (1 - global_modulation)
        coordinated_tir = enhanced_tir * w_tir * global_modulation + original_tir * (1 - global_modulation)

        return coordinated_rgb, coordinated_nir, coordinated_tir

class SimpleModalityGate(nn.Module):
    """简化的单模态门控"""

    def __init__(self, dim):
        super().__init__()

        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # 权重学习
        self.weight_net = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x_modal, y_fused):
        # 拼接特征
        concat_features = torch.cat([x_modal, y_fused], dim=-1)  # (L, N, 2*D)

        # 融合特征
        fused = self.fusion_net(concat_features)  # (L, N, D)

        # 计算门控权重
        gate = self.gate_net(concat_features)  # (L, N, D)
        gated_fused = fused * gate

        # 计算混合权重
        weights = self.weight_net(concat_features)  # (L, N, 2)
        w1, w2 = weights[..., 0:1], weights[..., 1:2]

        # 最终融合
        enhanced = w1 * x_modal + w2 * gated_fused

        return enhanced


class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int,
                 heads: int, output_dim: int, cfg: dict):
        super().__init__()
        self.prompt_sign = cfg.MODEL.PROMPT
        self.adapter_sign = cfg.MODEL.ADAPTER
        self.pattern = ['nothing']
        if self.prompt_sign:
            self.pattern.append('prompt')
        if self.adapter_sign:
            self.pattern.append('adapter')
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size,
                               bias=False)

        scale = width ** -0.5

        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution * w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, pattern=self.pattern)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # 原来的代码:
        # self.triple_dff = TripleInputDirectDFF(width)
        # self.quad_dffs = QuadInputHierarchicalDFF(width)

        # 替换为优化版本:
        self.triple_dff = OptimizedTripleInputDFF(width)
        self.quad_dffs = OptimizedQuadInputDFF(width)

        # 或者替换为超轻量级版本（最快）:
        # self.triple_dff = UltraLightTripleDFF(width)
        # self.quad_dffs = UltraLightQuadDFF(width)

        #self.gated_enhancement = GatedModalityEnhancement(width)
        self.gated_enhancement = GatedModalityEnhancement(width)

    def forward(self, x: torch.Tensor, cv_emb=None, modality=None):
        #ientify if input is dict
        if isinstance(x, dict):
            x_rgb = x['RGB']
            x_nir = x['NI']
            x_tir = x['TI']

            x_rgb = self.conv1(x_rgb)  # shape = [*, width, grid, grid]
            x_nir = self.conv1(x_nir)  # shape = [*, width, grid, grid]
            x_tir = self.conv1(x_tir)

            x_rgb = x_rgb.reshape(x_rgb.shape[0], x_rgb.shape[1], -1)
            x_nir = x_nir.reshape(x_nir.shape[0], x_nir.shape[1], -1)
            x_tir = x_tir.reshape(x_tir.shape[0], x_tir.shape[1], -1)

            x_rgb = x_rgb.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x_nir = x_nir.permute(0, 2, 1)
            x_tir = x_tir.permute(0, 2, 1)

            x_rgb = torch.cat(
                [self.class_embedding.to(x_rgb.dtype) + torch.zeros(x_rgb.shape[0], 1, x_rgb.shape[-1], dtype=x_rgb.dtype, device=x_rgb.device),
                 x_rgb], dim=1)
            x_nir = torch.cat(
                [self.class_embedding.to(x_nir.dtype) + torch.zeros(x_nir.shape[0], 1, x_nir.shape[-1], dtype=x_nir.dtype, device=x_nir.device),
                 x_nir], dim=1)
            x_tir = torch.cat(
                [self.class_embedding.to(x_tir.dtype) + torch.zeros(x_tir.shape[0], 1, x_tir.shape[-1], dtype=x_tir.dtype, device=x_tir.device),
                 x_tir], dim=1)
            if cv_emb is not None:
                x_rgb[:, 0] = x_rgb[:, 0] + cv_emb.squeeze(1)
                x_nir[:, 0] = x_nir[:, 0] + cv_emb.squeeze(1)
                x_tir[:, 0] = x_tir[:, 0] + cv_emb.squeeze(1)
            x_rgb = x_rgb + self.positional_embedding.to(x_rgb.dtype)
            x_nir = x_nir + self.positional_embedding.to(x_nir.dtype)
            x_tir = x_tir + self.positional_embedding.to(x_tir.dtype)

            x_rgb = self.ln_pre(x_rgb)
            x_nir = self.ln_pre(x_nir)
            x_tir = self.ln_pre(x_tir)

            x_rgb = x_rgb.permute(1, 0, 2)  # NLD -> LND (LND = tokenamount,batch,dim)
            x_nir = x_nir.permute(1, 0, 2)
            x_tir = x_tir.permute(1, 0, 2)

            # 初始化融合结果
            #y = torch.randn(x_rgb.shape[0], x_rgb.shape[1], x_rgb.shape[2], device=x_rgb.device, dtype=x_rgb.dtype)

            for i in range(len(self.transformer.resblocks)):
                x_rgb = self.transformer.resblocks[i](x_rgb, modality, i, None, prompt_sign=False, adapter_sign=False)
                x_nir = self.transformer.resblocks[i](x_nir, modality, i, None, prompt_sign=False, adapter_sign=False)
                x_tir = self.transformer.resblocks[i](x_tir, modality, i, None, prompt_sign=False, adapter_sign=False)
                # DFF融合逻辑
                if i == 0:
                    # 第一层：三输入融合
                    y = self.triple_dff(x_rgb, x_nir, x_tir)
                else:
                    # 后续层：四输入融合（包含上一层结果）
                    y = self.quad_dffs(x_rgb, x_nir, x_tir, y)

            x_rgb, x_nir, x_tir = self.gated_enhancement(x_rgb, x_nir, x_tir, y)
            # x_rgb = self.gated_enhancement(x_rgb, y)
            # x_nir = self.gated_enhancement(x_nir, y)
            # x_tir = self.gated_enhancement(x_tir, y)

            # x_rgb = x_rgb + y
            # x_nir = x_nir + y
            # x_tir = x_tir + y

            x_rgb = x_rgb.permute(1, 0, 2)
            x_nir = x_nir.permute(1, 0, 2)
            x_tir = x_tir.permute(1, 0, 2)
            x_rgb = self.ln_post(x_rgb)
            x_nir = self.ln_post(x_nir)
            x_tir = self.ln_post(x_tir)
            if self.proj is not None:
                x_rgb_proj = x_rgb @ self.proj
                x_nir_proj = x_nir @ self.proj
                x_tir_proj = x_tir @ self.proj
            return {'RGB': x_rgb_proj, 'NI': x_nir_proj, 'TI': x_tir_proj}

        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            if cv_emb != None:
                x[:, 0] = x[:, 0] + cv_emb.squeeze(1)
            x = x + self.positional_embedding.to(x.dtype)

            x = self.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND (LND = tokenamount,batch,dim)
            for i in range(len(self.transformer.resblocks)):
                if self.prompt_sign and self.adapter_sign:
                    if i == 0:
                        x, last_prompt = self.transformer.resblocks[i](x, modality, i, None, prompt_sign=True,
                                                                       adapter_sign=True)
                    else:
                        x, last_prompt = self.transformer.resblocks[i](x, modality, i, last_prompt, prompt_sign=True,
                                                                       adapter_sign=True)
                elif self.prompt_sign and not self.adapter_sign:
                    if i == 0:
                        x, last_prompt = self.transformer.resblocks[i](x, modality, i, None, prompt_sign=True,
                                                                       adapter_sign=False)
                    else:
                        x, last_prompt = self.transformer.resblocks[i](x, modality, i, last_prompt, prompt_sign=True,
                                                                       adapter_sign=False)
                elif not self.prompt_sign and self.adapter_sign:
                    x = self.transformer.resblocks[i](x, modality, i, None, prompt_sign=False, adapter_sign=True)
                else:
                    x = self.transformer.resblocks[i](x, modality, i, None, prompt_sign=False, adapter_sign=False)

            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_post(x)
            if self.proj is not None:
                xproj = x @ self.proj
            return xproj




# 优化后的高效InnovativeDFF - 移除所有插值操作
class OptimizedInnovativeDFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 简化的全局注意力 - 避免多尺度池化和插值
        self.global_attention = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.LayerNorm(dim * 2),
            nn.Sigmoid()
        )

        # 简化的门控机制
        self.gate_fc = nn.Sequential(
            nn.Linear(dim * 2, dim * 2, bias=False),
            nn.Sigmoid()
        )

        # 通道减少
        self.fc_redu = nn.Linear(dim * 2, dim, bias=False)

        # 残差分支
        self.residual_fc = nn.Linear(dim, dim, bias=False)

        # 最终注意力
        self.final_attention = nn.Sequential(
            nn.Linear(dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        """
        优化的前向传播 - 移除所有插值操作
        Args:
            x: (L, N, D) - 第一个输入
            skip: (L, N, D) - 第二个输入
        Returns:
            output: (L, N, D) - 融合后的输出
        """
        # 拼接输入特征
        concat_features = torch.cat([x, skip], dim=-1)  # (L, N, 2*D)

        # 全局注意力 - 使用全局平均代替多尺度池化
        global_context = concat_features.mean(dim=0, keepdim=True)  # (1, N, 2*D)
        attention_weights = self.global_attention(global_context)  # (1, N, 2*D)

        # 门控机制
        gate_weights = self.gate_fc(concat_features)  # (L, N, 2*D)

        # 应用注意力和门控
        enhanced_features = concat_features * attention_weights * gate_weights

        # 通道减少
        reduced_features = self.fc_redu(enhanced_features)  # (L, N, D)

        # 残差连接
        residual = self.residual_fc(x)  # (L, N, D)
        output = reduced_features + residual

        # 最终注意力
        final_att = self.final_attention(concat_features)  # (L, N, 1)
        output = output * final_att

        return output


# 优化后的三输入DFF - 大幅简化
class OptimizedTripleInputDFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 权重学习网络
        self.weight_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1)
        )

        # 统一的特征融合 - 避免复杂的多尺度处理
        self.fusion_net = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )

        # 简化的注意力
        self.attention_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # 残差网络
        self.residual_nets = nn.ModuleList([
            nn.Linear(dim, dim, bias=False) for _ in range(3)
        ])

    def forward(self, input1, input2, input3):
        """
        优化的三输入融合 - 移除所有插值和复杂操作
        """
        # 拼接所有输入
        concat_features = torch.cat([input1, input2, input3], dim=-1)  # (L, N, 3*D)

        # 学习自适应权重
        weights = self.weight_net(concat_features)  # (L, N, 3)
        w1, w2, w3 = weights[..., 0:1], weights[..., 1:2], weights[..., 2:3]

        # 加权输入
        weighted_concat = torch.cat([
            input1 * w1, input2 * w2, input3 * w3
        ], dim=-1)  # (L, N, 3*D)

        # 特征融合
        fused_features = self.fusion_net(weighted_concat)  # (L, N, D)

        # 残差连接 - 简化版本
        residual_sum = sum([
            net(inp) for net, inp in zip(self.residual_nets, [input1, input2, input3])
        ]) / 3  # 平均残差

        output = fused_features + residual_sum

        # 最终注意力
        attention = self.attention_net(concat_features)  # (L, N, 1)
        output = output * attention

        return output


# 优化后的四输入DFF - 简化层级结构
class OptimizedQuadInputDFF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 使用优化的三输入DFF
        self.stage1_triple_dff = OptimizedTripleInputDFF(dim)

        # 简化的特征增强
        self.stage1_enhancement = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )

        # 简化的第四输入预处理
        self.input4_preprocessing = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

        # 权重学习
        self.adaptive_fusion_weight = nn.Sequential(
            nn.Linear(dim * 2, 2),
            nn.Softmax(dim=-1)
        )

        # 使用优化的DFF进行第二阶段融合
        self.stage2_dff = OptimizedInnovativeDFF(dim)

        # 简化的记忆机制
        self.memory_fusion = nn.Linear(dim * 2, dim, bias=False)

        # 简化的最终处理
        self.final_processing = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, input1, input2, input3, input4):
        """
        优化的四输入融合 - 简化所有复杂操作
        """
        # 第一阶段：三输入融合
        stage1_result = self.stage1_triple_dff(input1, input2, input3)  # (L, N, D)

        # 特征增强 - 使用全局平均
        global_avg = stage1_result.mean(dim=0, keepdim=True)  # (1, N, D)
        enhancement_weight = self.stage1_enhancement(global_avg)  # (1, N, D)
        enhanced_stage1 = stage1_result * enhancement_weight  # (L, N, D)

        # 第四输入预处理
        processed_input4 = self.input4_preprocessing(input4)  # (L, N, D)

        # 自适应权重学习
        weight_input = torch.cat([enhanced_stage1, processed_input4], dim=-1)  # (L, N, 2*D)
        fusion_weights = self.adaptive_fusion_weight(weight_input)  # (L, N, 2)
        w1, w2 = fusion_weights[..., 0:1], fusion_weights[..., 1:2]

        # 加权特征
        weighted_stage1 = enhanced_stage1 * w1
        weighted_input4 = processed_input4 * w2

        # 第二阶段：使用优化的DFF
        stage2_result = self.stage2_dff(weighted_stage1, weighted_input4)  # (L, N, D)

        # 记忆机制 - 简化版本
        memory = self.memory_fusion(torch.cat([input1, input4], dim=-1))  # (L, N, D)

        # 最终融合
        final_output = stage2_result + memory
        final_output = self.final_processing(final_output)

        return final_output




# 超轻量级版本 - 如果还需要更快的速度
class UltraLightDFF(nn.Module):
    """超轻量级DFF - 最大化速度"""

    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        concat_input = torch.cat([x, skip], dim=-1)
        fused = self.fusion(concat_input)
        gate_weight = self.gate(concat_input)
        return x + gate_weight * fused


class UltraLightTripleDFF(nn.Module):
    """超轻量级三输入DFF"""

    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )

        self.weight_net = nn.Sequential(
            nn.Linear(dim * 3, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x1, x2, x3):
        concat_input = torch.cat([x1, x2, x3], dim=-1)

        # 学习权重
        weights = self.weight_net(concat_input)  # (L, N, 3)
        w1, w2, w3 = weights[..., 0:1], weights[..., 1:2], weights[..., 2:3]

        # 加权平均
        weighted_avg = w1 * x1 + w2 * x2 + w3 * x3

        # 简单融合
        fused = self.fusion(concat_input)

        return weighted_avg + 0.1 * fused


class UltraLightQuadDFF(nn.Module):
    """超轻量级四输入DFF"""

    def __init__(self, dim):
        super().__init__()
        self.triple_dff = UltraLightTripleDFF(dim)
        self.final_dff = UltraLightDFF(dim)

    def forward(self, x1, x2, x3, x4):
        # 先融合前三个
        triple_result = self.triple_dff(x1, x2, x3)
        # 再与第四个融合
        final_result = self.final_dff(triple_result, x4)
        return final_result



class CLIP(nn.Module):
    def __init__(self, cfg,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 h_resolution: int,
                 w_resolution: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=h_resolution * w_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                h_resolution=h_resolution,
                w_resolution=w_resolution,
                patch_size=vision_patch_size,
                stride_size=vision_stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                cfg=cfg
            )



        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)


def build_model(cfg, state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:  # RN50
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]  # 77 (77,512)
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(cfg,
                 embed_dim,
                 image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
                 context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
                 h_resolution, w_resolution
                 )
    if vit: # resize the positional embedding in pth
        state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"],
                                                                     model.visual.positional_embedding, h_resolution,
                                                                     w_resolution,cfg)
    else:  # RN50
        state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(
            state_dict["visual.attnpool.positional_embedding"], model.visual.attnpool.positional_embedding,
            h_resolution, w_resolution)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    # model.load_state_dict(state_dict, strict=False)
    try:
        print(f"Successfully load ckpt!")
        incompatibleKeys = model.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)
    except Exception as e:
        print(f"Failed loading checkpoint!")
    return model.eval()


import math


def resize_pos_embed(posemb, posemb_new, hight, width,cfg=None):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)

    ntok_new = posemb_new.shape[0]  # 129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb
