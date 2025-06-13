import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
from typing import Dict, List, Optional, Tuple


class CMQEConfig:
    """
    CMQE消融实验配置类 - 控制各组件的开启/关闭
    """

    def __init__(self,
                 enable_hierarchical_queries=True,  # HQ: Progressive query learning (层次化查询学习)
                 enable_query_propagation=True,  # QP: Inter-layer query propagation (跨层查询传播)
                 enable_cross_modal_exchange=True,  # CMQE: Cross-Modal Query Exchange (跨模态查询交换)
                 enable_adaptive_fusion=True,  # AF: Adaptive feature fusion (自适应特征融合)
                 exchange_ratio=0.2):  # CMQE交换强度
        self.enable_hierarchical_queries = enable_hierarchical_queries
        self.enable_query_propagation = enable_query_propagation
        self.enable_cross_modal_exchange = enable_cross_modal_exchange
        self.enable_adaptive_fusion = enable_adaptive_fusion
        self.exchange_ratio = exchange_ratio

    def get_ablation_name(self):
        """生成消融实验的名称标识"""
        components = []
        if self.enable_hierarchical_queries: components.append("HQ")
        if self.enable_query_propagation: components.append("QP")
        if self.enable_cross_modal_exchange: components.append("CMQE")
        if self.enable_adaptive_fusion: components.append("AF")
        return "_".join(components) if components else "CMQEBaseline"


class CMQEPositionalEncoding(nn.Module):
    """CMQE系统的位置编码"""

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


class CMQEBoQBlock(torch.nn.Module):
    """CMQE BoQ Block with Query State Access"""

    def __init__(self, in_dim, num_queries, nheads=8, layer_idx=0, total_layers=2,
                 prev_num_queries=None, cmqe_config=None):
        super(CMQEBoQBlock, self).__init__()

        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.num_queries = num_queries
        self.in_dim = in_dim
        self.config = cmqe_config if cmqe_config is not None else CMQEConfig()

        self.encoder = torch.nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=nheads, dim_feedforward=4 * in_dim,
            batch_first=True, dropout=0.)

        # 组件1: Progressive Query Learning (层次化查询学习)
        if self.config.enable_hierarchical_queries:
            init_scale = 1 * (1 + layer_idx * 0.5)
            print(f"[CMQE Ablation] ✓ Layer {layer_idx}: Hierarchical queries enabled with scale {init_scale:.3f}")
        else:
            init_scale = 1
            print(
                f"[CMQE Ablation] ✗ Layer {layer_idx}: Hierarchical queries disabled, using fixed scale {init_scale:.3f}")

        self.queries = torch.nn.Parameter(torch.randn(1, num_queries, in_dim) * init_scale)

        # 查询专门化投影
        self.query_projection = nn.Linear(in_dim, in_dim)

        # 组件2: Inter-layer Query Propagation (跨层查询传播)
        if self.config.enable_query_propagation and prev_num_queries is not None and prev_num_queries != num_queries:
            self.query_adapter = nn.Linear(prev_num_queries, num_queries)
            print(
                f"[CMQE Ablation] ✓ Layer {layer_idx}: Query propagation enabled with adapter ({prev_num_queries}->{num_queries})")
        elif self.config.enable_query_propagation:
            self.query_adapter = None
            print(f"[CMQE Ablation] ✓ Layer {layer_idx}: Query propagation enabled (no adapter needed)")
        else:
            self.query_adapter = None
            print(f"[CMQE Ablation] ✗ Layer {layer_idx}: Query propagation disabled")

        self.self_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = torch.nn.LayerNorm(in_dim)

        self.cross_attn = torch.nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = torch.nn.LayerNorm(in_dim)

        # 残差门控机制 (仅在查询传播启用时使用)
        if self.config.enable_query_propagation:
            self.gate = nn.Parameter(torch.ones(1))

    def forward(self, x, prev_queries=None, external_queries=None):
        """
        Args:
            x: input features [B, N, D]
            prev_queries: queries from previous layer [B, num_queries, D]
            external_queries: queries from CMQE [B, num_queries, D]
        """
        B = x.size(0)
        x = self.encoder(x)

        # 初始化查询
        if external_queries is not None:
            # 使用外部增强查询 (来自CMQE)
            q = external_queries
        else:
            q = self.queries.repeat(B, 1, 1)

        # 组件2: 层间查询传递
        if self.config.enable_query_propagation and prev_queries is not None and self.layer_idx > 0:
            if self.query_adapter is not None:
                adapted_queries = self.query_adapter(prev_queries.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                adapted_queries = prev_queries

            alpha = torch.sigmoid(self.gate)
            q = alpha * q + (1 - alpha) * adapted_queries

        # 查询专门化投影
        q = self.query_projection(q)

        # 自注意力
        q_self_attn = q + self.self_attn(q, q, q)[0]
        q_normed = self.norm_q(q_self_attn)

        # 交叉注意力
        out, attn = self.cross_attn(q_normed, x, x)
        out = self.norm_out(out)

        return x, out, attn.detach(), q_normed


class CrossModalQueryExchange(nn.Module):
    """
    跨模态查询交换机制 (CMQE)

    核心功能：
    1. 学习模态间相似性矩阵
    2. 智能跨模态查询信息交换
    3. 动态调节交换强度
    """

    def __init__(self, query_dim, num_heads=8, exchange_ratio=0.2):
        super().__init__()

        self.query_dim = query_dim
        self.num_heads = num_heads
        self.exchange_ratio = exchange_ratio

        # 跨模态注意力机制
        self.cross_modal_attn = nn.MultiheadAttention(
            query_dim, num_heads, batch_first=True, dropout=0.1
        )

        # 查询融合网络
        self.query_fusion = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(query_dim, query_dim)
        )

        # 交换强度控制 (可学习参数)
        self.exchange_gate = nn.Parameter(torch.tensor(exchange_ratio))

        # 模态相似性矩阵 (可学习) 7x7矩阵
        self.modality_similarity = nn.Parameter(torch.eye(7) + 0.1 * torch.randn(7, 7))

        # 查询大小适配器
        self.query_adapters = nn.ModuleDict()

    def _get_modality_index(self, modality):
        """获取模态在相似性矩阵中的索引"""
        mapping = {
            'RGB': 0, 'NI': 1, 'TI': 2,
            'RGB_NI': 3, 'RGB_TI': 4, 'NI_TI': 5, 'RGB_NI_TI': 6
        }
        return mapping.get(modality, 0)

    def _adapt_query_size(self, source_queries, target_num_queries):
        """适配不同模态间的查询数量差异"""
        B, N_source, D = source_queries.shape

        if N_source == target_num_queries:
            return source_queries

        # 创建或获取适配器
        adapter_key = f"{N_source}_to_{target_num_queries}"
        if adapter_key not in self.query_adapters:
            self.query_adapters[adapter_key] = nn.Linear(N_source, target_num_queries).to(source_queries.device)

        # 使用线性适配器调整查询数量
        adapted = self.query_adapters[adapter_key](source_queries.permute(0, 2, 1)).permute(0, 2, 1)
        return adapted

    def forward(self, query_dict):
        """
        Args:
            query_dict: {modality: queries} where queries is [B, N, D]
        Returns:
            enhanced_query_dict: 增强后的查询字典
        """
        enhanced_queries = {}

        # 计算模态相似性权重 (softmax归一化)
        similarity_matrix = torch.softmax(self.modality_similarity, dim=1)

        for modal, queries in query_dict.items():
            modal_idx = self._get_modality_index(modal)
            B, N, D = queries.shape

            # 获取与其他模态的相似性权重
            similarity_weights = similarity_matrix[modal_idx]

            # 收集来自其他模态的查询信息
            cross_modal_features = []
            valid_weights = []

            for other_modal, other_queries in query_dict.items():
                if other_modal != modal:
                    other_idx = self._get_modality_index(other_modal)
                    weight = similarity_weights[other_idx]

                    # 只考虑相似性权重大于阈值的模态
                    if weight > 0.1:
                        # 适配查询大小
                        adapted_queries = self._adapt_query_size(other_queries, N)
                        cross_modal_features.append(adapted_queries)
                        valid_weights.append(weight)

            if cross_modal_features and len(valid_weights) > 0:
                # 加权融合其他模态的查询
                valid_weights = torch.tensor(valid_weights, device=queries.device)
                valid_weights = valid_weights / valid_weights.sum()  # 归一化

                weighted_cross_queries = torch.zeros_like(queries)
                for cross_queries, weight in zip(cross_modal_features, valid_weights):
                    weighted_cross_queries += weight * cross_queries

                # 通过跨模态注意力增强原始查询
                enhanced_q, attn_weights = self.cross_modal_attn(
                    queries, weighted_cross_queries, weighted_cross_queries
                )

                # 查询融合
                concatenated = torch.cat([queries, enhanced_q], dim=-1)
                fused_queries = self.query_fusion(concatenated)

                # 门控机制：控制跨模态信息的融合比例
                gate_weight = torch.sigmoid(self.exchange_gate)
                final_queries = gate_weight * fused_queries + (1 - gate_weight) * queries

                enhanced_queries[modal] = final_queries
            else:
                # 没有有效的跨模态信息，保持原始查询
                enhanced_queries[modal] = queries

        return enhanced_queries


class CMQEAdaptiveBoQ(torch.nn.Module):
    """CMQE系统的自适应BoQ模块"""

    def __init__(self, input_dim=512, num_queries=32, num_layers=2, row_dim=32,
                 use_positional_encoding=True, cmqe_config=None):
        super().__init__()

        self.use_positional_encoding = use_positional_encoding
        self.norm_input = torch.nn.LayerNorm(input_dim)
        self.config = cmqe_config if cmqe_config is not None else CMQEConfig()
        self.input_dim = input_dim
        self.num_layers = num_layers

        if use_positional_encoding:
            self.pos_encoding = CMQEPositionalEncoding(input_dim)

        # 组件1: 层特异性的查询数量 (层次化查询学习)
        if self.config.enable_hierarchical_queries:
            layer_queries = self._get_layer_queries_hierarchical(num_queries, num_layers)
            print(f"[CMQE Ablation] ✓ Hierarchical queries: {layer_queries}")
        else:
            layer_queries = [num_queries] * num_layers
            print(f"[CMQE Ablation] ✗ Hierarchical queries disabled: {layer_queries}")

        self.layer_queries = layer_queries
        self.boqs = torch.nn.ModuleList()

        for i in range(num_layers):
            prev_queries = layer_queries[i - 1] if i > 0 else None
            self.boqs.append(
                CMQEBoQBlock(input_dim, layer_queries[i],
                             nheads=max(1, input_dim // 64),
                             layer_idx=i, total_layers=num_layers,
                             prev_num_queries=prev_queries,
                             cmqe_config=self.config)
            )

        # 组件4: 自适应特征融合
        total_query_outputs = sum(layer_queries)
        self.adaptive_fusion = CMQEAdaptiveFusion(input_dim, total_query_outputs, row_dim, self.config)

    def _get_layer_queries_hierarchical(self, base_queries, num_layers):
        """层次化查询分配 (组件1)"""
        if num_layers == 1:
            return [base_queries]

        queries_per_layer = []
        for i in range(num_layers):
            ratio = 1.0 - (i / (num_layers - 1)) * 0.3
            layer_q = max(8, int(base_queries * ratio))
            queries_per_layer.append(layer_q)

        return queries_per_layer

    def forward(self, x, enhanced_queries=None):
        """
        Args:
            x: input features [B, N, D]
            enhanced_queries: 来自CMQE的增强查询 (可选)
        """
        if self.use_positional_encoding:
            x = self.pos_encoding(x)
        x = self.norm_input(x)

        outs = []
        attns = []
        prev_queries = None

        for i, boq_layer in enumerate(self.boqs):
            # 如果第一层有增强查询，则使用；否则使用None
            external_q = enhanced_queries if i == 0 and enhanced_queries is not None else None

            x, out, attn, queries = boq_layer(x, prev_queries, external_q)
            outs.append(out)
            attns.append(attn)

            # 组件2: 查询传播控制
            if self.config.enable_query_propagation:
                prev_queries = queries
            else:
                prev_queries = None

        # 组件4: 自适应融合
        final_out = self.adaptive_fusion(outs)

        # 返回最终输出、注意力权重和第一层查询(用于CMQE)
        first_layer_queries = outs[0] if outs else None
        return final_out, attns, first_layer_queries


class CMQEAdaptiveFusion(nn.Module):
    """CMQE系统的自适应特征融合模块"""

    def __init__(self, feat_dim, total_queries, output_dim, cmqe_config=None):
        super().__init__()

        self.feat_dim = feat_dim
        self.total_queries = total_queries
        self.config = cmqe_config if cmqe_config is not None else CMQEConfig()

        if self.config.enable_adaptive_fusion:
            self.attention_net = nn.Sequential(
                nn.Linear(feat_dim, feat_dim // 4),
                nn.ReLU(),
                nn.Linear(feat_dim // 4, 1)
            )
            print("[CMQE Ablation] ✓ Adaptive fusion enabled")
        else:
            print("[CMQE Ablation] ✗ Adaptive fusion disabled (using simple concatenation)")

        self.final_proj = nn.Linear(total_queries, output_dim)

    def forward(self, layer_outputs):
        if self.config.enable_adaptive_fusion:
            layer_weights = []
            for out in layer_outputs:
                global_feat = torch.mean(out, dim=1)
                weight = torch.sigmoid(self.attention_net(global_feat))
                layer_weights.append(weight)

            weighted_outputs = []
            for out, weight in zip(layer_outputs, layer_weights):
                weighted_out = out * weight.unsqueeze(1)
                weighted_outputs.append(weighted_out)
        else:
            weighted_outputs = layer_outputs

        concat_out = torch.cat(weighted_outputs, dim=1)
        final_out = self.final_proj(concat_out.permute(0, 2, 1))
        final_out = final_out.flatten(1)
        final_out = torch.nn.functional.normalize(final_out, p=2, dim=-1)

        return final_out


class CMQEModalitySystem(nn.Module):
    """
    CMQE完整的模态特异性系统

    架构：
    1. 7个独立的BoQ模块 (基础设计)
    2. CMQE跨模态查询交换 (核心创新)
    3. 两阶段处理：查询交换 → 特征提取
    """

    def __init__(self, input_dim=512, num_queries=32, num_layers=2, row_dim=32, cmqe_config=None):
        super().__init__()

        self.config = cmqe_config if cmqe_config is not None else CMQEConfig()
        self.input_dim = input_dim

        print(f"[CMQE Ablation] Using configuration: {self.config.get_ablation_name()}")

        # 7个独立的BoQ模块 (现在是基础设计)
        self.boq_modules = nn.ModuleDict({
            'RGB': CMQEAdaptiveBoQ(input_dim, num_queries, num_layers, row_dim, True, self.config),
            'NI': CMQEAdaptiveBoQ(input_dim, num_queries, num_layers, row_dim, True, self.config),
            'TI': CMQEAdaptiveBoQ(input_dim, num_queries, num_layers, row_dim, True, self.config),
            'RGB_NI': CMQEAdaptiveBoQ(input_dim, int(num_queries * 1.2), num_layers, row_dim, True, self.config),
            'RGB_TI': CMQEAdaptiveBoQ(input_dim, int(num_queries * 1.2), num_layers, row_dim, True, self.config),
            'NI_TI': CMQEAdaptiveBoQ(input_dim, int(num_queries * 1.2), num_layers, row_dim, True, self.config),
            'RGB_NI_TI': CMQEAdaptiveBoQ(input_dim, int(num_queries * 1.5), num_layers, row_dim, True, self.config)
        })

        # 🚀 核心创新：跨模态查询交换模块 (CMQE)
        if self.config.enable_cross_modal_exchange:
            self.cmqe_module = CrossModalQueryExchange(
                query_dim=input_dim,
                num_heads=8,
                exchange_ratio=self.config.exchange_ratio
            )
            print("[CMQE Ablation] ✓ CMQE (Cross-Modal Query Exchange) enabled")
        else:
            self.cmqe_module = None
            print("[CMQE Ablation] ✗ CMQE (Cross-Modal Query Exchange) disabled")

    def forward(self, features_dict):
        """
        Args:
            features_dict: {modality: features} where features is [N, B, D]
        Returns:
            results: {modality: output}
            attentions: {modality: attention_weights}
        """
        # Phase 1: 获取初始查询状态
        initial_queries = {}

        for modality, feat in features_dict.items():
            feat = feat.permute(1, 0, 2)  # [N, B, D] -> [B, N, D]
            boq_module = self.boq_modules[modality]

            # 第一次前向传播，获取初始查询
            _, _, first_layer_queries = boq_module(feat)
            if first_layer_queries is not None:
                initial_queries[modality] = first_layer_queries
            else:
                # 如果没有查询输出，使用默认形状
                B = feat.size(0)
                num_q = boq_module.layer_queries[0]
                initial_queries[modality] = torch.randn(B, num_q, self.input_dim, device=feat.device)

        # Phase 2: 跨模态查询交换 (CMQE核心创新)
        if self.config.enable_cross_modal_exchange and self.cmqe_module is not None:
            enhanced_queries = self.cmqe_module(initial_queries)
            #print(f"[CMQE] Enhanced queries for {len(enhanced_queries)} modalities")
        else:
            enhanced_queries = initial_queries

        # Phase 3: 使用增强查询进行最终特征提取
        results = {}
        attentions = {}

        for modality, feat in features_dict.items():
            feat = feat.permute(1, 0, 2)  # [N, B, D] -> [B, N, D]
            boq_module = self.boq_modules[modality]

            # 使用增强查询进行最终推理
            enhanced_q = enhanced_queries.get(modality, None)
            final_out, attn_list, _ = boq_module(feat, enhanced_q)

            results[modality] = final_out
            attentions[modality] = attn_list

        return results, attentions

    def get_ablation_summary(self):
        """获取当前消融配置的总结"""
        enabled = []
        disabled = []

        components = [
            ("HQ", "Hierarchical Queries", self.config.enable_hierarchical_queries),
            ("QP", "Query Propagation", self.config.enable_query_propagation),
            ("CMQE", "Cross-Modal Query Exchange", self.config.enable_cross_modal_exchange),
            ("AF", "Adaptive Fusion", self.config.enable_adaptive_fusion)
        ]

        for abbr, full_name, enabled_flag in components:
            if enabled_flag:
                enabled.append(f"{abbr} ({full_name})")
            else:
                disabled.append(f"{abbr} ({full_name})")

        summary = f"""
=== CMQE Ablation Configuration ===
Configuration Name: {self.config.get_ablation_name()}
Total BoQ Modules: 7 (RGB, NI, TI, RGB_NI, RGB_TI, NI_TI, RGB_NI_TI)
Enabled Components: {', '.join(enabled) if enabled else 'None'}
Disabled Components: {', '.join(disabled) if disabled else 'None'}
CMQE Exchange Ratio: {self.config.exchange_ratio}
===================================
        """
        return summary.strip()


# 使用示例和测试代码
def create_cmqe_ablation_configs():
    """创建不同的CMQE消融实验配置"""

    configs = {
        # 基线配置：所有组件禁用
        'cmqe_baseline': CMQEConfig(
            enable_hierarchical_queries=False,
            enable_query_propagation=False,
            enable_cross_modal_exchange=False,
            enable_adaptive_fusion=False
        ),

        # 单个组件测试
        'cmqe_only_hq': CMQEConfig(
            enable_hierarchical_queries=True,
            enable_query_propagation=False,
            enable_cross_modal_exchange=False,
            enable_adaptive_fusion=False
        ),

        'cmqe_only_exchange': CMQEConfig(
            enable_hierarchical_queries=False,
            enable_query_propagation=False,
            enable_cross_modal_exchange=True,
            enable_adaptive_fusion=False
        ),

        # 完整配置：所有组件启用
        'cmqe_full': CMQEConfig(
            enable_hierarchical_queries=True,
            enable_query_propagation=True,
            enable_cross_modal_exchange=True,
            enable_adaptive_fusion=True
        ),

        # CMQE强度测试
        'cmqe_weak_exchange': CMQEConfig(
            enable_cross_modal_exchange=True,
            exchange_ratio=0.1
        ),

        'cmqe_strong_exchange': CMQEConfig(
            enable_cross_modal_exchange=True,
            exchange_ratio=0.5
        )
    }

    return configs


def test_cmqe_model():
    """测试CMQE模型的功能"""

    # 创建测试配置
    config = CMQEConfig(enable_cross_modal_exchange=True)

    # 创建模型
    model = CMQEModalitySystem(
        input_dim=512,
        num_queries=32,
        num_layers=2,
        row_dim=32,
        cmqe_config=config
    )

    # 创建测试数据
    batch_size = 4
    seq_len = 100
    feat_dim = 512

    test_features = {
        'RGB': torch.randn(seq_len, batch_size, feat_dim),
        'NI': torch.randn(seq_len, batch_size, feat_dim),
        'TI': torch.randn(seq_len, batch_size, feat_dim),
        'RGB_NI': torch.randn(seq_len * 2, batch_size, feat_dim),
        'RGB_TI': torch.randn(seq_len * 2, batch_size, feat_dim),
        'NI_TI': torch.randn(seq_len * 2, batch_size, feat_dim),
        'RGB_NI_TI': torch.randn(seq_len * 3, batch_size, feat_dim)
    }

    # 前向传播
    with torch.no_grad():
        results, attentions = model(test_features)

    # 打印结果
    print("\n=== CMQE Test Results ===")
    for modality, result in results.items():
        print(f"{modality}: {result.shape}")

    print(f"\n{model.get_ablation_summary()}")

    return model, results, attentions


if __name__ == "__main__":
    # 运行测试
    model, results, attentions = test_cmqe_model()

    # 展示不同配置
    print("\n=== Available CMQE Ablation Configurations ===")
    configs = create_cmqe_ablation_configs()
    for name, config in configs.items():
        print(f"{name}: {config.get_ablation_name()}")