import torch.nn as nn
from modeling.backbones.vit_pytorch import vit_base_patch16_224, vit_small_patch16_224, \
    deit_small_patch16_224
from modeling.backbones.t2t import t2t_vit_t_14, t2t_vit_t_24
from fvcore.nn import flop_count
from modeling.backbones.basic_cnn_params.flops import give_supported_ops
import copy
from modeling.meta_arch import build_transformer, weights_init_classifier, weights_init_kaiming,VRWKV6, build_transformer_new
from modeling.moe.AttnMOE import GeneralFusion, QuickGELU
import torch
import os

class DeMo(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(DeMo, self).__init__()
        self.rwkvbackbone = False
        self.cengjifusion = cfg.MODEL.CENGJIFUSION


        print('cengjifusion:', self.cengjifusion,'mxa')
        print('rwkvbackbone:', self.rwkvbackbone,'mxa')

        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 512
        elif 'VRWKV6BASE' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
            self.rwkvbackbone = True


            print('using rwkv backbone','mxa')
        self.cfg = cfg
        if cfg.MODEL.FROZEN and False:
            self.BACKBONE_R = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
            self.BACKBONE_N = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
            self.BACKBONE_T = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        else:

            if self.rwkvbackbone:
                self.BACKBONE = VRWKV6(img_size=(cfg.INPUT.SIZE_TRAIN[0],cfg.INPUT.SIZE_TRAIN[1]), cfg =cfg,num_classes = num_classes, camera_num = camera_num,embed_dims = self.feat_dim,num_heads = 12)  #写这里
                current_dir = os.getcwd()

                model_path = os.path.join(current_dir, 'vrwkv6_b_in1k_224.pth')
                self.BACKBONE.load_param(model_path)
            elif self.cengjifusion:

                self.BACKBONE = build_transformer_new(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
            else:
                self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim) #bulid_transformer actually is backbone

        self.num_classes = num_classes #
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.camera = camera_num
        self.view = view_num
        self.direct = cfg.MODEL.DIRECT
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN
        self.miss_type = cfg.TEST.MISS
        self.HDM = cfg.MODEL.HDM
        self.ATM = cfg.MODEL.ATM
        self.GLOBAL_LOCAL = cfg.MODEL.GLOBAL_LOCAL
        self.head = cfg.MODEL.HEAD
        if self.GLOBAL_LOCAL:   # 
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.rgb_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim),QuickGELU())
            self.nir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())
            self.tir_reduce = nn.Sequential(nn.LayerNorm(2 * self.feat_dim),
                                            nn.Linear(2 * self.feat_dim, self.feat_dim), QuickGELU())

        if self.HDM or self.ATM:
            self.generalFusion = GeneralFusion(feat_dim=self.feat_dim, num_experts=7, head=self.head, reg_weight=1,
                                               cfg=cfg)
            self.classifier_moe = nn.Linear(7 * self.feat_dim, self.num_classes, bias=False)
            self.classifier_moe.apply(weights_init_classifier)
            self.bottleneck_moe = nn.BatchNorm1d(7 * self.feat_dim)
            self.bottleneck_moe.bias.requires_grad_(False)
            self.bottleneck_moe.apply(weights_init_kaiming)
        if self.direct:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
        else:
            self.classifier_r = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_r.apply(weights_init_classifier)
            self.bottleneck_r = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_r.bias.requires_grad_(False)
            self.bottleneck_r.apply(weights_init_kaiming)
            self.classifier_n = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_n.apply(weights_init_classifier)
            self.bottleneck_n = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_n.bias.requires_grad_(False)
            self.bottleneck_n.apply(weights_init_kaiming)
            self.classifier_t = nn.Linear(self.feat_dim, self.num_classes, bias=False)
            self.classifier_t.apply(weights_init_classifier)
            self.bottleneck_t = nn.BatchNorm1d(self.feat_dim)
            self.bottleneck_t.bias.requires_grad_(False)

    def load_param(self, trained_path):
        state_dict = torch.load(trained_path, map_location="cpu")
        print(f"Successfully load ckpt!")
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def flops(self, shape=(3, 256, 128)):
        if self.image_size[0] != shape[1] or self.image_size[1] != shape[2]:
            shape = (3, self.image_size[0], self.image_size[1])
            # For vehicle reid, the input shape is (3, 128, 256)
        supported_ops = give_supported_ops()
        model = copy.deepcopy(self)
        model.cuda().eval()
        input_r = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_n = torch.randn((1, *shape), device=next(model.parameters()).device)
        input_t = torch.randn((1, *shape), device=next(model.parameters()).device)
        cam_label = 0
        input = {"RGB": input_r, "NI": input_n, "TI": input_t, "cam_label": cam_label}
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(
            "The out_proj here is called by the nn.MultiheadAttention, which has been calculated in th .forward(), so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("For the bottleneck or classifier, it is not calculated during inference, so just ignore it.")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        del model, input
        return sum(Gflops.values()) * 1e9

    def forward(self, x, label=None, cam_label=None, view_label=None, return_pattern=3, img_path=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            if self.cfg.MODEL.FROZEN and False:
                RGB_cash, RGB_global = self.BACKBONE_R(RGB, cam_label=cam_label, view_label=view_label) #RGB_cash 是除了cls token之外的所有token的特征
                NI_cash, NI_global = self.BACKBONE_N(NI, cam_label=cam_label, view_label=view_label)
                TI_cash, TI_global = self.BACKBONE_T(TI, cam_label=cam_label, view_label=view_label)
            elif self.cengjifusion:
                RGB_cash, RGB_global, NI_cash, NI_global, TI_cash, TI_global = self.BACKBONE(x, cam_label=cam_label, view_label=view_label) #RGB_cash 是除了cls token之外的所有token的特征
            else:
                RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label) #RGB_cash 是除了cls token之外的所有token的特征
                NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
                TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)
                if self.rwkvbackbone:
                    RGB_cash = RGB_cash.reshape(RGB_cash.shape[0], RGB_cash.shape[1], -1).permute(0, 2, 1)
                    NI_cash = NI_cash.reshape(NI_cash.shape[0], NI_cash.shape[1], -1).permute(0, 2, 1)
                    TI_cash = TI_cash.reshape(TI_cash.shape[0], TI_cash.shape[1], -1).permute(0, 2, 1)



            if self.GLOBAL_LOCAL:
                RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1)) #到这里 应该就是论文的前面的 PIFE部分 HDM ATM 文中也都有提到
            if self.HDM or self.ATM:
                moe_feat, loss_moe = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
                moe_score = self.classifier_moe(self.bottleneck_moe(moe_feat))
            if self.direct:
                ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
                ori_global = self.bottleneck(ori)
                ori_score = self.classifier(ori_global)
            else:
                RGB_ori_score = self.classifier_r(self.bottleneck_r(RGB_global))
                NI_ori_score = self.classifier_n(self.bottleneck_n(NI_global))
                TI_ori_score = self.classifier_t(self.bottleneck_t(TI_global))
            if self.direct:
                if self.HDM or self.ATM:
                    return moe_score, moe_feat, ori_score, ori, loss_moe
                return ori_score, ori
            else:
                if self.HDM or self.ATM:
                    return moe_score, moe_feat, RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global, loss_moe
                return RGB_ori_score, RGB_global, NI_ori_score, NI_global, TI_ori_score, TI_global

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            if self.miss_type == 'r':
                RGB = torch.zeros_like(RGB)
            elif self.miss_type == 'n':
                NI = torch.zeros_like(NI)
            elif self.miss_type == 't':
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'rn':
                RGB = torch.zeros_like(RGB)
                NI = torch.zeros_like(NI)
            elif self.miss_type == 'rt':
                RGB = torch.zeros_like(RGB)
                TI = torch.zeros_like(TI)
            elif self.miss_type == 'nt':
                NI = torch.zeros_like(NI)
                TI = torch.zeros_like(TI)

            if 'cam_label' in x:
                cam_label = x['cam_label']
            if self.cfg.MODEL.FROZEN and False:
                RGB_cash, RGB_global = self.BACKBONE_R(RGB, cam_label=cam_label, view_label=view_label)
                NI_cash, NI_global = self.BACKBONE_N(NI, cam_label=cam_label, view_label=view_label)
                TI_cash, TI_global = self.BACKBONE_T(TI, cam_label=cam_label, view_label=view_label)
            elif self.cengjifusion:
                RGB_cash, RGB_global, NI_cash, NI_global, TI_cash, TI_global = self.BACKBONE(x, cam_label=cam_label, view_label=view_label)
            else:
                RGB_cash, RGB_global = self.BACKBONE(RGB, cam_label=cam_label, view_label=view_label)
                NI_cash, NI_global = self.BACKBONE(NI, cam_label=cam_label, view_label=view_label)
                TI_cash, TI_global = self.BACKBONE(TI, cam_label=cam_label, view_label=view_label)
                if self.rwkvbackbone:
                    RGB_cash = RGB_cash.reshape(RGB_cash.shape[0], RGB_cash.shape[1], -1).permute(0, 2, 1)
                    NI_cash = NI_cash.reshape(NI_cash.shape[0], NI_cash.shape[1], -1).permute(0, 2, 1)
                    TI_cash = TI_cash.reshape(TI_cash.shape[0], TI_cash.shape[1], -1).permute(0, 2, 1)

            if self.GLOBAL_LOCAL:
                RGB_local = self.pool(RGB_cash.permute(0, 2, 1)).squeeze(-1)
                NI_local = self.pool(NI_cash.permute(0, 2, 1)).squeeze(-1)
                TI_local = self.pool(TI_cash.permute(0, 2, 1)).squeeze(-1)
                RGB_global = self.rgb_reduce(torch.cat([RGB_global, RGB_local], dim=-1))
                NI_global = self.nir_reduce(torch.cat([NI_global, NI_local], dim=-1))
                TI_global = self.tir_reduce(torch.cat([TI_global, TI_local], dim=-1))
            ori = torch.cat([RGB_global, NI_global, TI_global], dim=-1)
            if self.HDM or self.ATM:
                moe_feat = self.generalFusion(RGB_cash, NI_cash, TI_cash, RGB_global, NI_global, TI_global)
                if return_pattern == 1:
                    return ori
                elif return_pattern == 2:
                    return moe_feat
                elif return_pattern == 3:
                    return torch.cat([ori, moe_feat], dim=-1)
            return ori


__factory_T_type = {
    'vit_base_patch16_224': vit_base_patch16_224,
    'deit_base_patch16_224': vit_base_patch16_224,
    'vit_small_patch16_224': vit_small_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    't2t_vit_t_14': t2t_vit_t_14,
    't2t_vit_t_24': t2t_vit_t_24,
}


def make_model(cfg, num_class, camera_num, view_num=0):
    model = DeMo(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building DeMo===========')
    return model
