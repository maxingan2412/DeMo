import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from modeling.make_model_clipreid import load_clip_to_cpu
from modeling.clip.LoRA import mark_only_lora_as_trainable as lora_train


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory, feat_dim):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.in_planes = feat_dim
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        self.direct = cfg.MODEL.DIRECT
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            self.camera_num = camera_num
        else:
            self.camera_num = 0
        # No view
        self.view_num = 0
        if cfg.MODEL.TRANSFORMER_TYPE == 'vit_base_patch16_224':
            self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                            num_classes=num_classes,
                                                            camera=self.camera_num, view=self.view_num,
                                                            stride_size=cfg.MODEL.STRIDE_SIZE,
                                                            drop_path_rate=cfg.MODEL.DROP_PATH,
                                                            drop_rate=cfg.MODEL.DROP_OUT,
                                                            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
            self.clip = 0
            self.base.load_param(model_path)
            print('Loading pretrained model from ImageNet')
            if cfg.MODEL.FROZEN:
                lora_train(self.base)
        elif cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
            self.clip = 1
            self.sie_xishu = cfg.MODEL.SIE_COE
            clip_model = load_clip_to_cpu(cfg, self.model_name, cfg.INPUT.SIZE_TRAIN[0] // cfg.MODEL.STRIDE_SIZE[0],
                                          cfg.INPUT.SIZE_TRAIN[1] // cfg.MODEL.STRIDE_SIZE[1],
                                          cfg.MODEL.STRIDE_SIZE)
            print('Loading pretrained model from CLIP')
            clip_model.to("cuda")
            self.base = clip_model.visual
            if cfg.MODEL.FROZEN:
                lora_train(self.base)

            if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_CAMERA:
                self.cv_embed = nn.Parameter(torch.zeros(camera_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(camera_num))
            elif cfg.MODEL.SIE_VIEW:
                self.cv_embed = nn.Parameter(torch.zeros(view_num, 1, 768))
                trunc_normal_(self.cv_embed, std=.02)
                print('camera number is : {}'.format(view_num))

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

    def forward(self, x, label=None, cam_label=None, view_label=None, modality=None):
        if self.clip == 0:
            x = self.base(x, cam_label=cam_label, view_label=view_label) #这里base叫 visiontransformer，应该就是clip的 vision部分，在上面有 self.base = clip_model.visual
        else:
            if self.cv_embed_sign:
                cv_embed = self.sie_xishu * self.cv_embed[cam_label]
            else:
                cv_embed = None
            x = self.base(x, cv_embed, modality)

        global_feat = x[:, 0]
        x = x[:, 1:]
        return x, global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


import torch
import torch.nn as nn
from mmcls.models import build_classifier
from mmcv import Config

class BuildRWKV6(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory=None, feat_dim=768):
        super(BuildRWKV6, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.in_planes = feat_dim
        self.cv_embed_sign = cfg.MODEL.SIE_CAMERA
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        self.direct = cfg.MODEL.DIRECT
        self.sie_xishu = cfg.MODEL.SIE_COE
        print(f'Using {self.model_name} as RWKV6 backbone')

        self.camera_num = camera_num if self.cv_embed_sign else 0
        self.view_num = 0
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.clip = 0  # 标志是否为 clip 模型结构

        cfg_dict = dict(
            type='ImageClassifier',
            backbone=dict(
                type='VRWKV6',
                img_size=cfg.INPUT.SIZE_TRAIN[0],
                patch_size=16,
                embed_dims=feat_dim,
                num_heads=cfg.MODEL.NUM_HEADS,
                drop_path_rate=cfg.MODEL.DROP_PATH,
                init_values=cfg.MODEL.INIT_VAL,
                shift_pixel=1,
                shift_mode='q_shift_multihead',
                with_cls_token=True,
                post_norm=True,
                init_mode='fancy',
            ),
            neck=dict(type='GlobalAveragePooling'),
            head=dict(
                type='LinearClsHead',
                num_classes=num_classes,
                in_channels=feat_dim,
                init_cfg=None,
                loss=dict(
                    type='LabelSmoothLoss',
                    label_smooth_val=0.1,
                    mode='original'),
                cal_acc=False),
            init_cfg=[
                dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
                dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ],
            train_cfg=dict(augments=[
                dict(type='BatchMixup', alpha=0.8, num_classes=num_classes, prob=0.5),
                dict(type='BatchCutMix', alpha=1.0, num_classes=num_classes, prob=0.5)
            ])
        )

        cfg_model = Config(dict(model=cfg_dict))
        self.base = build_classifier(cfg_model.model)

        # 加载预训练权重
        if model_path is not None:
            self.load_param(model_path)

        # 相机/视角嵌入（可选）
        if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, 1, feat_dim))
            nn.init.trunc_normal_(self.cv_embed, std=0.02)
        elif cfg.MODEL.SIE_CAMERA:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, 1, feat_dim))
            nn.init.trunc_normal_(self.cv_embed, std=0.02)
        elif cfg.MODEL.SIE_VIEW:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, 1, feat_dim))
            nn.init.trunc_normal_(self.cv_embed, std=0.02)

    def forward(self, x, label=None, cam_label=None, view_label=None, modality=None):
        # cv_embed 可选注入
        if self.cv_embed_sign:
            cv_embed = self.sie_xishu * self.cv_embed[cam_label]
        else:
            cv_embed = None

        out = self.base.backbone(x)  # 假设只用 backbone 输出特征
        if isinstance(out, (list, tuple)):
            out = out[0]  # 取 patch token 特征
        B, C, H, W = out.shape
        global_feat = out.mean(dim=[2, 3])  # GAP

        return out, global_feat

    def load_param(self, trained_path):
        print(f'Loading pretrained model from {trained_path}')
        param_dict = torch.load(trained_path, map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k in param_dict:
            if k in self.state_dict() and self.state_dict()[k].shape == param_dict[k].shape:
                self.state_dict()[k].copy_(param_dict[k])

    def load_param_finetune(self, model_path):
        print(f'Loading finetune model from {model_path}')
        param_dict = torch.load(model_path, map_location='cpu')
        for k in param_dict:
            if k in self.state_dict() and self.state_dict()[k].shape == param_dict[k].shape:
                self.state_dict()[k].copy_(param_dict[k])

