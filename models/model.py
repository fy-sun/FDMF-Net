import torch
import torch.nn as nn
import torch.nn.functional as F


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    count = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) \
                or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Parameter):
            group_decay.append(m)

    assert len(list(module.parameters())) >= len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


class Baseline(nn.Module):
    def __init__(self, cfg=None, num_classes=None, norm_layer=nn.BatchNorm2d, in_chans=None):
        super(Baseline, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        if in_chans is not None:
            self.in_chans = in_chans
        else:
            self.in_chans = [3, 1]

        if cfg.backbone == 'mit_b5':
            from .encoder_agg import mit_b5 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b4':
            from .encoder_agg import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b2':
            from .encoder_agg import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b1':
            from .encoder_agg import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
        elif cfg.backbone == 'mit_b0':
            from .encoder_agg import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)
            self.channels = [32, 64, 160, 256]
        else:
            from .encoder_agg import mit_b4 as backbone
            self.backbone = backbone(norm_fuse=norm_layer, in_chans=self.in_chans)

        from .Seg_head import DecoderHead
        self.decode_head = DecoderHead(in_channels=self.channels, num_classes=num_classes, norm_layer=norm_layer,
                                       embed_dim=cfg.decoder_embed_dim)

        self.init_weights(cfg, pretrained=cfg.pretrained_backbone)


    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                    self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                    mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        ori_size = rgb.shape
        x_semantic, MIloss, low_MIloss = self.backbone(rgb, modal_x)

        out_semantic = self.decode_head.forward(x_semantic)
        out_semantic = F.interpolate(out_semantic, size=ori_size[2:], mode='bilinear', align_corners=False)

        return out_semantic, MIloss, low_MIloss

    def forward(self, rgb, modal_x):
        if modal_x.ndim == 3:
            modal_x = torch.unsqueeze(modal_x, dim=1)
        outputs, MIloss, low_MIloss = self.encode_decode(rgb, modal_x)

        return outputs, MIloss, low_MIloss
