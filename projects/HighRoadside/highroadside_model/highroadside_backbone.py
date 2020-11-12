# Copyright taofuyu
# fm means feature map
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.backbone.fpn import FPN

class HighRoadsideBackbone(Backbone):
    """
    Create the backbone for high roadside model.

    Returns:
    A high roadside model backbone instance
    """
    def __init__(self):
        super().__init__()
        #common params
        cm_f = 3 #common filter size
        cm_s = 1 #common stride
        cm_p = 1 #common padding
        cm_g = 1 #common group

        self.conv0_1 = self.create_conv(3, 8, cm_f, cm_s, cm_p, cm_g)
        self.conv0_2 = self.create_conv(8, 16, cm_f, cm_s, cm_p, cm_g)
        
        self.conv1_1 = self.create_conv(16, 32, cm_f, 2, cm_p, cm_g) # stride=2
        self.conv1_2 = self.create_conv(32, 32, cm_f, cm_s, cm_p, cm_g)
        self.conv1_3 = self.create_conv(32, 32, cm_f, cm_s, cm_p, cm_g)
        self.conv1_4 = self.create_conv(32, 32, cm_f, cm_s, cm_p, cm_g)

        self.conv2_1 = self.create_conv(32, 48, cm_f, 2, cm_p, cm_g) # stride=4
        self.conv2_2 = self.create_conv(48, 48, cm_f, cm_s, cm_p, cm_g)
        self.conv2_3 = self.create_conv(48, 48, cm_f, cm_s, cm_p, cm_g)
        self.conv2_4 = self.create_conv(48, 48, cm_f, cm_s, cm_p, cm_g)
        self.conv_stage1_out = self.create_conv(48, 48, cm_f, cm_s, cm_p, cm_g) # stage 1

        self.conv3_1 = self.create_conv(48, 64, cm_f, 2, cm_p, cm_g) # stride=8
        self.conv3_2 = self.create_conv(64, 64, cm_f, cm_s, cm_p, cm_g)
        self.conv3_3 = self.create_conv(64, 64, cm_f, cm_s, cm_p, cm_g)
        self.conv_stage2_out = self.create_conv(64, 64, cm_f, cm_s, cm_p, cm_g) # stage 2

        self.conv4_1 = self.create_conv(64, 80, cm_f, 2, cm_p, cm_g) # stride=16
        self.conv4_2 = self.create_conv(80, 80, cm_f, cm_s, cm_p, cm_g)
        self.conv4_3 = self.create_conv(80, 80, cm_f, cm_s, cm_p, cm_g)
        self.conv_stage3_out = self.create_conv(80, 80, cm_f, cm_s, cm_p, cm_g) # stage 3

        self.conv5_1 = self.create_conv(80, 96, cm_f, 2, cm_p, cm_g) # stride=32
        self.conv5_2 = self.create_conv(96, 96, cm_f, cm_s, cm_p, cm_g)
        self.conv5_3 = self.create_conv(96, 96, cm_f, cm_s, cm_p, cm_g)
        self.conv_stage4_out = self.create_conv(96, 96, cm_f, cm_s, cm_p, cm_g) # stage 4

        self.conv6_1 = self.create_conv(96, 112, cm_f, 2, cm_p, cm_g) # stride=64
        self.conv6_2 = self.create_conv(112, 112, cm_f, cm_s, cm_p, cm_g)
        self.conv_stage5_out = self.create_conv(112, 112, cm_f, cm_s, cm_p, cm_g) # stage 5

    def create_conv(self, in_cn, out_cn, f_sz, stride, padding, group):
        '''
        create a conv layer with bn&relu
        '''
        conv = nn.Sequential(
            nn.Conv2d(in_cn, out_cn, kernel_size=f_sz, stride=stride, padding=padding, groups=group, bias=False),
            nn.BatchNorm2d(out_cn),
            nn.ReLU()
        )
        #init with msra
        nn.init.kaiming_normal_(conv[0].weight, nonlinearity='relu')

        return conv

    def forward(self, x):
        out = self.conv0_1(x)
        out = self.conv0_2(out)

        out = self.conv1_1(out)
        out = self.conv1_2(out)
        out = self.conv1_3(out)
        out = self.conv1_4(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.conv2_3(out)
        out = self.conv2_4(out)
        stage1_fm = self.conv_stage1_out(out)

        out = self.conv3_1(stage1_fm)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        stage2_fm = self.conv_stage2_out(out)

        out = self.conv4_1(stage2_fm)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        stage3_fm = self.conv_stage3_out(out)

        out = self.conv5_1(stage3_fm)
        out = self.conv5_2(out)
        out = self.conv5_3(out)
        stage4_fm = self.conv_stage4_out(out)

        out = self.conv6_1(stage4_fm)
        out = self.conv6_2(out)
        stage5_fm = self.conv_stage5_out(out)

        return {'conv_stage1_out':stage1_fm, 'conv_stage2_out':stage2_fm, 'conv_stage3_out':stage3_fm, 'conv_stage4_out':stage4_fm, 'conv_stage5_out':stage5_fm}

    def output_shape(self):
        stage1_shape = ShapeSpec(channels=48, stride=4)
        stage2_shape = ShapeSpec(channels=64, stride=8)
        stage3_shape = ShapeSpec(channels=80, stride=16)
        stage4_shape = ShapeSpec(channels=96, stride=32)
        stage5_shape = ShapeSpec(channels=112, stride=64)

        return {'conv_stage1_out':stage1_shape, 'conv_stage2_out':stage2_shape, 'conv_stage3_out':stage3_shape, 'conv_stage4_out':stage4_shape, 'conv_stage5_out':stage5_shape}

@BACKBONE_REGISTRY.register()
def build_highroadside_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Implement FPN based on high roadside backbone.

    Returns:
    high_backbone + fpn
    """
    backbone = HighRoadsideBackbone()
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS

    fpn_backbone = FPN(
                        bottom_up = backbone,
                        in_features = in_features,
                        out_channels = out_channels,
                        norm = cfg.MODEL.FPN.NORM,
                        top_block = None,
                        fuse_type = cfg.MODEL.FPN.FUSE_TYPE)

    return fpn_backbone

    
    
