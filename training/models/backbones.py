import sys
import os

from .resnet3d_csn_local import ResNet3dCSN

def build_csn_backbone(config):
        
    # Constructs model: num_classes=2, depth 50 (as per train_header.py)
    model = ResNet3dCSN(
        pretrained2d=False,
        pretrained=None,
        depth=50,
        with_pool2=False,
        bottleneck_mode='ir',
        norm_eval=False,
        zero_init_residual=False,
        bn_frozen=False,
        num_classes=2
    )
    return model
