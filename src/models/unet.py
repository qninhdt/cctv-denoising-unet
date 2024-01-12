from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .double_conv_block import DoubleConvBlock
from .up_block import UpBlock


class UNet(nn.Module):
    DEPTH = 5

    def __init__(self):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        down_blocks = []
        up_blocks = []

        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]

        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential) and len(down_blocks) < 3:
                down_blocks.append(bottleneck)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = DoubleConvBlock(1024, 1024)

        up_blocks.append(UpBlock(1024, 512))
        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(
            UpBlock(
                in_channels=128 + 64,
                out_channels=128,
                up_conv_in_channels=256,
                up_conv_out_channels=128,
            )
        )
        up_blocks.append(
            UpBlock(
                in_channels=64 + 3,
                out_channels=64,
                up_conv_in_channels=128,
                up_conv_out_channels=64,
            )
        )

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, 3, kernel_size=1, stride=1)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (self.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{self.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])

        output_feature_map = x

        x = self.out(x)

        del pre_pools

        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x
