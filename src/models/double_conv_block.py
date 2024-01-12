import torch
import torch.nn as nn

from .conv_block import ConvBlock


class DoubleConvBlock(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
