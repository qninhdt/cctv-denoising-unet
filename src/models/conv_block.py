import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Im just a simple conv block: Conv2d -> BatchNorm2d -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 1,
        kernel_size: int = 3,
        stride: int = 1,
        with_nonlinearity: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x
