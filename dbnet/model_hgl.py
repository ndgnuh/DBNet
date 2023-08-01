from functools import partial
from typing import List

from torch import nn

from .model_fpn import register, FeaturePyramidNetwork


class ResidualModule(nn.Module):
    def __init__(self, channels: int, k: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // k, 1),
            nn.BatchNorm2d(channels // k),
            nn.ReLU(),
            nn.Conv2d(channels // k, channels // k, 3, padding=1),
            nn.BatchNorm2d(channels // k),
            nn.ReLU(),
            nn.Conv2d(channels // k, channels, 1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.conv(x)
        x = self.relu(x)
        return x


class HourGlass(nn.Module):
    def __init__(self, channels, k: int = 4, levels: int = 4):
        super().__init__()
        self.levels = levels
        self.channels = channels
        Block = partial(ResidualModule, k=k)
        self.in_convs = nn.ModuleList([Block(channels) for _ in range(levels)])
        self.out_convs = nn.ModuleList([Block(channels) for _ in range(levels)])
        self.skip_convs = nn.ModuleList([Block(channels) for i in range(levels)])
        self.mid_conv = nn.Sequential(Block(channels), Block(channels), Block(channels))
        self.down = nn.MaxPool2d((2, 2))
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x):
        levels = self.levels
        skips = []
        # In branch
        for i in range(levels):
            skips.append(x)
            x = self.in_convs[i](x)
            x = self.down(x)

        # Skip connections
        skips = [layer(skip) for layer, skip in zip(self.skip_convs, skips)]

        # Out branch
        skips = list(reversed(skips))
        x = self.up(self.mid_conv(x))
        for i in range(levels):
            x = x + skips[i]
            x = self.out_convs[i](x)
            if i != levels - 1:
                x = self.up(x)
        return x

    def __repr__(self):
        return f"HourGlass({self.channels}, levels={self.levels})"


class StackedHourGlass(nn.Sequential):
    def __init__(
        self,
        list_channels: List[int],
        out_channels: List[int],
        expand: int = 4,
        levels: int = 4,
    ):
        super().__init__()
        ThisHourGlass = partial(HourGlass, k=expand, levels=levels)
        DownConv = partial(nn.Conv2d, kernel_size=(3, 3), stride=2, padding=1)
        self.stem = nn.Sequential(
            DownConv(3, list_channels[0]),
            nn.BatchNorm2d(list_channels[0]),
            nn.ReLU(),
            DownConv(list_channels[0], list_channels[0]),
            nn.BatchNorm2d(list_channels[0]),
            nn.ReLU(),
        )
        for i, in_channels in enumerate(list_channels):
            try:
                _out_channels = list_channels[i + 1]
            except IndexError:
                _out_channels = out_channels
            hgl = nn.Sequential()
            hgl.hourglass = ThisHourGlass(in_channels)
            hgl.downscale = nn.Sequential(
                DownConv(in_channels, _out_channels),
                nn.BatchNorm2d(_out_channels),
                nn.ReLU(),
            )
            setattr(self, f"hourglass_{i}", hgl)


@register("stacked_hourglass")
def fpn_hourglass(out_channels: int):
    net = StackedHourGlass([32, 64, 128, 192, 256], 256, levels=4, expand=4)
    fpn = FeaturePyramidNetwork(net, [0, 1, 2, 3, 4], out_channels)
    return fpn
