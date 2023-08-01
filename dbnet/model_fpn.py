from typing import List

import torch
from torch import nn, Tensor
from torchvision import models

backbones = {}


def register(name: str):
    """Returns a wrapper add the wrapped function `f`
    to the dicionary of backbones with name `name`"""

    def wrapper(f):
        backbones[name] = f
        return f

    return wrapper


class HiddenLayerGetter(nn.Module):
    def __init__(self, model, positions):
        super().__init__()
        self.model = model
        self.masks = len(model) * [False]
        self.num_keeps = 0
        for i in positions:
            self.masks[i] = True
            self.num_keeps += 1

    @torch.no_grad()
    def get_out_channels(self):
        inputs = torch.rand(1, 3, 2048, 2048)
        return [output.shape[1] for output in self(inputs)]

    def forward(self, inputs) -> List[Tensor]:
        x = inputs
        outputs = [torch.rand(1, 1, 1, 1, device=inputs.device)] * self.num_keeps
        count = 0
        for i, layer in enumerate(self.model):
            x = layer(x)
            if self.masks[i]:
                outputs[count] = x
                count = count + 1
        return outputs


class FeaturePyramidNeck(nn.Module):
    def __init__(self, list_in_channels, out_channels):
        super().__init__()
        assert out_channels % len(list_in_channels) == 0
        hidden_channels = out_channels // len(list_in_channels)
        self.upsample = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")
        self.in_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
                for in_channels in list_in_channels
            ]
        )
        self.out_branch = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, hidden_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(),
                    nn.Upsample(
                        scale_factor=2**idx, mode="bilinear", align_corners=True
                    ),
                )
                for idx, _ in enumerate(list_in_channels)
            ]
        )

        self.num_branches = len(list_in_channels)

    def forward(self, features: List[Tensor]):
        assert len(features) == self.num_branches
        # Input features
        features = [self.in_branch[i](ft) for i, ft in enumerate(features)]

        # Upscale combine
        outputs = [features[-1]]
        for i, ft in enumerate(reversed(features)):
            if i == 0:
                outputs.append(ft)
            else:
                output = self.upsample(outputs[-1])
                output = output + ft
                outputs.append(output)

        # Upscale concat
        features = [layer(ft) for layer, ft in zip(self.out_branch, reversed(outputs))]
        features = torch.cat(features, dim=1)
        return features


class FeaturePyramidNetwork(nn.Sequential):
    def __init__(self, net, imm_layers, out_channels: int):
        super().__init__()
        self.backbone = HiddenLayerGetter(net, imm_layers)
        self.fpn = FeaturePyramidNeck(self.backbone.get_out_channels(), out_channels)


@register("mobilenet_v3_large")
def fpn_mobilenet_v3_large(out_channels: int):
    net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights).features
    return FeaturePyramidNetwork(net, [3, 6, 9, 16], out_channels)


@register("mobilenet_v3_small")
def fpn_mobilenet_v3_small(out_channels: int):
    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights).features
    model = FeaturePyramidNetwork(net, [1, 3, 8, 12], out_channels)
    return model
