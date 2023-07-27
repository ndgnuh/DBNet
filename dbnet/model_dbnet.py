from typing import Optional

from torch import nn, Tensor
from torchvision import transforms as T

from .losses import dbnet_loss
from .model_fpn import backbones


def make_head(in_channels: int, num_classes: int):
    """Make a prediction head"""
    aux_size = in_channels // 4
    return nn.Sequential(
        #
        # Projection
        #
        nn.Conv2d(in_channels, aux_size, 3, padding=1, bias=False),
        nn.InstanceNorm2d(aux_size),
        nn.ReLU(),
        #
        # First upsample
        #
        nn.ConvTranspose2d(
            aux_size,
            aux_size,
            kernel_size=2,
            stride=2,
            bias=False,
        ),
        nn.InstanceNorm2d(aux_size),
        nn.ReLU(),
        #
        # Prediction
        #
        nn.ConvTranspose2d(
            aux_size,
            num_classes,
            kernel_size=2,
            stride=2,
        ),
    )


class DBNetHead(nn.Module):
    """DBNet Prediction head

    Args:
        in_channels (int):
            Number of feature channels.
        num_classes (int):
            Number of classes.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.probas = make_head(in_channels, num_classes)
        self.thresholds = make_head(in_channels, num_classes)

    def forward(
        self,
        features: Tensor,
        gt_probas: Optional[Tensor] = None,
        gt_thresholds: Optional[Tensor] = None,
    ):
        probas = self.probas(features)
        # Compute threshold if needed
        if self.training or gt_probas is not None:
            thresholds = self.thresholds(features)
        else:
            thresholds = None

        # Compute loss
        if gt_probas is not None:
            loss = dbnet_loss(probas, thresholds, gt_probas, gt_thresholds)
        else:
            loss = None

        return probas, thresholds, loss


class DBNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        hidden_size: int,
        num_classes: int,
    ):
        super().__init__()
        self.preprocess = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.backbone = backbones[backbone](hidden_size)
        self.head = DBNetHead(hidden_size, num_classes)

    def forward(
        self,
        images: Tensor,
        gt_probas: Optional[Tensor] = None,
        gt_thresholds: Optional[Tensor] = None,
    ):
        x = self.preprocess(images)
        x = self.backbone(x)
        outs = self.head(x, gt_probas, gt_thresholds)
        return outs


# Documentation
DBNet.__doc__ = f"""DBNet

Args:
    backbone (str):
        Backbone name, must be one of {backbones.keys()}
    hidden_size (int):
        The number of FPN feature channels.
    num_classes (int):
        Number of output classes.

Inputs:
    images (Tensor):
        Input image, shape [N, 3, H, W]
    gt_probas (Optional[Tensor]):
        Ground truth probability maps. Default: `None`.
    gt_thresholds (Optional[Tensor]):
        Ground truth threshold maps. Default: `None`.

Outputs:
    proba_maps (Tensor):
        Output logits of proba maps (not normalized).
    threshold_maps (Optional[Tensor]):
        Output logits of threholds maps (not normalized).
        If the model is in eval mode or the ground truth is not specified,
        this will returns None.
    loss (Optional[Tensor]):
        Loss value, None if the ground truth is not specified.
"""
