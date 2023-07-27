from typing import Optional

from torch import Tensor
from torch.nn import functional as F


def dice_loss(
    pr: Tensor,
    gt: Tensor,
    reduction: str = "mean",
    weights: Optional[Tensor] = None,
):
    # Apply weights if any
    if weights is None:
        inter = pr * gt
    else:
        inter = pr * gt * weights
    union = (pr + gt).clamp(0, 1)
    losses = 1 - (2 * inter) / (union + 1e-12)

    # Reduce and return
    if reduction == "mean":
        return losses.mean()
    elif reduction == "none":
        return losses
    elif reduction == "sum":
        return losses.sum()
    else:
        raise NotImplementedError(f"Unknown reduction {reduction}")


def compute_loss_mining(
    losses: Tensor,
    pos: Tensor,
    neg: Optional[Tensor] = None,
    k: int = 3,
):
    # Negative mask,
    # you can optionally passed it to this function to reduce computation
    if neg is None:
        neg = ~pos

    # Counting
    num_pos = torch.count_nonzero(pos)
    num_neg = torch.min(num_pos * k, torch.count_nonzero(neg))

    # Postive and negative losses
    pos_losses = torch.topk(losses[pos], k=num_pos).values
    neg_losses = torch.topk(losses[neg], k=num_neg).values
    pos_loss = pos_losses.sum()
    neg_loss = neg_losses.sum()

    # Reduction
    loss = (pos_loss + neg_loss) / (num_pos + num_neg + 1e-6)
    return loss


def dbnet_loss(
    pr_proba: Tensor,
    pr_threshold: Tensor,
    gt_proba: Tensor,
    gt_threshold: Tensor,
    proba_scale: float = 5,
    bin_scale: float = 1,
    threshold_scale: float = 10,
) -> Tensor:
    # Create differentiable binary map
    pr_bin = torch.sigmoid(50 * (pr_proba - pr_threshold))

    # Probamap loss
    p_losses = F.binary_cross_entropy_with_logits(
        pr_proba,
        gt_proba,
        reduction="none",
    )
    p_loss = compute_loss_mining(p_losses, gt_proba > 0)

    # Binary map loss
    b = p_losses.max()
    a = p_losses.min()
    weights = (p_losses - a) / (b - a) + 1
    b_loss = dice_loss(pr_bin, gt_proba, weights=weights)

    # Threshold map loss
    t_loss = F.l1_loss(torch.sigmoid(pr_threshold), gt_threshold)

    loss = proba_scale * p_loss + b_loss * bin_scale + t_loss * threshold_scale
    return loss
