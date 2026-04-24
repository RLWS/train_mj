from __future__ import annotations

import torch
from torch.nn import functional as F


def policy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none")
    if sample_weights is None:
        return losses.mean()
    return (losses * sample_weights).mean()
