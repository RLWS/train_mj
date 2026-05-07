from __future__ import annotations

import torch
from torch.nn import functional as F


def policy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算候选策略分类损失。
    
    参数:
        logits (torch.Tensor): 候选动作分数张量。
        labels (torch.Tensor): 标签索引张量。
        sample_weights (torch.Tensor | None): 相关数据列表。
    
    返回:
        torch.Tensor: 张量结果。
    
    异常:
        无。
    """
    losses = F.cross_entropy(logits, labels, reduction="none")
    if sample_weights is None:
        return losses.mean()
    return (losses * sample_weights).mean()
