from __future__ import annotations

import torch
from torch import nn


class TurnHead(nn.Module):
    """Score legal turn candidates from encoded state and action features."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scorer = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        state_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        joint = torch.tanh(candidate_hidden + state_hidden.unsqueeze(1))
        logits = self.scorer(joint).squeeze(-1)
        return logits.masked_fill(~candidate_mask, -1e9)
