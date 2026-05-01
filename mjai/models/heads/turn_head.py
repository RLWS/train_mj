from __future__ import annotations

import torch
from torch import nn


class TurnHead(nn.Module):
    """Score legal turn candidates from encoded state and action features."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        joint_size = hidden_size * 4
        self.scorer = nn.Sequential(
            nn.LayerNorm(joint_size),
            nn.Linear(joint_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        state_hidden: torch.Tensor,
        candidate_hidden: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_state = state_hidden.unsqueeze(1).expand_as(candidate_hidden)
        joint = torch.cat(
            [
                expanded_state,
                candidate_hidden,
                expanded_state * candidate_hidden,
                torch.abs(expanded_state - candidate_hidden),
            ],
            dim=-1,
        )
        logits = self.scorer(joint).squeeze(-1)
        return logits.masked_fill(~candidate_mask, -1e9)
