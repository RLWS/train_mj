from __future__ import annotations

import torch
from torch import nn


class MahjongEncoder(nn.Module):
    """Small shared encoder for state and candidate action features."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

    def forward(
        self,
        state_feats: torch.Tensor,
        candidate_feats: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_hidden = self.state_encoder(state_feats)
        candidate_hidden = self.action_encoder(candidate_feats)
        return state_hidden, candidate_hidden
