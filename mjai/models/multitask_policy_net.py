from __future__ import annotations

import torch
from torch import nn

from mjai.features.action_features import RESPONSE_ACTION_DIM, TURN_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM
from mjai.models.heads.turn_head import TurnHead


class MultitaskMahjongPolicyNet(nn.Module):
    """Shared-state multitask policy for turn and response candidate ranking."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.turn_state_projection = _projection(TURN_STATE_DIM, hidden_size, dropout)
        self.response_state_projection = _projection(RESPONSE_STATE_DIM, hidden_size, dropout)
        self.shared_state_encoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.turn_action_encoder = _projection(TURN_ACTION_DIM, hidden_size, dropout)
        self.response_action_encoder = _projection(RESPONSE_ACTION_DIM, hidden_size, dropout)
        self.turn_head = TurnHead(hidden_size=hidden_size, dropout=dropout)
        self.response_head = TurnHead(hidden_size=hidden_size, dropout=dropout)

    def forward_turn(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state_hidden = self.shared_state_encoder(
            self.turn_state_projection(batch["state_feats"])
        )
        candidate_hidden = self.turn_action_encoder(batch["candidate_feats"])
        return self.turn_head(
            state_hidden=state_hidden,
            candidate_hidden=candidate_hidden,
            candidate_mask=batch["candidate_mask"],
        )

    def forward_response(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state_hidden = self.shared_state_encoder(
            self.response_state_projection(batch["state_feats"])
        )
        candidate_hidden = self.response_action_encoder(batch["candidate_feats"])
        return self.response_head(
            state_hidden=state_hidden,
            candidate_hidden=candidate_hidden,
            candidate_mask=batch["candidate_mask"],
        )


def _projection(input_dim: int, hidden_size: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_size),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
    )
