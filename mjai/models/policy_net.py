from __future__ import annotations

import torch
from torch import nn

from mjai.models.encoder import MahjongEncoder
from mjai.models.heads.turn_head import TurnHead


class MahjongPolicyNet(nn.Module):
    """Candidate-ranking policy for turn or response batches.

    The batch contract is documented in docs/model-inputs.md:
    state_feats=[B, state_dim], candidate_feats=[B, C, action_dim],
    candidate_mask=[B, C], labels=[B].
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = MahjongEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        self.turn_head = TurnHead(hidden_size=hidden_size, dropout=dropout)
        self.response_head = TurnHead(hidden_size=hidden_size, dropout=dropout)

    def forward_turn(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Returns logits over legal turn candidates with shape [B, C].
        state_hidden, candidate_hidden = self.encoder(
            batch["state_feats"],
            batch["candidate_feats"],
        )
        return self.turn_head(
            state_hidden=state_hidden,
            candidate_hidden=candidate_hidden,
            candidate_mask=batch["candidate_mask"],
        )

    def forward_response(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Same candidate-ranking contract as turn, but using response dimensions.
        state_hidden, candidate_hidden = self.encoder(
            batch["state_feats"],
            batch["candidate_feats"],
        )
        return self.response_head(
            state_hidden=state_hidden,
            candidate_hidden=candidate_hidden,
            candidate_mask=batch["candidate_mask"],
        )
