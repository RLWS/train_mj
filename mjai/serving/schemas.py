from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class RankedAction:
    rank: int
    index: int
    action: dict[str, Any]
    score: float
    prob: float


@dataclass(slots=True, frozen=True)
class ModelMetadata:
    checkpoint: str
    checkpoint_task: str
    task: str
    device: str
    state_dim: int
    action_dim: int
    candidate_limit: int
    hidden_size: int
    dropout: float
    default_pass_bias: float | None = None
    no_yaku_call_penalty: float = 0.0


@dataclass(slots=True)
class LoadedPolicy:
    task: str
    checkpoint_path: Path
    checkpoint_task: str
    device: torch.device
    state_dim: int
    action_dim: int
    hidden_size: int
    dropout: float
    candidate_limit: int
    default_pass_bias: float | None
    no_yaku_call_penalty: float
    model: nn.Module
    config: dict[str, Any]

    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            checkpoint=self.checkpoint_path.as_posix(),
            checkpoint_task=self.checkpoint_task,
            task=self.task,
            device=str(self.device),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            candidate_limit=self.candidate_limit,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            default_pass_bias=self.default_pass_bias,
            no_yaku_call_penalty=self.no_yaku_call_penalty,
        )


@dataclass(slots=True, frozen=True)
class PredictionResult:
    task: str
    selection_mode: str
    selected_action: dict[str, Any]
    predictions: list[RankedAction]
    model: ModelMetadata
    raw_selected_action: dict[str, Any] | None = None
    pass_bias: float | None = None
    no_yaku_call_penalty: float | None = None
