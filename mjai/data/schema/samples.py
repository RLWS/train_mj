from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TurnSample:
    """Supervised sample for a player's turn discard decision."""

    sample_id: str
    game_id: str
    hand_index: int
    step_index: int
    actor: int
    decision_type: str
    state: dict[str, Any] = field(default_factory=dict)
    legal_actions: list[dict[str, Any]] = field(default_factory=list)
    label_action: dict[str, Any] = field(default_factory=dict)
