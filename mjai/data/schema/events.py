from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Event:
    """Normalized replay event emitted by the Tenhou parser."""

    game_id: str
    hand_index: int
    step_index: int
    event_type: str
    actor: int | None = None
    raw_tag: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
