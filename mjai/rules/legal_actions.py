from __future__ import annotations

from dataclasses import dataclass

from mjai.rules.state import HandState
from mjai.rules.tiles import AKA_BASE_INDICES, base_index_to_tile


@dataclass(frozen=True, slots=True)
class Action:
    """Minimal legal action representation for early rule tests."""

    kind: str
    tile: str | None = None


def legal_discards(hand: HandState) -> list[Action]:
    """Return exact discard actions that are currently legal."""
    actions: list[Action] = []

    for index, count in enumerate(hand.counts):
        if count <= 0:
            continue

        red_count = hand.aka_counts.get(index, 0)
        normal_count = count - red_count

        if normal_count > 0:
            actions.append(Action(kind="discard", tile=base_index_to_tile(index)))

        if index in AKA_BASE_INDICES and red_count > 0:
            actions.append(
                Action(
                    kind="discard",
                    tile={
                        4: "0m",
                        13: "0p",
                        22: "0s",
                    }[index],
                )
            )

    return actions
