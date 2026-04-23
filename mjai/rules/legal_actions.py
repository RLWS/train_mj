from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from mjai.rules.state import HandState
from mjai.rules.tiles import AKA_BASE_INDICES, base_index_to_tile, tile_to_base_index


@dataclass(frozen=True, slots=True)
class Action:
    """Minimal legal action representation for early rule tests."""

    kind: str
    tile: str | None = None
    consumed_tiles: tuple[str, ...] = ()
    source: int | None = None


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


def legal_responses(hand: HandState, actor: int, source: int, tile: str) -> list[Action]:
    """Return legal response actions after another player discards one tile."""
    actions: list[Action] = [Action(kind="pass")]
    target_base = tile_to_base_index(tile)

    if source == (actor + 3) % 4:
        actions.extend(_legal_chi_actions(hand, source, tile, target_base))

    actions.extend(_legal_pon_actions(hand, source, tile, target_base))
    actions.extend(_legal_daiminkan_actions(hand, source, tile, target_base))
    return _deduplicate_actions(actions)


def _legal_chi_actions(hand: HandState, source: int, tile: str, target_base: int) -> list[Action]:
    if target_base >= 27:
        return []

    rank = (target_base % 9) + 1
    suit_offset = (target_base // 9) * 9
    actions: list[Action] = []

    sequence_patterns = []
    if rank >= 3:
        sequence_patterns.append((target_base - 2, target_base - 1))
    if 2 <= rank <= 8:
        sequence_patterns.append((target_base - 1, target_base + 1))
    if rank <= 7:
        sequence_patterns.append((target_base + 1, target_base + 2))

    for left_base, right_base in sequence_patterns:
        if left_base < suit_offset or right_base >= suit_offset + 9:
            continue

        left_options = hand.exact_tiles_for_base(left_base)
        right_options = hand.exact_tiles_for_base(right_base)
        if not left_options or not right_options:
            continue

        for left_tile, right_tile in product(left_options, right_options):
            consumed = canonicalize_consumed((left_tile, right_tile))
            actions.append(
                Action(
                    kind="chi",
                    tile=tile,
                    consumed_tiles=consumed,
                    source=source,
                )
            )

    return _deduplicate_actions(actions)


def _legal_pon_actions(hand: HandState, source: int, tile: str, target_base: int) -> list[Action]:
    exact_tiles = hand.exact_tiles_for_base(target_base)
    if len(exact_tiles) < 2:
        return []

    actions = [
        Action(
            kind="pon",
            tile=tile,
            consumed_tiles=canonicalize_consumed(consumed),
            source=source,
        )
        for consumed in combinations(exact_tiles, 2)
    ]
    return _deduplicate_actions(actions)


def _legal_daiminkan_actions(hand: HandState, source: int, tile: str, target_base: int) -> list[Action]:
    exact_tiles = hand.exact_tiles_for_base(target_base)
    if len(exact_tiles) < 3:
        return []

    actions = [
        Action(
            kind="daiminkan",
            tile=tile,
            consumed_tiles=canonicalize_consumed(consumed),
            source=source,
        )
        for consumed in combinations(exact_tiles, 3)
    ]
    return _deduplicate_actions(actions)


def canonicalize_consumed(consumed: tuple[str, ...] | tuple[str, str] | tuple[str, str, str]) -> tuple[str, ...]:
    return tuple(sorted(consumed, key=_tile_sort_key))


def _tile_sort_key(tile: str) -> tuple[int, int, str]:
    base = tile_to_base_index(tile)
    exact_priority = 0 if tile[0] == "0" else 1
    return (base, exact_priority, tile)


def _deduplicate_actions(actions: list[Action]) -> list[Action]:
    unique: dict[tuple[str, str | None, tuple[str, ...], int | None], Action] = {}
    for action in actions:
        key = (action.kind, action.tile, action.consumed_tiles, action.source)
        unique[key] = action
    return list(unique.values())
