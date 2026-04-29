from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from mjai.rules.hand_checker import is_agari, is_tenpai_after_discard, winning_tile_base_indices
from mjai.rules.state import GameState, HandState
from mjai.rules.tiles import AKA_BASE_INDICES, base_index_to_tile, tile_to_base_index

YAOCHUU_BASE_INDICES = frozenset({0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33})


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


def legal_turn_actions(hand: HandState, state: GameState | None = None, actor: int | None = None) -> list[Action]:
    """Return current-turn actions.

    This pass includes discard, optional riichi-discard, tsumo, ankan, kakan, and kyuushu_kyuuhai.
    """
    actions = legal_discards(hand)
    if state is not None and actor is not None and state.can_declare_riichi(actor):
        riichi_actions = [
            Action(kind="riichi_discard", tile=action.tile)
            for action in actions
            if action.tile is not None and is_tenpai_after_discard(hand, action.tile)
        ]
        actions.extend(riichi_actions)
    if is_agari(hand):
        actions.append(Action(kind="tsumo"))
    if state is not None and actor is not None:
        actions.extend(_legal_ankan_actions(hand, actor))
        actions.extend(_legal_kakan_actions(hand, state, actor))
        if _can_declare_kyuushu_kyuuhai(hand, state, actor):
            actions.append(Action(kind="kyuushu_kyuuhai"))
    return actions


def legal_responses(
    hand: HandState,
    actor: int,
    source: int,
    tile: str,
    state: GameState | None = None,
    allow_chankan: bool = False,
) -> list[Action]:
    """Return legal response actions after another player discards one tile."""
    actions: list[Action] = [Action(kind="pass")]
    target_base = tile_to_base_index(tile)

    if not allow_chankan:
        if source == (actor + 3) % 4:
            actions.extend(_legal_chi_actions(hand, source, tile, target_base))

        actions.extend(_legal_pon_actions(hand, source, tile, target_base))
        actions.extend(_legal_daiminkan_actions(hand, source, tile, target_base))
    can_agari = is_agari(hand, incoming_tile=tile) and not _is_ron_blocked_by_furiten(
        hand=hand,
        actor=actor,
        tile=tile,
        state=state,
    )
    if can_agari:
        actions.append(Action(kind="ron", tile=tile, source=source))
        if allow_chankan:
            actions.append(Action(kind="chankan", tile=tile, source=source))
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


def _legal_ankan_actions(hand: HandState, actor: int) -> list[Action]:
    actions: list[Action] = []
    for base_index in range(34):
        exact_tiles = hand.exact_tiles_for_base(base_index)
        if len(exact_tiles) != 4:
            continue
        actions.append(
            Action(
                kind="ankan",
                tile=exact_tiles[0],
                consumed_tiles=canonicalize_consumed(tuple(exact_tiles)),
                source=actor,
            )
        )
    return actions


def _legal_kakan_actions(hand: HandState, state: GameState, actor: int) -> list[Action]:
    actions: list[Action] = []
    for meld in state.melds[actor]:
        if meld.kind != "pon":
            continue
        exact_tiles = hand.exact_tiles_for_base(meld.base_index)
        for added_tile in exact_tiles:
            actions.append(
                Action(
                    kind="kakan",
                    tile=meld.called_tile,
                    consumed_tiles=canonicalize_consumed(meld.consumed_tiles + (added_tile,)),
                    source=meld.source,
                )
            )
    return _deduplicate_actions(actions)


def _can_declare_kyuushu_kyuuhai(hand: HandState, state: GameState, actor: int) -> bool:
    if hand.total_tiles() != 14:
        return False

    if state.discards[actor]:
        return False

    if any(player_melds for player_melds in state.melds):
        return False

    distinct_yaochuu = sum(1 for base_index in YAOCHUU_BASE_INDICES if hand.counts[base_index] > 0)
    return distinct_yaochuu >= 9


def _is_ron_blocked_by_furiten(hand: HandState, actor: int, tile: str, state: GameState | None) -> bool:
    if state is None:
        return False

    winning_bases = winning_tile_base_indices(hand)
    if not winning_bases:
        return False

    return (
        state.has_discard_furiten(actor, winning_bases)
        or state.temporary_furiten[actor]
        or state.riichi_furiten[actor]
    )


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
