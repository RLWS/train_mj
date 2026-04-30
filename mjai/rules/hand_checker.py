from __future__ import annotations

from functools import lru_cache

from mjai.rules.state import HandState
from mjai.rules.tiles import base_index_to_tile, tile_to_base_index


def is_agari(hand: HandState, incoming_tile: str | None = None) -> bool:
    """Check standard four-meld-one-pair, seven pairs, and thirteen orphans."""
    counts = hand.counts.copy()
    if incoming_tile is not None:
        counts[tile_to_base_index(incoming_tile)] += 1

    return _is_agari_counts(tuple(counts))


def is_tenpai_after_discard(hand: HandState, discard_tile: str) -> bool:
    """Check whether removing one exact tile leaves the hand in tenpai."""
    if not hand.has_tile(discard_tile):
        return False

    reduced = hand.copy()
    reduced.remove_tile(discard_tile)

    return bool(winning_tile_base_indices(reduced))


def winning_tile_base_indices(hand: HandState) -> set[int]:
    """Return base tiles that would complete the current hand."""
    winning_tiles: set[int] = set()
    for base_index, count in enumerate(hand.counts):
        if count >= 4:
            continue
        if is_agari(hand, incoming_tile=base_index_to_tile(base_index)):
            winning_tiles.add(base_index)
    return winning_tiles


@lru_cache(maxsize=1_000_000)
def _is_agari_counts(counts: tuple[int, ...]) -> bool:
    total_tiles = sum(counts)
    if total_tiles % 3 != 2:
        return False

    if _is_chiitoitsu(counts):
        return True
    if _is_kokushi(counts):
        return True
    return _is_standard_agari(counts)


def _is_chiitoitsu(counts: tuple[int, ...]) -> bool:
    return sum(1 for count in counts if count == 2) == 7 and all(count in {0, 2} for count in counts)


def _is_kokushi(counts: tuple[int, ...]) -> bool:
    terminals = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    pair_found = False
    for index in terminals:
        if counts[index] == 0:
            return False
        if counts[index] >= 2:
            pair_found = True
    for index, count in enumerate(counts):
        if index not in terminals and count != 0:
            return False
    return pair_found


@lru_cache(maxsize=1_000_000)
def _is_standard_agari(counts: tuple[int, ...]) -> bool:
    for index, count in enumerate(counts):
        if count < 2:
            continue
        reduced = list(counts)
        reduced[index] -= 2
        if _can_form_all_sets_by_group(tuple(reduced)):
            return True
    return False


def _can_form_all_sets_by_group(counts: tuple[int, ...]) -> bool:
    return (
        _can_form_suit_sets(counts[0:9])
        and _can_form_suit_sets(counts[9:18])
        and _can_form_suit_sets(counts[18:27])
        and _can_form_honor_sets(counts[27:34])
    )


@lru_cache(maxsize=250_000)
def _can_form_suit_sets(counts: tuple[int, ...]) -> bool:
    try:
        first = next(index for index, count in enumerate(counts) if count > 0)
    except StopIteration:
        return True

    current = list(counts)

    if current[first] >= 3:
        current[first] -= 3
        if _can_form_suit_sets(tuple(current)):
            return True
        current[first] += 3

    if first <= 6 and current[first + 1] > 0 and current[first + 2] > 0:
        current[first] -= 1
        current[first + 1] -= 1
        current[first + 2] -= 1
        if _can_form_suit_sets(tuple(current)):
            return True

    return False


def _can_form_honor_sets(counts: tuple[int, ...]) -> bool:
    return all(count in {0, 3} for count in counts)


@lru_cache(maxsize=500_000)
def _can_form_all_sets(counts: tuple[int, ...]) -> bool:
    try:
        first = next(index for index, count in enumerate(counts) if count > 0)
    except StopIteration:
        return True

    current = list(counts)

    if current[first] >= 3:
        current[first] -= 3
        if _can_form_all_sets(tuple(current)):
            return True
        current[first] += 3

    if first < 27:
        suit_offset = (first // 9) * 9
        rank = first % 9
        if rank <= 6 and first + 2 < suit_offset + 9:
            if current[first + 1] > 0 and current[first + 2] > 0:
                current[first] -= 1
                current[first + 1] -= 1
                current[first + 2] -= 1
                if _can_form_all_sets(tuple(current)):
                    return True

    return False
