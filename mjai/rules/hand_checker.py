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


def shanten_number(hand: HandState) -> int:
    """Return the minimum shanten across standard, seven-pairs, and kokushi shapes."""
    return shanten_number_from_counts(tuple(hand.counts))


def effective_tile_base_indices(hand: HandState) -> set[int]:
    """Return base tiles that improve the current hand's shanten."""
    return effective_tile_base_indices_from_counts(tuple(hand.counts))


@lru_cache(maxsize=1_000_000)
def shanten_number_from_counts(counts: tuple[int, ...]) -> int:
    """Return the minimum shanten for a 34-count hand tuple.

    Complete hands return -1. The calculation is intentionally compact and deterministic
    so it can be reused by feature extraction during batch collation.
    """
    if sum(counts) == 0:
        return 8
    return min(
        _standard_shanten(counts),
        _chiitoitsu_shanten(counts),
        _kokushi_shanten(counts),
    )


@lru_cache(maxsize=1_000_000)
def effective_tile_base_indices_from_counts(counts: tuple[int, ...]) -> frozenset[int]:
    current_shanten = shanten_number_from_counts(counts)
    improving_tiles: set[int] = set()
    for base_index, count in enumerate(counts):
        if count >= 4:
            continue
        next_counts = list(counts)
        next_counts[base_index] += 1
        if shanten_number_from_counts(tuple(next_counts)) < current_shanten:
            improving_tiles.add(base_index)
    return frozenset(improving_tiles)


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


def _standard_shanten(counts: tuple[int, ...]) -> int:
    return _standard_shanten_by_groups(counts)


def _standard_shanten_by_groups(counts: tuple[int, ...]) -> int:
    option_groups = (
        _suit_shanten_options(counts[0:9]),
        _suit_shanten_options(counts[9:18]),
        _suit_shanten_options(counts[18:27]),
        _honor_shanten_options(counts[27:34]),
    )
    best = 8
    for first in option_groups[0]:
        for second in option_groups[1]:
            first_second = (
                first[0] + second[0],
                first[1] + second[1],
                first[2] + second[2],
            )
            for third in option_groups[2]:
                first_three = (
                    first_second[0] + third[0],
                    first_second[1] + third[1],
                    first_second[2] + third[2],
                )
                for fourth in option_groups[3]:
                    melds = first_three[0] + fourth[0]
                    taatsu = first_three[1] + fourth[1]
                    pairs = first_three[2] + fourth[2]
                    pair_count = min(1, pairs)
                    extra_pair_taatsu = max(0, pairs - pair_count)
                    useful_taatsu = min(taatsu + extra_pair_taatsu, 4 - melds)
                    best = min(best, 8 - 2 * melds - useful_taatsu - pair_count)
    return best


@lru_cache(maxsize=500_000)
def _suit_shanten_options(counts: tuple[int, ...]) -> frozenset[tuple[int, int, int]]:
    return frozenset(_prune_shanten_options(_build_group_options(counts, honor=False)))


@lru_cache(maxsize=100_000)
def _honor_shanten_options(counts: tuple[int, ...]) -> frozenset[tuple[int, int, int]]:
    return frozenset(_prune_shanten_options(_build_group_options(counts, honor=True)))


def _build_group_options(counts: tuple[int, ...], honor: bool) -> set[tuple[int, int, int]]:
    size = len(counts)

    @lru_cache(maxsize=None)
    def visit(current_counts: tuple[int, ...]) -> frozenset[tuple[int, int, int]]:
        try:
            first = next(index for index, count in enumerate(current_counts) if count > 0)
        except StopIteration:
            return frozenset({(0, 0, 0)})

        options: set[tuple[int, int, int]] = set()

        current = list(current_counts)
        current[first] -= 1
        options.update(visit(tuple(current)))

        current = list(current_counts)
        if current[first] >= 3:
            current[first] -= 3
            options.update(_add_to_options(visit(tuple(current)), melds=1))

        current = list(current_counts)
        if current[first] >= 2:
            current[first] -= 2
            options.update(_add_to_options(visit(tuple(current)), pairs=1))

        if not honor:
            rank = first
            current = list(current_counts)
            if rank <= size - 3 and current[first + 1] > 0 and current[first + 2] > 0:
                current[first] -= 1
                current[first + 1] -= 1
                current[first + 2] -= 1
                options.update(_add_to_options(visit(tuple(current)), melds=1))

            current = list(current_counts)
            if rank <= size - 2 and current[first + 1] > 0:
                current[first] -= 1
                current[first + 1] -= 1
                options.update(_add_to_options(visit(tuple(current)), taatsu=1))

            current = list(current_counts)
            if rank <= size - 3 and current[first + 2] > 0:
                current[first] -= 1
                current[first + 2] -= 1
                options.update(_add_to_options(visit(tuple(current)), taatsu=1))

        return frozenset(options)

    return set(visit(counts))


def _add_to_options(
    options: frozenset[tuple[int, int, int]],
    melds: int = 0,
    taatsu: int = 0,
    pairs: int = 0,
) -> set[tuple[int, int, int]]:
    return {
        (option_melds + melds, option_taatsu + taatsu, option_pairs + pairs)
        for option_melds, option_taatsu, option_pairs in options
    }


def _prune_shanten_options(options: set[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    pruned: set[tuple[int, int, int]] = set()
    for option in options:
        melds, taatsu, pairs = option
        dominated = False
        for other in options:
            if other == option:
                continue
            other_melds, other_taatsu, other_pairs = other
            if (
                other_melds >= melds
                and other_taatsu >= taatsu
                and other_pairs >= pairs
                and (other_melds, other_taatsu, other_pairs) != (melds, taatsu, pairs)
            ):
                dominated = True
                break
        if not dominated:
            pruned.add(option)
    return pruned


def _chiitoitsu_shanten(counts: tuple[int, ...]) -> int:
    pair_count = sum(1 for count in counts if count >= 2)
    unique_count = sum(1 for count in counts if count > 0)
    return 6 - pair_count + max(0, 7 - unique_count)


def _kokushi_shanten(counts: tuple[int, ...]) -> int:
    terminals = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    unique_terminals = sum(1 for index in terminals if counts[index] > 0)
    has_terminal_pair = any(counts[index] >= 2 for index in terminals)
    return 13 - unique_terminals - int(has_terminal_pair)


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
