from __future__ import annotations

from functools import lru_cache

from mjai.rules.state import HandState
from mjai.rules.tiles import base_index_to_tile, tile_to_base_index


def is_agari(hand: HandState, incoming_tile: str | None = None) -> bool:
    """判断手牌在可选进张后是否和牌。
    
    参数:
        hand (HandState): 手牌状态。
        incoming_tile (str | None): 函数执行所需的输入参数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    counts = hand.counts.copy()
    if incoming_tile is not None:
        counts[tile_to_base_index(incoming_tile)] += 1

    return _is_agari_counts(tuple(counts))


def is_tenpai_after_discard(hand: HandState, discard_tile: str) -> bool:
    """判断弃掉指定精确牌后是否听牌。
    
    参数:
        hand (HandState): 手牌状态。
        discard_tile (str): 函数执行所需的输入参数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if not hand.has_tile(discard_tile):
        return False

    reduced = hand.copy()
    reduced.remove_tile(discard_tile)

    return bool(winning_tile_base_indices(reduced))


def winning_tile_base_indices(hand: HandState) -> set[int]:
    """计算当前手牌的所有和牌基础索引。
    
    参数:
        hand (HandState): 手牌状态。
    
    返回:
        set[int]: 集合结果。
    
    异常:
        无。
    """
    winning_tiles: set[int] = set()
    for base_index, count in enumerate(hand.counts):
        if count >= 4:
            continue
        if is_agari(hand, incoming_tile=base_index_to_tile(base_index)):
            winning_tiles.add(base_index)
    return winning_tiles


def shanten_number(hand: HandState) -> int:
    """计算手牌在多种手型下的最小向听数。
    
    参数:
        hand (HandState): 手牌状态。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    return shanten_number_from_counts(tuple(hand.counts))


def effective_tile_base_indices(hand: HandState) -> set[int]:
    """计算能改善当前向听数的基础索引集合。
    
    参数:
        hand (HandState): 手牌状态。
    
    返回:
        set[int]: 集合结果。
    
    异常:
        无。
    """
    return effective_tile_base_indices_from_counts(tuple(hand.counts))


@lru_cache(maxsize=1_000_000)
def shanten_number_from_counts(counts: tuple[int, ...]) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
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
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        frozenset[int]: 集合结果。
    
    异常:
        无。
    """
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
def standard_shanten_from_counts_with_fixed_melds(
    counts: tuple[int, ...],
    fixed_melds: int,
) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
        fixed_melds (int): 相关数据列表。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    fixed_meld_count = max(0, min(4, int(fixed_melds)))
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
                    closed_melds = first_three[0] + fourth[0]
                    taatsu = first_three[1] + fourth[1]
                    pairs = first_three[2] + fourth[2]
                    melds = min(4, closed_melds + fixed_meld_count)
                    pair_count = min(1, pairs)
                    extra_pair_taatsu = max(0, pairs - pair_count)
                    useful_taatsu = min(taatsu + extra_pair_taatsu, max(0, 4 - melds))
                    best = min(best, 8 - 2 * melds - useful_taatsu - pair_count)
    return best


@lru_cache(maxsize=1_000_000)
def effective_tile_base_indices_from_counts_with_fixed_melds(
    counts: tuple[int, ...],
    fixed_melds: int,
) -> frozenset[int]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
        fixed_melds (int): 相关数据列表。
    
    返回:
        frozenset[int]: 集合结果。
    
    异常:
        无。
    """
    current_shanten = standard_shanten_from_counts_with_fixed_melds(counts, fixed_melds)
    improving_tiles: set[int] = set()
    for base_index, count in enumerate(counts):
        if count >= 4:
            continue
        next_counts = list(counts)
        next_counts[base_index] += 1
        if (
            standard_shanten_from_counts_with_fixed_melds(tuple(next_counts), fixed_melds)
            < current_shanten
        ):
            improving_tiles.add(base_index)
    return frozenset(improving_tiles)


@lru_cache(maxsize=1_000_000)
def _is_agari_counts(counts: tuple[int, ...]) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    total_tiles = sum(counts)
    if total_tiles % 3 != 2:
        return False

    if _is_chiitoitsu(counts):
        return True
    if _is_kokushi(counts):
        return True
    return _is_standard_agari(counts)


def _is_chiitoitsu(counts: tuple[int, ...]) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    return sum(1 for count in counts if count == 2) == 7 and all(count in {0, 2} for count in counts)


def _is_kokushi(counts: tuple[int, ...]) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
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
    """判断与函数名对应的条件是否成立。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    for index, count in enumerate(counts):
        if count < 2:
            continue
        reduced = list(counts)
        reduced[index] -= 2
        if _can_form_all_sets_by_group(tuple(reduced)):
            return True
    return False


def _can_form_all_sets_by_group(counts: tuple[int, ...]) -> bool:
    """判断与函数名对应的操作是否可执行。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    return (
        _can_form_suit_sets(counts[0:9])
        and _can_form_suit_sets(counts[9:18])
        and _can_form_suit_sets(counts[18:27])
        and _can_form_honor_sets(counts[27:34])
    )


@lru_cache(maxsize=250_000)
def _can_form_suit_sets(counts: tuple[int, ...]) -> bool:
    """判断与函数名对应的操作是否可执行。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
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
    """判断与函数名对应的操作是否可执行。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    return all(count in {0, 3} for count in counts)


def _standard_shanten(counts: tuple[int, ...]) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    return _standard_shanten_by_groups(counts)


def _standard_shanten_by_groups(counts: tuple[int, ...]) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
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
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        frozenset[tuple[int, int, int]]: 集合结果。
    
    异常:
        无。
    """
    return frozenset(_prune_shanten_options(_build_group_options(counts, honor=False)))


@lru_cache(maxsize=100_000)
def _honor_shanten_options(counts: tuple[int, ...]) -> frozenset[tuple[int, int, int]]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        frozenset[tuple[int, int, int]]: 集合结果。
    
    异常:
        无。
    """
    return frozenset(_prune_shanten_options(_build_group_options(counts, honor=True)))


def _build_group_options(counts: tuple[int, ...], honor: bool) -> set[tuple[int, int, int]]:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
        honor (bool): 函数执行所需的输入参数。
    
    返回:
        set[tuple[int, int, int]]: 集合结果。
    
    异常:
        无。
    """
    size = len(counts)

    @lru_cache(maxsize=None)
    def visit(current_counts: tuple[int, ...]) -> frozenset[tuple[int, int, int]]:
        """执行与函数名对应的业务逻辑。
        
        参数:
            current_counts (tuple[int, ...]): 相关数据列表。
        
        返回:
            frozenset[tuple[int, int, int]]: 集合结果。
        
        异常:
            无。
        """
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
    """向当前状态添加输入数据。
    
    参数:
        options (frozenset[tuple[int, int, int]]): 相关数据列表。
        melds (int): 相关数据列表。
        taatsu (int): 函数执行所需的输入参数。
        pairs (int): 相关数据列表。
    
    返回:
        set[tuple[int, int, int]]: 集合结果。
    
    异常:
        无。
    """
    return {
        (option_melds + melds, option_taatsu + taatsu, option_pairs + pairs)
        for option_melds, option_taatsu, option_pairs in options
    }


def _prune_shanten_options(options: set[tuple[int, int, int]]) -> set[tuple[int, int, int]]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        options (set[tuple[int, int, int]]): 相关数据列表。
    
    返回:
        set[tuple[int, int, int]]: 集合结果。
    
    异常:
        无。
    """
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
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    pair_count = sum(1 for count in counts if count >= 2)
    unique_count = sum(1 for count in counts if count > 0)
    return 6 - pair_count + max(0, 7 - unique_count)


def _kokushi_shanten(counts: tuple[int, ...]) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    terminals = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    unique_terminals = sum(1 for index in terminals if counts[index] > 0)
    has_terminal_pair = any(counts[index] >= 2 for index in terminals)
    return 13 - unique_terminals - int(has_terminal_pair)


@lru_cache(maxsize=500_000)
def _can_form_all_sets(counts: tuple[int, ...]) -> bool:
    """判断与函数名对应的操作是否可执行。
    
    参数:
        counts (tuple[int, ...]): 三十四种牌计数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
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
