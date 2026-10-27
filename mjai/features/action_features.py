from __future__ import annotations

from mjai.features.perspective import relative_action, relative_state
from mjai.features.state_features import EXACT_TILE_INDEX, EXACT_TILE_VOCAB
from mjai.rules.hand_checker import (
    effective_tile_base_indices_from_counts,
    effective_tile_base_indices_from_counts_with_fixed_melds,
    shanten_number_from_counts,
    standard_shanten_from_counts_with_fixed_melds,
)
from mjai.rules.tiles import AKA_BASE_INDICES, dora_from_indicator, tile_to_base_index


TURN_ACTION_KINDS = ("discard", "riichi_discard", "tsumo", "ankan", "kakan", "kyuushu_kyuuhai", "unknown")
# Candidate action layouts are documented in docs/model-inputs.md. These dims
# are serialized into parquet, so changing them requires rebuilding samples.
TURN_RIVER_RECENT_WINDOW = 6
TURN_RIVER_EARLY_WINDOW = 6
TURN_RIVER_TURN_SCALE = 18.0
TURN_DISCARD_FEATURE_DIM = 45
TURN_ACTION_DIM = len(TURN_ACTION_KINDS) + len(EXACT_TILE_VOCAB) + TURN_DISCARD_FEATURE_DIM
RESPONSE_ACTION_KINDS = ("pass", "chi", "pon", "daiminkan", "ron", "chankan", "unknown")
RESPONSE_CALL_FEATURE_DIM = 43
RESPONSE_ACTION_DIM = (
    len(RESPONSE_ACTION_KINDS)
    + len(EXACT_TILE_VOCAB)
    + 4
    + RESPONSE_CALL_FEATURE_DIM
)


def _encode_turn_action(
    action: dict,
    state: dict | None = None,
    context: dict | None = None,
) -> list[float]:
    # Layout: action kind one-hot | action tile one-hot | discard-candidate features.
    # Non-discard turn actions leave the discard-candidate block as zero.
    """将自摸巡候选动作编码为定长特征向量。
    
    参数:
        action (dict): 动作字典或动作对象。
        state (dict | None): 游戏状态或样本状态。
        context (dict | None): 预计算特征上下文。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    vector = [0.0] * TURN_ACTION_DIM

    kind = str(action.get("kind", "unknown"))
    try:
        vector[TURN_ACTION_KINDS.index(kind)] = 1.0
    except ValueError:
        vector[TURN_ACTION_KINDS.index("unknown")] = 1.0

    tile = action.get("tile")
    if tile is not None:
        vector[len(TURN_ACTION_KINDS) + EXACT_TILE_INDEX[str(tile)]] = 1.0
    feature_offset = len(TURN_ACTION_KINDS) + len(EXACT_TILE_VOCAB)
    discard_features = encode_turn_discard_features(action, state, context)
    vector[feature_offset : feature_offset + TURN_DISCARD_FEATURE_DIM] = discard_features
    return vector


def encode_turn_candidate_actions(actions: list[dict], state: dict) -> list[list[float]]:
    """批量编码一个样本的自摸巡候选动作。
    
    参数:
        actions (list[dict]): 动作列表。
        state (dict): 游戏状态或样本状态。
    
    返回:
        list[list[float]]: 列表结果。
    
    异常:
        无。
    """
    actor = int(state.get("actor", 0))
    state = relative_state(state)
    context = build_turn_discard_feature_context(state)
    return [_encode_turn_action(relative_action(action, actor), state, context) for action in actions]


def _encode_response_action(
    action: dict,
    state: dict | None = None,
    context: dict | None = None,
) -> list[float]:
    # Layout: response kind one-hot | target tile one-hot | source actor one-hot
    # | call/context features. Pass/ron/chankan still receive context features
    # so they can be compared with call candidates in the same legal set.
    """将响应候选动作编码为定长特征向量。
    
    参数:
        action (dict): 动作字典或动作对象。
        state (dict | None): 游戏状态或样本状态。
        context (dict | None): 预计算特征上下文。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    vector = [0.0] * RESPONSE_ACTION_DIM

    kind = str(action.get("kind", "unknown"))
    try:
        vector[RESPONSE_ACTION_KINDS.index(kind)] = 1.0
    except ValueError:
        vector[RESPONSE_ACTION_KINDS.index("unknown")] = 1.0

    tile = action.get("tile")
    offset = len(RESPONSE_ACTION_KINDS)
    if tile is not None:
        vector[offset + EXACT_TILE_INDEX[str(tile)]] = 1.0

    source = action.get("source")
    if source is not None:
        vector[offset + len(EXACT_TILE_VOCAB) + int(source)] = 1.0
    feature_offset = offset + len(EXACT_TILE_VOCAB) + 4
    call_features = encode_response_call_features(action, state, context)
    vector[feature_offset : feature_offset + RESPONSE_CALL_FEATURE_DIM] = call_features
    return vector


def encode_response_candidate_actions(actions: list[dict], state: dict) -> list[list[float]]:
    """批量编码一个样本的响应候选动作。
    
    参数:
        actions (list[dict]): 动作列表。
        state (dict): 游戏状态或样本状态。
    
    返回:
        list[list[float]]: 列表结果。
    
    异常:
        无。
    """
    actor = int(state.get("actor", 0))
    state = relative_state(state)
    context = build_response_call_feature_context(state)
    return [_encode_response_action(relative_action(action, actor), state, context) for action in actions]


def build_turn_discard_feature_context(state: dict | None) -> dict | None:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        state (dict | None): 游戏状态或样本状态。
    
    返回:
        dict | None: 字典结果。
    
    异常:
        无。
    """
    if state is None:
        return None
    counts = _hand_counts_from_state(state)
    fixed_melds = _fixed_meld_count(state)
    # Shared once per sample so each discard candidate does not rerun the same
    # current-hand shanten and dora extraction.
    return {
        "counts": counts,
        "current_shanten": _turn_shanten_from_counts(tuple(counts), fixed_melds),
        "fixed_melds": fixed_melds,
        "dora_bases": _dora_base_indices(state),
        "discard_timing": _discard_timing_context(state),
    }


def build_response_call_feature_context(state: dict | None) -> dict | None:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        state (dict | None): 游戏状态或样本状态。
    
    返回:
        dict | None: 字典结果。
    
    异常:
        无。
    """
    if state is None:
        return None
    counts = _hand_counts_from_state(state)
    current_effective_tiles = effective_tile_base_indices_from_counts(tuple(counts))
    current_visible_effective_tile_copies = _visible_effective_tile_copies(
        state,
        counts,
        current_effective_tiles,
    )
    # Shared once per response sample. Candidate-specific encoding only mutates
    # a copy for the call being evaluated.
    return {
        "counts": counts,
        "current_shanten": shanten_number_from_counts(tuple(counts)),
        "current_chiitoitsu_shanten": _chiitoitsu_shanten_from_counts(tuple(counts)),
        "current_kokushi_shanten": _kokushi_shanten_from_counts(tuple(counts)),
        "current_effective_tiles": current_effective_tiles,
        "current_effective_tile_copies": _effective_tile_copies(counts, current_effective_tiles),
        "current_visible_effective_tile_copies": current_visible_effective_tile_copies,
        "dora_bases": _dora_base_indices(state),
        "has_any_melds": bool(state.get("own_melds", [])),
        "has_open_melds": _has_open_melds(state),
    }


def encode_response_call_features(
    action: dict,
    state: dict | None,
    context: dict | None = None,
) -> list[float]:
    """编码响应动作的鸣牌价值与代价特征。
    
    参数:
        action (dict): 动作字典或动作对象。
        state (dict | None): 游戏状态或样本状态。
        context (dict | None): 预计算特征上下文。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    features = [0.0] * RESPONSE_CALL_FEATURE_DIM
    if state is None:
        return features
    if context is None:
        context = build_response_call_feature_context(state)
    assert context is not None

    kind = str(action.get("kind", "unknown"))
    counts = list(context["counts"])
    current_shanten = int(context["current_shanten"])
    current_effective_tiles = context["current_effective_tiles"]
    current_effective_tile_copies = int(context["current_effective_tile_copies"])
    current_visible_effective_tile_copies = int(context["current_visible_effective_tile_copies"])
    current_chiitoitsu_shanten = int(context["current_chiitoitsu_shanten"])
    current_kokushi_shanten = int(context["current_kokushi_shanten"])
    has_any_melds = bool(context["has_any_melds"])
    has_open_melds = bool(context["has_open_melds"])
    target_tile = action.get("tile") or state.get("response_target", {}).get("tile")
    target_base = tile_to_base_index(str(target_tile)) if target_tile is not None else None
    consumed_tiles = [str(tile) for tile in action.get("consumed_tiles", [])]
    consumed_bases = [tile_to_base_index(tile) for tile in consumed_tiles]
    dora_bases = context["dora_bases"]

    after_counts = list(counts)
    action_is_call = kind in {"chi", "pon", "daiminkan"}
    # For calls, after_* features simulate consuming the tiles and adding one
    # fixed meld. For pass/ron/chankan, after_* intentionally equals current.
    call_is_valid = action_is_call and _can_apply_response_call(after_counts, consumed_tiles)
    if call_is_valid:
        for tile in consumed_tiles:
            after_counts[tile_to_base_index(tile)] -= 1
        after_counts_tuple = tuple(after_counts)
        fixed_melds_after_call = _fixed_melds_after_response_call(state)
        after_shanten = standard_shanten_from_counts_with_fixed_melds(
            after_counts_tuple,
            fixed_melds_after_call,
        )
        after_effective_tiles = effective_tile_base_indices_from_counts_with_fixed_melds(
            after_counts_tuple,
            fixed_melds_after_call,
        )
        after_effective_tile_copies = _effective_tile_copies(after_counts, after_effective_tiles)
        after_visible_effective_tile_copies = _visible_effective_tile_copies(
            state,
            after_counts,
            after_effective_tiles,
            extra_visible_bases=consumed_bases,
        )
        after_shape = _shape_summary(after_counts)
    else:
        after_shanten = current_shanten
        after_effective_tiles = current_effective_tiles
        after_effective_tile_copies = current_effective_tile_copies
        after_visible_effective_tile_copies = current_visible_effective_tile_copies
        after_shape = _shape_summary(after_counts)

    opens_closed_hand = call_is_valid and not has_open_melds
    after_hand_is_open = call_is_valid and (has_open_melds or opens_closed_hand)
    yaku_routes = _open_yaku_route_summary(
        state=state,
        kind=kind,
        target_base=target_base,
        consumed_bases=consumed_bases,
        after_counts=after_counts,
    )
    has_obvious_open_yaku_route = any(yaku_routes.values())
    no_obvious_open_yaku = after_hand_is_open and not has_obvious_open_yaku_route
    chiitoitsu_route_active = (
        call_is_valid
        and not has_any_melds
        and current_chiitoitsu_shanten <= 2
        and current_chiitoitsu_shanten == current_shanten
    )
    kokushi_route_active = (
        call_is_valid
        and not has_any_melds
        and current_kokushi_shanten <= 2
        and current_kokushi_shanten == current_shanten
    )
    special_route_better_than_open = (
        call_is_valid
        and not has_any_melds
        and min(current_chiitoitsu_shanten, current_kokushi_shanten) <= 2
        and min(current_chiitoitsu_shanten, current_kokushi_shanten) < after_shanten
    )

    features[0] = 1.0 if action_is_call else 0.0
    features[1] = 1.0 if call_is_valid else 0.0
    features[2] = _scale_shanten(current_shanten)
    features[3] = _scale_shanten(after_shanten)
    features[4] = float(current_shanten - after_shanten)
    features[5] = min(1.0, len(current_effective_tiles) / 34.0)
    features[6] = min(1.0, len(after_effective_tiles) / 34.0)
    features[7] = _scale_delta(len(after_effective_tiles) - len(current_effective_tiles), 34.0)
    features[8] = min(1.0, current_effective_tile_copies / 136.0)
    features[9] = min(1.0, after_effective_tile_copies / 136.0)
    features[10] = _scale_delta(after_effective_tile_copies - current_effective_tile_copies, 136.0)
    features[11] = 1.0 if any(tile.startswith("0") for tile in consumed_tiles) else 0.0
    features[12] = 1.0 if _action_consumes_dora(consumed_tiles, dora_bases) else 0.0
    features[13] = 1.0 if target_base is not None and target_base in dora_bases else 0.0
    features[14] = 1.0 if _consumes_pair(counts, consumed_tiles) else 0.0
    features[15] = 1.0 if _breaks_taatsu(counts, consumed_tiles) else 0.0
    features[16] = 1.0 if kind == "pon" and target_base is not None and _is_yakuhai(state, target_base) else 0.0
    features[17] = 1.0 if target_base is not None and _is_terminal(target_base) else 0.0
    features[18] = 1.0 if target_base is not None and _is_honor(target_base) else 0.0
    features[19] = 1.0 if kind == "chi" and _chi_shape(action) == "ryanmen" else 0.0
    features[20] = 1.0 if kind == "chi" and _chi_shape(action) == "closed_wait" else 0.0
    features[21] = 1.0 if kind == "chi" and _chi_shape(action) == "edge_wait" else 0.0
    features[22] = min(1.0, after_shape["complete_sets"] / 4.0)
    features[23] = min(1.0, after_shape["pairs"] / 7.0)
    features[24] = min(1.0, after_shape["ryanmen"] / 8.0)
    features[25] = min(1.0, after_shape["closed_waits"] / 8.0)
    features[26] = min(1.0, after_shape["isolated"] / 14.0)
    features[27] = min(1.0, float(state.get("turn_number", 0)) / 18.0)
    features[28] = min(1.0, float(state.get("remaining_draws_estimate", 0)) / 18.0)
    features[29] = min(1.0, current_visible_effective_tile_copies / 136.0)
    features[30] = min(1.0, after_visible_effective_tile_copies / 136.0)
    features[31] = _scale_delta(
        after_visible_effective_tile_copies - current_visible_effective_tile_copies,
        136.0,
    )
    features[32] = 1.0 if opens_closed_hand else 0.0
    features[33] = 1.0 if after_hand_is_open else 0.0
    features[34] = 1.0 if call_is_valid and yaku_routes["yakuhai"] else 0.0
    features[35] = 1.0 if call_is_valid and yaku_routes["tanyao"] else 0.0
    features[36] = 1.0 if call_is_valid and yaku_routes["flush"] else 0.0
    features[37] = 1.0 if call_is_valid and yaku_routes["toitoi"] else 0.0
    features[38] = 1.0 if no_obvious_open_yaku else 0.0
    features[39] = _scale_shanten(current_chiitoitsu_shanten)
    features[40] = 1.0 if chiitoitsu_route_active else 0.0
    features[41] = _scale_shanten(current_kokushi_shanten)
    features[42] = 1.0 if kokushi_route_active or special_route_better_than_open else 0.0
    return features


def encode_turn_discard_features(
    action: dict,
    state: dict | None,
    context: dict | None = None,
) -> list[float]:
    """编码弃牌类自摸巡动作的候选特征。
    
    参数:
        action (dict): 动作字典或动作对象。
        state (dict | None): 游戏状态或样本状态。
        context (dict | None): 预计算特征上下文。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    features = [0.0] * TURN_DISCARD_FEATURE_DIM
    kind = str(action.get("kind", "unknown"))
    tile = action.get("tile")
    if kind not in {"discard", "riichi_discard"} or tile is None or state is None:
        return features

    if context is None:
        context = build_turn_discard_feature_context(state)
    assert context is not None
    counts = list(context["counts"])
    base_index = tile_to_base_index(str(tile))
    if counts[base_index] <= 0:
        return features

    # Candidate-specific discard features simulate the hand after throwing this
    # exact tile. They are the main signal for choosing among discard candidates.
    after_counts = list(counts)
    after_counts[base_index] -= 1
    after_counts_tuple = tuple(after_counts)
    current_shanten = int(context["current_shanten"])
    fixed_melds = int(context.get("fixed_melds", 0))
    after_shanten = _turn_shanten_from_counts(after_counts_tuple, fixed_melds)
    effective_tiles = _turn_effective_tile_base_indices(after_counts_tuple, fixed_melds)
    effective_tile_copies = _visible_effective_tile_copies(
        state,
        after_counts,
        effective_tiles,
        include_called_meld_tiles=True,
    )
    shape = _shape_summary(after_counts)

    features[0] = 1.0
    features[1] = _scale_shanten(current_shanten)
    features[2] = _scale_shanten(after_shanten)
    features[3] = float(current_shanten - after_shanten)
    features[4] = min(1.0, len(effective_tiles) / 34.0)
    features[5] = min(1.0, effective_tile_copies / 136.0)
    features[6] = min(1.0, shape["complete_sets"] / 4.0)
    features[7] = min(1.0, shape["pairs"] / 7.0)
    features[8] = min(1.0, shape["ryanmen"] / 8.0)
    features[9] = min(1.0, shape["closed_waits"] / 8.0)
    features[10] = min(1.0, shape["isolated"] / 14.0)
    features[11] = min(1.0, after_counts[base_index] / 4.0)
    features[12] = 1.0 if _is_terminal(base_index) else 0.0
    features[13] = 1.0 if _is_honor(base_index) else 0.0
    features[14] = 1.0 if base_index in AKA_BASE_INDICES else 0.0
    features[15] = 1.0 if str(tile).startswith("0") else 0.0
    features[16] = 1.0 if state.get("riichi_declared", False) else 0.0
    features[17] = 1.0 if kind == "riichi_discard" else 0.0
    features[18] = 1.0 if _is_genbutsu(state, base_index) else 0.0
    features[19] = min(1.0, _visible_count(state, base_index) / 4.0)
    features[20] = 1.0 if _is_suji_safe(state, base_index) else 0.0
    features[21] = 1.0 if _is_one_chance(state, after_counts, base_index) else 0.0
    features[22] = 1.0 if after_shanten == 0 else 0.0
    features[23] = 1.0 if after_shanten < current_shanten else 0.0
    features[24] = 1.0 if after_shanten > current_shanten else 0.0
    _fill_discard_timing_features(features, context["discard_timing"], base_index)
    dora_bases = context["dora_bases"]
    features[14] = min(1.0, (1.0 if base_index in AKA_BASE_INDICES else 0.0) + float(base_index in dora_bases))
    return features


def _hand_counts_from_state(state: dict) -> list[int]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        state (dict): 游戏状态或样本状态。
    
    返回:
        list[int]: 列表结果。
    
    异常:
        无。
    """
    counts = [0] * 34
    for tile, count in state.get("hand", {}).items():
        counts[tile_to_base_index(str(tile))] += int(count)
    return counts


def _can_apply_response_call(counts: list[int], consumed_tiles: list[str]) -> bool:
    """判断与函数名对应的操作是否可执行。
    
    参数:
        counts (list[int]): 三十四种牌计数。
        consumed_tiles (list[str]): 副露消耗牌列表。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if not consumed_tiles:
        return False
    available_counts = list(counts)
    for tile in consumed_tiles:
        base_index = tile_to_base_index(tile)
        if available_counts[base_index] <= 0:
            return False
        available_counts[base_index] -= 1
    return True


def _fixed_melds_after_response_call(state: dict) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        state (dict): 游戏状态或样本状态。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    return min(4, len(state.get("own_melds", [])) + 1)


def _fixed_meld_count(state: dict) -> int:
    return min(4, len(state.get("own_melds", [])))


def _turn_shanten_from_counts(counts: tuple[int, ...], fixed_melds: int) -> int:
    if fixed_melds <= 0:
        return shanten_number_from_counts(counts)
    return standard_shanten_from_counts_with_fixed_melds(counts, fixed_melds)


def _turn_effective_tile_base_indices(counts: tuple[int, ...], fixed_melds: int):
    if fixed_melds <= 0:
        return effective_tile_base_indices_from_counts(counts)
    return effective_tile_base_indices_from_counts_with_fixed_melds(counts, fixed_melds)


def _effective_tile_copies(counts: list[int], effective_tiles) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (list[int]): 三十四种牌计数。
        effective_tiles (list[Any]): 相关数据列表。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    return sum(max(0, 4 - counts[index]) for index in effective_tiles)


def _visible_effective_tile_copies(
    state: dict,
    counts: list[int],
    effective_tiles,
    extra_visible_bases: list[int] | None = None,
    *,
    include_called_meld_tiles: bool = False,
) -> int:
    """按当前可见牌修正有效牌枚数。"""
    extra_visible_bases = extra_visible_bases or []
    return sum(
        max(
            0,
            4
            - _visible_base_count_with_hand_counts(
                state,
                counts,
                index,
                extra_visible_bases,
                include_called_meld_tiles=include_called_meld_tiles,
            ),
        )
        for index in effective_tiles
    )


def _visible_base_count_with_hand_counts(
    state: dict,
    hand_counts: list[int],
    base_index: int,
    extra_visible_bases: list[int],
    *,
    include_called_meld_tiles: bool = False,
) -> int:
    """统计某 base tile 已知不可摸的张数。"""
    visible = int(hand_counts[base_index])

    for player_discards in state.get("all_discards", []):
        for tile in player_discards:
            if tile_to_base_index(str(tile)) == base_index:
                visible += 1

    meld_tiles = (
        _visible_meld_tiles_with_called_tiles(state)
        if include_called_meld_tiles
        else _visible_meld_tiles_without_called_tiles(state)
    )
    for tile in meld_tiles:
        if tile_to_base_index(tile) == base_index:
            visible += 1

    for tile in state.get("dora_indicators", []):
        if tile_to_base_index(str(tile)) == base_index:
            visible += 1

    visible += sum(1 for extra_base in extra_visible_bases if extra_base == base_index)
    return min(4, visible)


def _visible_meld_tiles_without_called_tiles(state: dict):
    for meld in _visible_melds(state):
        for tile in _meld_visible_tiles(meld):
            yield tile


def _visible_meld_tiles_with_called_tiles(state: dict):
    called_tile_quotas = _missing_called_tile_quotas(state)
    for meld in _visible_melds(state):
        kind = str(meld.get("kind", ""))
        if kind == "ankan":
            yield from _meld_tiles(meld)
            continue

        for tile in meld.get("consumed_tiles", []):
            yield str(tile)

        called_tile = meld.get("called_tile")
        if called_tile is None:
            continue
        called_base = tile_to_base_index(str(called_tile))
        source = meld.get("source")
        if source is None:
            yield str(called_tile)
            continue

        key = (int(source), called_base)
        if called_tile_quotas.get(key, 0) > 0:
            called_tile_quotas[key] -= 1
            yield str(called_tile)


def _missing_called_tile_quotas(state: dict) -> dict[tuple[int, int], int]:
    history_counts: dict[tuple[int, int], int] = {}
    for event in state.get("discard_history", []):
        if isinstance(event, dict):
            actor = event.get("actor")
            tile = event.get("tile")
        else:
            try:
                actor, tile = event
            except (TypeError, ValueError):
                continue
        if actor is None or tile is None:
            continue
        key = (int(actor), tile_to_base_index(str(tile)))
        history_counts[key] = history_counts.get(key, 0) + 1

    if not history_counts:
        quotas: dict[tuple[int, int], int] = {}
        for meld in _visible_melds(state):
            called_tile = meld.get("called_tile")
            source = meld.get("source")
            if called_tile is None or source is None:
                continue
            called_base = tile_to_base_index(str(called_tile))
            if _source_discards_include_base(state, int(source), called_base):
                continue
            key = (int(source), called_base)
            quotas[key] = quotas.get(key, 0) + 1
        return quotas

    river_counts: dict[tuple[int, int], int] = {}
    for actor, player_discards in enumerate(state.get("all_discards", [])):
        for tile in player_discards:
            key = (actor, tile_to_base_index(str(tile)))
            river_counts[key] = river_counts.get(key, 0) + 1

    return {
        key: max(0, history_count - river_counts.get(key, 0))
        for key, history_count in history_counts.items()
    }


def _source_discards_include_base(state: dict, source: int, base_index: int) -> bool:
    all_discards = state.get("all_discards", [])
    if source < 0 or source >= len(all_discards):
        return False
    return any(tile_to_base_index(str(tile)) == base_index for tile in all_discards[source])


def _visible_count_with_called_meld_tiles(
    state: dict,
    base_index: int,
    include_hand: bool = True,
) -> int:
    hand_counts = _hand_counts_from_state(state) if include_hand else [0] * 34
    return _visible_base_count_with_hand_counts(
        state,
        hand_counts,
        base_index,
        [],
        include_called_meld_tiles=True,
    )

def _open_yaku_route_summary(
    *,
    state: dict,
    kind: str,
    target_base: int | None,
    consumed_bases: list[int],
    after_counts: list[int],
) -> dict[str, bool]:
    """粗粒度判断开副露后是否存在明显役路。"""
    candidate_bases = ([target_base] if target_base is not None else []) + list(consumed_bases)
    own_melds = list(state.get("own_melds", []))
    meld_tile_bases = _existing_meld_tile_bases(own_melds) + candidate_bases
    triplet_meld_bases = _triplet_meld_bases(own_melds)
    if kind in {"pon", "daiminkan"} and target_base is not None:
        triplet_meld_bases.append(target_base)

    return {
        "yakuhai": _has_yakuhai_route(state, after_counts, triplet_meld_bases),
        "tanyao": _has_tanyao_route(after_counts, meld_tile_bases),
        "flush": _has_flush_route(after_counts, meld_tile_bases),
        "toitoi": _has_toitoi_route(kind, state, after_counts, triplet_meld_bases),
    }


def _has_yakuhai_route(
    state: dict,
    after_counts: list[int],
    triplet_meld_bases: list[int],
) -> bool:
    if any(_is_yakuhai(state, base_index) for base_index in triplet_meld_bases):
        return True
    for base_index in range(27, 34):
        if _is_yakuhai(state, base_index) and after_counts[base_index] >= 2:
            return True
    return False


def _has_tanyao_route(after_counts: list[int], meld_tile_bases: list[int]) -> bool:
    for base_index, count in enumerate(after_counts):
        if count > 0 and not _is_simple(base_index):
            return False
    return all(_is_simple(base_index) for base_index in meld_tile_bases)


def _has_flush_route(after_counts: list[int], meld_tile_bases: list[int]) -> bool:
    suits = {
        _tile_suit_index(base_index)
        for base_index in _base_indices_from_counts(after_counts) + list(meld_tile_bases)
        if base_index < 27
    }
    return len(suits) == 1


def _has_toitoi_route(
    kind: str,
    state: dict,
    after_counts: list[int],
    triplet_meld_bases: list[int],
) -> bool:
    if kind == "chi" or any(str(meld.get("kind")) == "chi" for meld in state.get("own_melds", [])):
        return False
    if not triplet_meld_bases:
        return False
    pair_like = sum(1 for count in after_counts if count >= 2)
    concealed_triplets = sum(1 for count in after_counts if count >= 3)
    return len(triplet_meld_bases) + concealed_triplets >= 2 or pair_like >= 3


def _existing_meld_tile_bases(melds: list[dict]) -> list[int]:
    bases: list[int] = []
    for meld in melds:
        bases.extend(tile_to_base_index(tile) for tile in _meld_tiles(meld))
    return bases


def _triplet_meld_bases(melds: list[dict]) -> list[int]:
    bases: list[int] = []
    for meld in melds:
        kind = str(meld.get("kind", ""))
        if kind not in {"pon", "daiminkan", "ankan", "kakan"}:
            continue
        tiles = _meld_tiles(meld)
        if not tiles:
            continue
        bases.append(tile_to_base_index(tiles[0]))
    return bases


def _meld_tiles(meld: dict) -> list[str]:
    tiles: list[str] = []
    called_tile = meld.get("called_tile")
    if called_tile is not None:
        tiles.append(str(called_tile))
    tiles.extend(str(tile) for tile in meld.get("consumed_tiles", []))
    return tiles


def _meld_visible_tiles(meld: dict) -> list[str]:
    kind = str(meld.get("kind", ""))
    if kind == "ankan":
        return _meld_tiles(meld)
    return [str(tile) for tile in meld.get("consumed_tiles", [])]


def _visible_melds(state: dict):
    all_melds = state.get("all_melds")
    if all_melds is None:
        yield from state.get("own_melds", [])
        return
    for player_melds in all_melds:
        yield from player_melds


def _base_indices_from_counts(counts: list[int]) -> list[int]:
    return [base_index for base_index, count in enumerate(counts) for _ in range(int(count))]


def _is_simple(base_index: int) -> bool:
    return base_index < 27 and base_index % 9 not in {0, 8}


def _tile_suit_index(base_index: int) -> int:
    return base_index // 9


def _has_open_melds(state: dict) -> bool:
    return any(str(meld.get("kind")) in {"chi", "pon", "daiminkan", "kakan"} for meld in state.get("own_melds", []))


def _chiitoitsu_shanten_from_counts(counts: tuple[int, ...]) -> int:
    pair_count = sum(1 for count in counts if count >= 2)
    unique_count = sum(1 for count in counts if count > 0)
    return 6 - pair_count + max(0, 7 - unique_count)


def _kokushi_shanten_from_counts(counts: tuple[int, ...]) -> int:
    terminals = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    unique_terminals = sum(1 for index in terminals if counts[index] > 0)
    has_terminal_pair = any(counts[index] >= 2 for index in terminals)
    return 13 - unique_terminals - int(has_terminal_pair)


def _scale_delta(value: int | float, denominator: float) -> float:
    """执行与函数名对应的业务逻辑。
    
    参数:
        value (int | float): 输入值。
        denominator (float): 函数执行所需的输入参数。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    return max(-1.0, min(1.0, float(value) / denominator))


def _action_consumes_dora(consumed_tiles: list[str], dora_bases: set[int]) -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        consumed_tiles (list[str]): 副露消耗牌列表。
        dora_bases (set[int]): 相关数据列表。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    return any(tile_to_base_index(tile) in dora_bases or tile.startswith("0") for tile in consumed_tiles)


def _consumes_pair(counts: list[int], consumed_tiles: list[str]) -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (list[int]): 三十四种牌计数。
        consumed_tiles (list[str]): 副露消耗牌列表。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    consumed_bases = {tile_to_base_index(tile) for tile in consumed_tiles}
    return any(counts[base_index] >= 2 for base_index in consumed_bases)


def _breaks_taatsu(counts: list[int], consumed_tiles: list[str]) -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (list[int]): 三十四种牌计数。
        consumed_tiles (list[str]): 副露消耗牌列表。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    consumed_bases = {tile_to_base_index(tile) for tile in consumed_tiles}
    for base_index in consumed_bases:
        if base_index >= 27:
            continue
        suit_offset = (base_index // 9) * 9
        for neighbor in (base_index - 2, base_index - 1, base_index + 1, base_index + 2):
            if suit_offset <= neighbor < suit_offset + 9 and counts[neighbor] > 0:
                return True
    return False


def _is_yakuhai(state: dict, base_index: int) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        state (dict): 游戏状态或样本状态。
        base_index (int): 三十四种牌基础索引。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if base_index < 27:
        return False
    honor = base_index - 27
    if 4 <= honor <= 6:
        return True
    seat_wind = int(state.get("seat_wind", -1))
    dealer = int(state.get("dealer", -1))
    actor = int(state.get("actor", -1))
    round_wind = _require_round_wind(state)
    if honor == seat_wind:
        return True
    if honor == round_wind:
        return True
    return honor == ((actor - dealer) % 4)


def _require_round_wind(state: dict) -> int:
    if "round_wind" not in state:
        raise KeyError("response action features require state.round_wind")
    return int(state["round_wind"])


def _chi_shape(action: dict) -> str:
    """执行与函数名对应的业务逻辑。
    
    参数:
        action (dict): 动作字典或动作对象。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    if str(action.get("kind", "")) != "chi":
        return "none"
    tiles = [action.get("tile"), *action.get("consumed_tiles", [])]
    if any(tile is None for tile in tiles):
        return "none"
    base_indices = sorted(tile_to_base_index(str(tile)) for tile in tiles)
    if len(base_indices) != 3 or base_indices[-1] - base_indices[0] != 2:
        return "none"
    if base_indices[0] >= 27:
        return "none"

    target_base = tile_to_base_index(str(action["tile"]))
    start_rank = base_indices[0] % 9
    target_rank = target_base % 9
    if target_rank == start_rank + 1:
        return "closed_wait"
    if start_rank == 0 and target_rank == 2:
        return "edge_wait"
    if start_rank == 6 and target_rank == 6:
        return "edge_wait"
    return "ryanmen"


def _shape_summary(counts: list[int]) -> dict[str, int]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        counts (list[int]): 三十四种牌计数。
    
    返回:
        dict[str, int]: 字典结果。
    
    异常:
        无。
    """
    complete_sets = 0
    pairs = 0
    ryanmen = 0
    closed_waits = 0
    isolated = 0

    for index, count in enumerate(counts):
        complete_sets += count // 3
        if count >= 2:
            pairs += 1
        if count <= 0:
            continue
        if _is_isolated(counts, index):
            isolated += 1

    for suit_offset in (0, 9, 18):
        for rank in range(8):
            left = suit_offset + rank
            right = left + 1
            if counts[left] <= 0 or counts[right] <= 0:
                continue
            if 1 <= rank <= 5:
                ryanmen += 1
            else:
                closed_waits += 1
        for rank in range(7):
            left = suit_offset + rank
            right = left + 2
            if counts[left] > 0 and counts[right] > 0:
                closed_waits += 1

    return {
        "complete_sets": complete_sets,
        "pairs": pairs,
        "ryanmen": ryanmen,
        "closed_waits": closed_waits,
        "isolated": isolated,
    }


def _is_isolated(counts: list[int], base_index: int) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        counts (list[int]): 三十四种牌计数。
        base_index (int): 三十四种牌基础索引。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if counts[base_index] >= 2:
        return False
    if base_index >= 27:
        return True
    suit_offset = (base_index // 9) * 9
    for neighbor in range(max(suit_offset, base_index - 2), min(suit_offset + 8, base_index + 2) + 1):
        if neighbor != base_index and counts[neighbor] > 0:
            return False
    return True


def _scale_shanten(shanten: int) -> float:
    """执行与函数名对应的业务逻辑。
    
    参数:
        shanten (int): 函数执行所需的输入参数。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    return max(-1.0, min(1.0, shanten / 6.0))


def _is_terminal(base_index: int) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        base_index (int): 三十四种牌基础索引。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    return base_index < 27 and base_index % 9 in {0, 8}


def _is_honor(base_index: int) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        base_index (int): 三十四种牌基础索引。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    return base_index >= 27


def _is_genbutsu(state: dict, base_index: int) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        state (dict): 游戏状态或样本状态。
        base_index (int): 三十四种牌基础索引。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    actor = int(state.get("actor", 0))
    for tile in _iter_other_discards(state, actor):
        if tile_to_base_index(str(tile)) == base_index:
            return True
    return False


def _visible_count(state: dict, base_index: int, include_hand: bool = True) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        state (dict): 游戏状态或样本状态。
        base_index (int): 三十四种牌基础索引。
        include_hand (bool): 函数执行所需的输入参数。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    return _visible_count_with_called_meld_tiles(state, base_index, include_hand)


def _discard_timing_context(state: dict) -> dict[str, object]:
    actor = int(state.get("actor", 0))
    all_discards = state.get("all_discards", [])
    player_discards = [
        list(all_discards[player_index]) if player_index < len(all_discards) else []
        for player_index in range(4)
    ]
    other_players = [player_index for player_index in range(4) if player_index != actor]
    per_player = {
        player_index: _player_discard_timing_summary(player_discards[player_index])
        for player_index in range(4)
    }
    other_summary = _combined_discard_timing_summary(
        [player_discards[player_index] for player_index in other_players]
    )
    own_summary = _combined_discard_timing_summary([player_discards[actor]])
    relative_players = [((actor + offset) % 4) for offset in (1, 2, 3)]
    return {
        "actor": actor,
        "own": own_summary,
        "other": other_summary,
        "per_player": per_player,
        "relative_players": relative_players,
    }


def _player_discard_timing_summary(discards: list[str]) -> dict[int, dict[str, float | int]]:
    summary: dict[int, dict[str, float | int]] = {}
    for turn_index, tile in enumerate(discards, start=1):
        base_index = tile_to_base_index(str(tile))
        tile_summary = summary.setdefault(
            base_index,
            {
                "count": 0,
                "first_turn": turn_index,
                "last_turn": turn_index,
                "recent_count": 0,
                "early_count": 0,
            },
        )
        tile_summary["count"] = int(tile_summary["count"]) + 1
        tile_summary["first_turn"] = min(int(tile_summary["first_turn"]), turn_index)
        tile_summary["last_turn"] = max(int(tile_summary["last_turn"]), turn_index)
        if turn_index <= TURN_RIVER_EARLY_WINDOW:
            tile_summary["early_count"] = int(tile_summary["early_count"]) + 1

    recent_start = max(1, len(discards) - TURN_RIVER_RECENT_WINDOW + 1)
    for turn_index, tile in enumerate(discards, start=1):
        if turn_index < recent_start:
            continue
        base_index = tile_to_base_index(str(tile))
        summary[base_index]["recent_count"] = int(summary[base_index]["recent_count"]) + 1
    return summary


def _combined_discard_timing_summary(discard_rows: list[list[str]]) -> dict[int, dict[str, float | int]]:
    combined: dict[int, dict[str, float | int]] = {}
    for discards in discard_rows:
        player_summary = _player_discard_timing_summary(discards)
        for base_index, tile_summary in player_summary.items():
            combined_summary = combined.setdefault(
                base_index,
                {
                    "count": 0,
                    "first_turn": int(tile_summary["first_turn"]),
                    "last_turn": int(tile_summary["last_turn"]),
                    "recent_count": 0,
                    "early_count": 0,
                    "players": 0,
                    "recent_players": 0,
                    "early_players": 0,
                },
            )
            combined_summary["count"] = int(combined_summary["count"]) + int(tile_summary["count"])
            combined_summary["first_turn"] = min(
                int(combined_summary["first_turn"]),
                int(tile_summary["first_turn"]),
            )
            combined_summary["last_turn"] = max(
                int(combined_summary["last_turn"]),
                int(tile_summary["last_turn"]),
            )
            combined_summary["recent_count"] = int(combined_summary["recent_count"]) + int(
                tile_summary["recent_count"]
            )
            combined_summary["early_count"] = int(combined_summary["early_count"]) + int(
                tile_summary["early_count"]
            )
            combined_summary["players"] = int(combined_summary["players"]) + 1
            if int(tile_summary["recent_count"]) > 0:
                combined_summary["recent_players"] = int(combined_summary["recent_players"]) + 1
            if int(tile_summary["early_count"]) > 0:
                combined_summary["early_players"] = int(combined_summary["early_players"]) + 1
    return combined


def _fill_discard_timing_features(
    features: list[float],
    timing_context: dict[str, object],
    base_index: int,
) -> None:
    own_summary = timing_context["own"]
    other_summary = timing_context["other"]
    per_player = timing_context["per_player"]
    relative_players = timing_context["relative_players"]
    assert isinstance(own_summary, dict)
    assert isinstance(other_summary, dict)
    assert isinstance(per_player, dict)
    assert isinstance(relative_players, list)

    own_tile = own_summary.get(base_index, {})
    other_tile = other_summary.get(base_index, {})
    features[25] = _scaled_summary_count(own_tile, "count", 4.0)
    features[26] = _scaled_summary_count(other_tile, "count", 4.0)
    features[27] = _scaled_summary_count(other_tile, "players", 3.0)
    features[28] = _scaled_summary_count(other_tile, "recent_players", 3.0)
    features[29] = _scaled_summary_count(other_tile, "early_players", 3.0)
    features[30] = _scaled_summary_turn(other_tile, "first_turn")
    features[31] = _scaled_summary_turn(other_tile, "last_turn")
    features[32] = 1.0 if int(other_tile.get("players", 0)) >= 3 else 0.0
    features[33] = 1.0 if int(other_tile.get("players", 0)) >= 2 else 0.0
    features[34] = 1.0 if int(other_tile.get("recent_players", 0)) > 0 else 0.0

    for offset, player_index in enumerate(relative_players[:3]):
        player_tile = per_player.get(player_index, {}).get(base_index, {})
        features[35 + offset] = 1.0 if int(player_tile.get("count", 0)) > 0 else 0.0
        features[38 + offset] = 1.0 if int(player_tile.get("recent_count", 0)) > 0 else 0.0

    actor = int(timing_context["actor"])
    for player_index in range(4):
        if player_index == actor:
            continue
        player_tile = per_player.get(player_index, {}).get(base_index, {})
        features[41 + player_index] = 1.0 if int(player_tile.get("count", 0)) > 0 else 0.0


def _scaled_summary_count(summary: dict, key: str, denominator: float) -> float:
    return min(1.0, float(summary.get(key, 0)) / denominator)


def _scaled_summary_turn(summary: dict, key: str) -> float:
    turn = int(summary.get(key, 0))
    if turn <= 0:
        return 0.0
    return min(1.0, float(turn) / TURN_RIVER_TURN_SCALE)


def _is_suji_safe(state: dict, base_index: int) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        state (dict): 游戏状态或样本状态。
        base_index (int): 三十四种牌基础索引。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if base_index >= 27:
        return False
    rank = base_index % 9
    if rank not in {0, 3, 5, 8}:
        return False
    actor = int(state.get("actor", 0))
    seen_bases = {tile_to_base_index(str(tile)) for tile in _iter_other_discards(state, actor)}
    if rank == 0:
        return base_index + 3 in seen_bases
    if rank == 3:
        return base_index - 3 in seen_bases or base_index + 3 in seen_bases
    if rank == 5:
        return base_index - 3 in seen_bases or base_index + 3 in seen_bases
    return base_index - 3 in seen_bases


def _is_one_chance(state: dict, after_counts: list[int], base_index: int) -> bool:
    """判断与函数名对应的条件是否成立。
    
    参数:
        state (dict): 游戏状态或样本状态。
        after_counts (list[int]): 相关数据列表。
        base_index (int): 三十四种牌基础索引。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if base_index >= 27:
        return False
    neighbor_bases = []
    suit_offset = (base_index // 9) * 9
    for neighbor in (base_index - 1, base_index + 1):
        if suit_offset <= neighbor < suit_offset + 9:
            neighbor_bases.append(neighbor)
    return any(
        _visible_count(state, neighbor, include_hand=False) + after_counts[neighbor] >= 3
        for neighbor in neighbor_bases
    )


def _iter_other_discards(state: dict, actor: int):
    """执行与函数名对应的业务逻辑。
    
    参数:
        state (dict): 游戏状态或样本状态。
        actor (int): 玩家座位索引。
    
    返回:
        Iterator[Any]: 逐条生成的迭代结果。
    
    异常:
        无。
    """
    for player_index, player_discards in enumerate(state.get("all_discards", [])):
        if player_index == actor:
            continue
        yield from player_discards


def _dora_base_indices(state: dict) -> set[int]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        state (dict): 游戏状态或样本状态。
    
    返回:
        set[int]: 集合结果。
    
    异常:
        无。
    """
    configured = state.get("dora_base_indices")
    if configured is not None:
        return {int(base_index) for base_index in configured}
    return {
        tile_to_base_index(dora_from_indicator(str(indicator)))
        for indicator in state.get("dora_indicators", [])
    }


def encode_turn_action(action: dict, state: dict | None = None, context: dict | None = None) -> list[float]:
    if state is None:
        return _encode_turn_action(action, state, context)
    actor = int(state.get("actor", 0))
    canonical = relative_state(state)
    return _encode_turn_action(relative_action(action, actor), canonical,
                               build_turn_discard_feature_context(canonical))


def encode_response_action(action: dict, state: dict | None = None, context: dict | None = None) -> list[float]:
    if state is None:
        return _encode_response_action(action, state, context)
    actor = int(state.get("actor", 0))
    canonical = relative_state(state)
    return _encode_response_action(relative_action(action, actor), canonical,
                                   build_response_call_feature_context(canonical))
