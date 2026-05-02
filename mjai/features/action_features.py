from __future__ import annotations

from mjai.features.state_features import EXACT_TILE_INDEX, EXACT_TILE_VOCAB
from mjai.rules.hand_checker import effective_tile_base_indices_from_counts, shanten_number_from_counts
from mjai.rules.tiles import AKA_BASE_INDICES, dora_from_indicator, tile_to_base_index


TURN_ACTION_KINDS = ("discard", "riichi_discard", "tsumo", "ankan", "kakan", "kyuushu_kyuuhai", "unknown")
TURN_DISCARD_FEATURE_DIM = 25
TURN_ACTION_DIM = len(TURN_ACTION_KINDS) + len(EXACT_TILE_VOCAB) + TURN_DISCARD_FEATURE_DIM
RESPONSE_ACTION_KINDS = ("pass", "chi", "pon", "daiminkan", "ron", "chankan", "unknown")
RESPONSE_ACTION_DIM = len(RESPONSE_ACTION_KINDS) + len(EXACT_TILE_VOCAB) + 4


def encode_turn_action(
    action: dict,
    state: dict | None = None,
    context: dict | None = None,
) -> list[float]:
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
    """Encode all turn candidates for one sample, sharing shanten context once."""
    context = build_turn_discard_feature_context(state)
    return [encode_turn_action(action, state, context) for action in actions]


def encode_response_action(action: dict) -> list[float]:
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
    return vector


def build_turn_discard_feature_context(state: dict | None) -> dict | None:
    if state is None:
        return None
    counts = _hand_counts_from_state(state)
    return {
        "counts": counts,
        "current_shanten": shanten_number_from_counts(tuple(counts)),
        "dora_bases": _dora_base_indices(state),
    }


def encode_turn_discard_features(
    action: dict,
    state: dict | None,
    context: dict | None = None,
) -> list[float]:
    """Encode candidate-specific hand-shape features for discard-like actions."""
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

    after_counts = list(counts)
    after_counts[base_index] -= 1
    after_counts_tuple = tuple(after_counts)
    current_shanten = int(context["current_shanten"])
    after_shanten = shanten_number_from_counts(after_counts_tuple)
    effective_tiles = effective_tile_base_indices_from_counts(after_counts_tuple)
    effective_tile_copies = sum(max(0, 4 - after_counts[index]) for index in effective_tiles)
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
    dora_bases = context["dora_bases"]
    features[14] = min(1.0, (1.0 if base_index in AKA_BASE_INDICES else 0.0) + float(base_index in dora_bases))
    return features


def _hand_counts_from_state(state: dict) -> list[int]:
    counts = [0] * 34
    for tile, count in state.get("hand", {}).items():
        counts[tile_to_base_index(str(tile))] += int(count)
    return counts


def _shape_summary(counts: list[int]) -> dict[str, int]:
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
    return max(-1.0, min(1.0, shanten / 6.0))


def _is_terminal(base_index: int) -> bool:
    return base_index < 27 and base_index % 9 in {0, 8}


def _is_honor(base_index: int) -> bool:
    return base_index >= 27


def _is_genbutsu(state: dict, base_index: int) -> bool:
    actor = int(state.get("actor", 0))
    for tile in _iter_other_discards(state, actor):
        if tile_to_base_index(str(tile)) == base_index:
            return True
    return False


def _visible_count(state: dict, base_index: int, include_hand: bool = True) -> int:
    visible = 0
    if include_hand:
        for tile, count in state.get("hand", {}).items():
            if tile_to_base_index(str(tile)) == base_index:
                visible += int(count)
    for player_discards in state.get("all_discards", []):
        for tile in player_discards:
            if tile_to_base_index(str(tile)) == base_index:
                visible += 1
    for meld in state.get("own_melds", []):
        for tile in [meld.get("called_tile"), *meld.get("consumed_tiles", [])]:
            if tile is not None and tile_to_base_index(str(tile)) == base_index:
                visible += 1
    return visible


def _is_suji_safe(state: dict, base_index: int) -> bool:
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
    for player_index, player_discards in enumerate(state.get("all_discards", [])):
        if player_index == actor:
            continue
        yield from player_discards


def _dora_base_indices(state: dict) -> set[int]:
    configured = state.get("dora_base_indices")
    if configured is not None:
        return {int(base_index) for base_index in configured}
    return {
        tile_to_base_index(dora_from_indicator(str(indicator)))
        for indicator in state.get("dora_indicators", [])
    }
