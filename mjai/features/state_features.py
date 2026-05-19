from __future__ import annotations

def _build_exact_tile_vocab() -> list[str]:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        无。
    
    返回:
        list[str]: 列表结果。
    
    异常:
        无。
    """
    vocab: list[str] = []
    for suit in ("m", "p", "s"):
        vocab.extend([f"{rank}{suit}" for rank in range(1, 6)])
        vocab.append(f"0{suit}")
        vocab.extend([f"{rank}{suit}" for rank in range(6, 10)])
    vocab.extend([f"{rank}z" for rank in range(1, 8)])
    return vocab


EXACT_TILE_VOCAB = _build_exact_tile_vocab()
EXACT_TILE_INDEX = {tile: index for index, tile in enumerate(EXACT_TILE_VOCAB)}

# Exact-tile features distinguish red fives ("0m/0p/0s") from normal fives.
EXACT_TILE_DIM = len(EXACT_TILE_VOCAB)

# Optional categorical features reserve one extra slot for "missing / not present".
OPTIONAL_TILE_DIM = EXACT_TILE_DIM + 1
OPTIONAL_ACTOR_DIM = 5  # four seats plus one "missing" bucket
ACTOR_DIM = 4

# Keep this layout in sync with docs/model-inputs.md. These vectors are
# serialized into parquet as float32 bytes, so dimension or order changes require
# rebuilding data/silver_samples and manifests.
#
# encode_turn_state layout:
# hand counts | 4 players' discard counts | last discard tile | last discard actor |
# current actor | riichi flag | furiten flags | own meld kind counts |
# 4-player riichi flags | opponent riichi turns | discard-history tokens |
# round context
HAND_COUNTS_DIM = EXACT_TILE_DIM
ALL_DISCARD_COUNTS_DIM = 4 * EXACT_TILE_DIM
LAST_DISCARD_TILE_DIM = OPTIONAL_TILE_DIM
LAST_DISCARD_ACTOR_DIM = OPTIONAL_ACTOR_DIM
CURRENT_ACTOR_DIM = ACTOR_DIM
RIICHI_FLAG_DIM = 1
FURITEN_FLAGS_DIM = 2
OWN_MELD_COUNTS_DIM = 5
RIICHI_BY_PLAYER_DIM = ACTOR_DIM
RIICHI_TURN_BY_PLAYER_DIM = ACTOR_DIM
DISCARD_HISTORY_MAX_LEN = 192
DISCARD_HISTORY_TOKEN_DIM = ACTOR_DIM + EXACT_TILE_DIM
DISCARD_HISTORY_DIM = DISCARD_HISTORY_MAX_LEN * DISCARD_HISTORY_TOKEN_DIM
DEALER_DIM = ACTOR_DIM
SEAT_WIND_DIM = ACTOR_DIM
DORA_COUNTS_DIM = EXACT_TILE_DIM
SCORE_DIM = 4
ROUND_NUMERIC_DIM = 4
ROUND_CONTEXT_DIM = DEALER_DIM + SEAT_WIND_DIM + DORA_COUNTS_DIM + SCORE_DIM + ROUND_NUMERIC_DIM

TURN_STATE_DIM = (
    HAND_COUNTS_DIM
    + ALL_DISCARD_COUNTS_DIM
    + LAST_DISCARD_TILE_DIM
    + LAST_DISCARD_ACTOR_DIM
    + CURRENT_ACTOR_DIM
    + RIICHI_FLAG_DIM
    + FURITEN_FLAGS_DIM
    + OWN_MELD_COUNTS_DIM
    + RIICHI_BY_PLAYER_DIM
    + RIICHI_TURN_BY_PLAYER_DIM
    + DISCARD_HISTORY_DIM
    + ROUND_CONTEXT_DIM
)

# encode_response_state appends the pending call/ron target to the turn-state
# features so response-head models can identify who discarded which tile.
RESPONSE_TARGET_SOURCE_DIM = OPTIONAL_ACTOR_DIM
RESPONSE_TARGET_TILE_DIM = OPTIONAL_TILE_DIM
RESPONSE_STATE_DIM = TURN_STATE_DIM + RESPONSE_TARGET_SOURCE_DIM + RESPONSE_TARGET_TILE_DIM


def encode_turn_state(state: dict) -> list[float]:
    """将自摸巡状态字典编码为定长特征向量。
    
    参数:
        state (dict): 游戏状态或样本状态。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    features: list[float] = []
    # 37 exact-tile hand counts. Red fives use their own exact-tile slots.
    features.extend(_encode_exact_tile_counter(state.get("hand", {})))

    all_discards = state.get("all_discards", [])
    player_discards_rows = [
        list(all_discards[player_index]) if player_index < len(all_discards) else []
        for player_index in range(4)
    ]
    # Four discard-count blocks, one per player. Full river order is encoded
    # separately below in the discard-history token block.
    for player_discards in player_discards_rows:
        discard_counts: dict[str, int] = {}
        for tile in player_discards:
            discard_counts[tile] = discard_counts.get(tile, 0) + 1
        features.extend(_encode_exact_tile_counter(discard_counts))

    # Optional last-discard context is useful for turn decisions after calls and
    # keeps the state vector shape shared with response state.
    last_discard_tile = state.get("last_discard_tile")
    features.extend(_encode_optional_tile(last_discard_tile))
    features.extend(_encode_optional_actor(state.get("last_discard_actor")))
    features.extend(_encode_actor_onehot(int(state.get("actor", 0))))
    features.append(1.0 if state.get("riichi_declared", False) else 0.0)
    features.append(1.0 if state.get("temporary_furiten", False) else 0.0)
    features.append(1.0 if state.get("riichi_furiten", False) else 0.0)
    features.extend(_encode_own_meld_counts(state.get("own_melds", [])))
    features.extend(_encode_riichi_by_player(state))
    features.extend(_encode_opponent_riichi_turns(state))
    features.extend(_encode_discard_history(state.get("discard_history", [])))
    features.extend(_encode_actor_onehot(int(state.get("dealer", 0))))
    features.extend(_encode_actor_onehot(int(state.get("seat_wind", 0))))
    features.extend(_encode_dora_counts(state))
    features.extend(_encode_scores(state.get("scores", [])))
    features.extend(_encode_round_numeric(state))
    return features


def encode_response_state(state: dict) -> list[float]:
    """将响应状态字典编码为定长特征向量。
    
    参数:
        state (dict): 游戏状态或样本状态。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    features = encode_turn_state(state)
    # Response adds the pending discard/kakan target to the common turn-state
    # prefix so pass/chi/pon/ron candidates are scored against the same target.
    response_target = state.get("response_target", {})
    features.extend(_encode_optional_actor(response_target.get("source")))
    features.extend(_encode_optional_tile(response_target.get("tile")))
    return features


def _encode_exact_tile_counter(counter: dict[str, int]) -> list[float]:
    """将输入数据编码为模型可用特征。
    
    参数:
        counter (dict[str, int]): 函数执行所需的输入参数。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    vector = [0.0] * len(EXACT_TILE_VOCAB)
    for tile, count in counter.items():
        index = EXACT_TILE_INDEX.get(tile)
        if index is not None:
            vector[index] = float(count)
    return vector


def _encode_optional_tile(tile: str | None) -> list[float]:
    """将输入数据编码为模型可用特征。
    
    参数:
        tile (str | None): 精确牌字符串。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    vector = [0.0] * (len(EXACT_TILE_VOCAB) + 1)
    if tile is None:
        vector[-1] = 1.0
        return vector
    vector[EXACT_TILE_INDEX[tile]] = 1.0
    return vector


def _encode_optional_actor(actor: int | None) -> list[float]:
    """将输入数据编码为模型可用特征。
    
    参数:
        actor (int | None): 玩家座位索引。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    vector = [0.0] * 5
    if actor is None:
        vector[-1] = 1.0
        return vector
    vector[int(actor)] = 1.0
    return vector


def _encode_actor_onehot(actor: int) -> list[float]:
    """将输入数据编码为模型可用特征。
    
    参数:
        actor (int): 玩家座位索引。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    vector = [0.0] * 4
    vector[actor] = 1.0
    return vector


def _encode_own_meld_counts(melds: list[dict]) -> list[float]:
    """将输入数据编码为模型可用特征。
    
    参数:
        melds (list[dict]): 相关数据列表。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    kinds = ["chi", "pon", "daiminkan", "ankan", "kakan"]
    counts = {kind: 0.0 for kind in kinds}
    for meld in melds:
        kind = meld.get("kind")
        if kind in counts:
            counts[kind] += 1.0
    return [counts[kind] for kind in kinds]


def _encode_riichi_by_player(state: dict) -> list[float]:
    flags = _normalize_fixed_list(state.get("riichi_declared_by_player"), False)
    if state.get("riichi_declared_by_player") is None:
        actor = int(state.get("actor", 0))
        flags[actor] = bool(state.get("riichi_declared", False))
    return [1.0 if bool(flag) else 0.0 for flag in flags]


def _encode_opponent_riichi_turns(state: dict) -> list[float]:
    actor = int(state.get("actor", 0))
    turns = _normalize_fixed_list(state.get("riichi_turn_by_player"), None)
    encoded: list[float] = []
    for player_index, turn in enumerate(turns):
        if player_index == actor or turn is None:
            encoded.append(0.0)
            continue
        encoded.append(min(1.0, float(turn) / 18.0))
    return encoded


def _encode_discard_history(history: list[dict]) -> list[float]:
    vector = [0.0] * DISCARD_HISTORY_DIM
    for token_index, item in enumerate(history[:DISCARD_HISTORY_MAX_LEN]):
        parsed = _discard_history_item(item)
        if parsed is None:
            continue
        actor, tile = parsed
        tile_index = EXACT_TILE_INDEX.get(tile)
        if actor < 0 or actor >= ACTOR_DIM or tile_index is None:
            continue
        offset = token_index * DISCARD_HISTORY_TOKEN_DIM
        vector[offset + actor] = 1.0
        vector[offset + ACTOR_DIM + tile_index] = 1.0
    return vector


def _discard_history_item(item: object) -> tuple[int, str] | None:
    if isinstance(item, dict):
        actor = item.get("actor")
        tile = item.get("tile")
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        actor = item[0]
        tile = item[1]
    else:
        return None
    if actor is None or tile is None:
        return None
    return int(actor), str(tile)


def _normalize_fixed_list(value: object, default: object) -> list[object]:
    if not isinstance(value, (list, tuple)):
        return [default] * 4
    rows = list(value[:4])
    while len(rows) < 4:
        rows.append(default)
    return rows


def _encode_dora_counts(state: dict) -> list[float]:
    """将输入数据编码为模型可用特征。
    
    参数:
        state (dict): 游戏状态或样本状态。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    counts = [0.0] * len(EXACT_TILE_VOCAB)
    for base_index in state.get("dora_base_indices", []):
        tile = _base_index_to_feature_tile(int(base_index))
        counts[EXACT_TILE_INDEX[tile]] += 1.0
    return counts


def _encode_scores(scores: list[int]) -> list[float]:
    """将输入数据编码为模型可用特征。
    
    参数:
        scores (list[int]): 相关数据列表。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    encoded = [0.0] * 4
    for index, score in enumerate(scores[:4]):
        encoded[index] = float(score) / 1000.0
    return encoded


def _encode_round_numeric(state: dict) -> list[float]:
    """将输入数据编码为模型可用特征。
    
    参数:
        state (dict): 游戏状态或样本状态。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        无。
    """
    return [
        min(1.0, float(state.get("turn_number", 0)) / 18.0),
        min(1.0, float(state.get("remaining_draws_estimate", 0)) / 18.0),
        min(1.0, float(state.get("honba", 0)) / 8.0),
        min(1.0, float(state.get("riichi_sticks", 0)) / 8.0),
    ]


def _base_index_to_feature_tile(base_index: int) -> str:
    """执行与函数名对应的业务逻辑。
    
    参数:
        base_index (int): 三十四种牌基础索引。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    if base_index < 9:
        return f"{base_index + 1}m"
    if base_index < 18:
        return f"{base_index - 8}p"
    if base_index < 27:
        return f"{base_index - 17}s"
    return f"{base_index - 26}z"
