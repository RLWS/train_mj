from __future__ import annotations

def _build_exact_tile_vocab() -> list[str]:
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

# encode_turn_state layout:
# hand counts | 4 players' discard counts | last discard tile | last discard actor |
# current actor | riichi flag | furiten flags | own meld kind counts | round context
HAND_COUNTS_DIM = EXACT_TILE_DIM
ALL_DISCARD_COUNTS_DIM = 4 * EXACT_TILE_DIM
LAST_DISCARD_TILE_DIM = OPTIONAL_TILE_DIM
LAST_DISCARD_ACTOR_DIM = OPTIONAL_ACTOR_DIM
CURRENT_ACTOR_DIM = ACTOR_DIM
RIICHI_FLAG_DIM = 1
FURITEN_FLAGS_DIM = 2
OWN_MELD_COUNTS_DIM = 5
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
    + ROUND_CONTEXT_DIM
)

# encode_response_state appends the pending call/ron target to the turn-state
# features so response-head models can identify who discarded which tile.
RESPONSE_TARGET_SOURCE_DIM = OPTIONAL_ACTOR_DIM
RESPONSE_TARGET_TILE_DIM = OPTIONAL_TILE_DIM
RESPONSE_STATE_DIM = TURN_STATE_DIM + RESPONSE_TARGET_SOURCE_DIM + RESPONSE_TARGET_TILE_DIM


def encode_turn_state(state: dict) -> list[float]:
    features: list[float] = []
    features.extend(_encode_exact_tile_counter(state.get("hand", {})))

    all_discards = state.get("all_discards", [])
    for player_discards in all_discards[:4]:
        discard_counts: dict[str, int] = {}
        for tile in player_discards:
            discard_counts[tile] = discard_counts.get(tile, 0) + 1
        features.extend(_encode_exact_tile_counter(discard_counts))

    last_discard_tile = state.get("last_discard_tile")
    features.extend(_encode_optional_tile(last_discard_tile))
    features.extend(_encode_optional_actor(state.get("last_discard_actor")))
    features.extend(_encode_actor_onehot(int(state.get("actor", 0))))
    features.append(1.0 if state.get("riichi_declared", False) else 0.0)
    features.append(1.0 if state.get("temporary_furiten", False) else 0.0)
    features.append(1.0 if state.get("riichi_furiten", False) else 0.0)
    features.extend(_encode_own_meld_counts(state.get("own_melds", [])))
    features.extend(_encode_actor_onehot(int(state.get("dealer", 0))))
    features.extend(_encode_actor_onehot(int(state.get("seat_wind", 0))))
    features.extend(_encode_dora_counts(state))
    features.extend(_encode_scores(state.get("scores", [])))
    features.extend(_encode_round_numeric(state))
    return features


def encode_response_state(state: dict) -> list[float]:
    features = encode_turn_state(state)
    response_target = state.get("response_target", {})
    features.extend(_encode_optional_actor(response_target.get("source")))
    features.extend(_encode_optional_tile(response_target.get("tile")))
    return features


def _encode_exact_tile_counter(counter: dict[str, int]) -> list[float]:
    vector = [0.0] * len(EXACT_TILE_VOCAB)
    for tile, count in counter.items():
        index = EXACT_TILE_INDEX.get(tile)
        if index is not None:
            vector[index] = float(count)
    return vector


def _encode_optional_tile(tile: str | None) -> list[float]:
    vector = [0.0] * (len(EXACT_TILE_VOCAB) + 1)
    if tile is None:
        vector[-1] = 1.0
        return vector
    vector[EXACT_TILE_INDEX[tile]] = 1.0
    return vector


def _encode_optional_actor(actor: int | None) -> list[float]:
    vector = [0.0] * 5
    if actor is None:
        vector[-1] = 1.0
        return vector
    vector[int(actor)] = 1.0
    return vector


def _encode_actor_onehot(actor: int) -> list[float]:
    vector = [0.0] * 4
    vector[actor] = 1.0
    return vector


def _encode_own_meld_counts(melds: list[dict]) -> list[float]:
    kinds = ["chi", "pon", "daiminkan", "ankan", "kakan"]
    counts = {kind: 0.0 for kind in kinds}
    for meld in melds:
        kind = meld.get("kind")
        if kind in counts:
            counts[kind] += 1.0
    return [counts[kind] for kind in kinds]


def _encode_dora_counts(state: dict) -> list[float]:
    counts = [0.0] * len(EXACT_TILE_VOCAB)
    for base_index in state.get("dora_base_indices", []):
        tile = _base_index_to_feature_tile(int(base_index))
        counts[EXACT_TILE_INDEX[tile]] += 1.0
    return counts


def _encode_scores(scores: list[int]) -> list[float]:
    encoded = [0.0] * 4
    for index, score in enumerate(scores[:4]):
        encoded[index] = float(score) / 1000.0
    return encoded


def _encode_round_numeric(state: dict) -> list[float]:
    return [
        min(1.0, float(state.get("turn_number", 0)) / 18.0),
        min(1.0, float(state.get("remaining_draws_estimate", 0)) / 18.0),
        min(1.0, float(state.get("honba", 0)) / 8.0),
        min(1.0, float(state.get("riichi_sticks", 0)) / 8.0),
    ]


def _base_index_to_feature_tile(base_index: int) -> str:
    if base_index < 9:
        return f"{base_index + 1}m"
    if base_index < 18:
        return f"{base_index - 8}p"
    if base_index < 27:
        return f"{base_index - 17}s"
    return f"{base_index - 26}z"
