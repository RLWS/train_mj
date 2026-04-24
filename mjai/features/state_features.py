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
TURN_STATE_DIM = len(EXACT_TILE_VOCAB) + (4 * len(EXACT_TILE_VOCAB)) + (len(EXACT_TILE_VOCAB) + 1) + 5 + 4 + 1 + 5
RESPONSE_STATE_DIM = TURN_STATE_DIM + 5 + (len(EXACT_TILE_VOCAB) + 1)


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
    features.extend(_encode_own_meld_counts(state.get("own_melds", [])))
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
