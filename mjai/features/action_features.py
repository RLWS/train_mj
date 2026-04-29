from __future__ import annotations

from mjai.features.state_features import EXACT_TILE_INDEX, EXACT_TILE_VOCAB


TURN_ACTION_KINDS = ("discard", "riichi_discard", "tsumo", "ankan", "kakan", "kyuushu_kyuuhai", "unknown")
TURN_ACTION_DIM = len(TURN_ACTION_KINDS) + len(EXACT_TILE_VOCAB)
RESPONSE_ACTION_KINDS = ("pass", "chi", "pon", "daiminkan", "ron", "chankan", "unknown")
RESPONSE_ACTION_DIM = len(RESPONSE_ACTION_KINDS) + len(EXACT_TILE_VOCAB) + 4


def encode_turn_action(action: dict) -> list[float]:
    vector = [0.0] * TURN_ACTION_DIM

    kind = str(action.get("kind", "unknown"))
    try:
        vector[TURN_ACTION_KINDS.index(kind)] = 1.0
    except ValueError:
        vector[TURN_ACTION_KINDS.index("unknown")] = 1.0

    tile = action.get("tile")
    if tile is not None:
        vector[len(TURN_ACTION_KINDS) + EXACT_TILE_INDEX[str(tile)]] = 1.0
    return vector


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
