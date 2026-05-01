from __future__ import annotations

from mjai.features.state_features import (
    ACTOR_DIM,
    ALL_DISCARD_COUNTS_DIM,
    HAND_COUNTS_DIM,
    LAST_DISCARD_ACTOR_DIM,
    LAST_DISCARD_TILE_DIM,
    RESPONSE_STATE_DIM,
    RIICHI_FLAG_DIM,
    TURN_STATE_DIM,
    encode_response_state,
    encode_turn_state,
)
from mjai.features.action_features import (
    RESPONSE_ACTION_KINDS,
    TURN_ACTION_KINDS,
    TURN_DISCARD_FEATURE_DIM,
    encode_response_action,
    encode_turn_action,
)


def test_encode_turn_action_supports_tsumo():
    vector = encode_turn_action({"kind": "tsumo"})

    assert len(vector) > len(TURN_ACTION_KINDS) + TURN_DISCARD_FEATURE_DIM
    assert vector[TURN_ACTION_KINDS.index("tsumo")] == 1.0


def test_encode_turn_action_supports_ankan_and_kakan():
    ankan_vector = encode_turn_action({"kind": "ankan", "tile": "1p"})
    kakan_vector = encode_turn_action({"kind": "kakan", "tile": "5m"})

    assert ankan_vector[TURN_ACTION_KINDS.index("ankan")] == 1.0
    assert kakan_vector[TURN_ACTION_KINDS.index("kakan")] == 1.0


def test_encode_turn_action_supports_kyuushu_kyuuhai():
    vector = encode_turn_action({"kind": "kyuushu_kyuuhai"})

    assert vector[TURN_ACTION_KINDS.index("kyuushu_kyuuhai")] == 1.0


def test_encode_response_action_supports_chankan():
    vector = encode_response_action({"kind": "chankan", "tile": "5m", "source": 1})

    assert vector[RESPONSE_ACTION_KINDS.index("chankan")] == 1.0


def test_encode_turn_state_includes_furiten_flags():
    state = {
        "actor": 1,
        "hand": {"1m": 1},
        "all_discards": [[], [], [], []],
        "riichi_declared": True,
        "temporary_furiten": True,
        "riichi_furiten": False,
        "own_melds": [],
        "last_discard_actor": None,
        "last_discard_tile": None,
    }

    vector = encode_turn_state(state)

    assert len(vector) == TURN_STATE_DIM
    furiten_offset = _furiten_offset()
    assert vector[furiten_offset - RIICHI_FLAG_DIM] == 1.0
    assert vector[furiten_offset] == 1.0
    assert vector[furiten_offset + 1] == 0.0


def test_encode_response_state_includes_furiten_flags():
    state = {
        "actor": 2,
        "hand": {"1m": 1},
        "all_discards": [[], [], [], []],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": True,
        "own_melds": [],
        "last_discard_actor": None,
        "last_discard_tile": None,
        "response_target": {"source": 0, "tile": "5z"},
    }

    vector = encode_response_state(state)

    assert len(vector) == RESPONSE_STATE_DIM
    furiten_offset = _furiten_offset()
    assert vector[furiten_offset] == 0.0
    assert vector[furiten_offset + 1] == 1.0


def test_encode_turn_action_adds_discard_candidate_features():
    state = {
        "actor": 0,
        "hand": {
            "1m": 1,
            "2m": 1,
            "3m": 1,
            "4m": 1,
            "5m": 1,
            "6m": 1,
            "7m": 1,
            "8m": 1,
            "9m": 1,
            "2p": 1,
            "3p": 1,
            "4p": 1,
            "5z": 2,
        },
        "all_discards": [["9m"], [], [], []],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "last_discard_actor": None,
        "last_discard_tile": None,
    }

    vector = encode_turn_action({"kind": "discard", "tile": "9m"}, state)

    assert len(vector) == len(TURN_ACTION_KINDS) + 37 + TURN_DISCARD_FEATURE_DIM
    assert vector[-TURN_DISCARD_FEATURE_DIM] == 1.0


def test_encode_turn_state_includes_round_context():
    state = {
        "actor": 0,
        "hand": {"1m": 1},
        "all_discards": [[], [], [], []],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "last_discard_actor": None,
        "last_discard_tile": None,
        "dealer": 2,
        "seat_wind": 2,
        "scores": [250, 260, 240, 250],
        "dora_base_indices": [1],
        "turn_number": 9,
        "remaining_draws_estimate": 9,
        "honba": 2,
        "riichi_sticks": 1,
    }

    vector = encode_turn_state(state)

    assert len(vector) == TURN_STATE_DIM
    assert vector[-4:] == [0.5, 0.5, 0.25, 0.125]


def _furiten_offset() -> int:
    return (
        HAND_COUNTS_DIM
        + ALL_DISCARD_COUNTS_DIM
        + LAST_DISCARD_TILE_DIM
        + LAST_DISCARD_ACTOR_DIM
        + ACTOR_DIM
        + RIICHI_FLAG_DIM
    )
