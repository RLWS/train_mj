from __future__ import annotations

from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM, encode_response_state, encode_turn_state
from mjai.features.action_features import TURN_ACTION_KINDS, encode_turn_action


def test_encode_turn_action_supports_tsumo():
    vector = encode_turn_action({"kind": "tsumo"})

    assert len(vector) > len(TURN_ACTION_KINDS)
    assert vector[TURN_ACTION_KINDS.index("tsumo")] == 1.0


def test_encode_turn_action_supports_ankan_and_kakan():
    ankan_vector = encode_turn_action({"kind": "ankan", "tile": "1p"})
    kakan_vector = encode_turn_action({"kind": "kakan", "tile": "5m"})

    assert ankan_vector[TURN_ACTION_KINDS.index("ankan")] == 1.0
    assert kakan_vector[TURN_ACTION_KINDS.index("kakan")] == 1.0


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
    assert vector[-8] == 1.0
    assert vector[-7] == 1.0
    assert vector[-6] == 0.0


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
    assert vector[TURN_STATE_DIM - 7] == 0.0
    assert vector[TURN_STATE_DIM - 6] == 1.0
