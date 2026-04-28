from __future__ import annotations

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
