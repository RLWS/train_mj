from __future__ import annotations

from mjai.serving.validation import validate_and_normalize_state


def test_turn_state_validation_normalizes_riichi_and_discard_history() -> None:
    state = validate_and_normalize_state(
        "turn",
        {
            "actor": 3,
            "hand": {"1m": 1},
            "all_discards": [["1m"], ["2m"], ["3m"], ["4m"]],
            "riichi_declared": False,
            "riichi_declared_by_player": [True, 0, 1],
            "riichi_turn_by_player": [3, None, "8"],
            "discard_history": [
                {"actor": "0", "tile": "1m"},
                {"actor": 2, "tile": "0p"},
            ],
        },
    )

    assert state["riichi_declared_by_player"] == [True, False, True, False]
    assert state["riichi_turn_by_player"] == [3, None, 8, None]
    assert state["discard_history"] == [
        {"actor": 0, "tile": "1m"},
        {"actor": 2, "tile": "0p"},
    ]


def test_turn_state_validation_backfills_legacy_own_riichi_flag() -> None:
    state = validate_and_normalize_state(
        "turn",
        {
            "actor": 2,
            "hand": {"1m": 1},
            "all_discards": [[], [], [], []],
            "riichi_declared": True,
        },
    )

    assert state["riichi_declared_by_player"] == [False, False, True, False]
    assert state["riichi_turn_by_player"] == [None, None, None, None]
    assert state["discard_history"] == []
