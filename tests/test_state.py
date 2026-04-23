from __future__ import annotations

import pytest

from mjai.rules.state import HandState


def test_hand_state_counts_red_and_normal_five_separately():
    hand = HandState.from_tiles(["0m", "5m", "5m", "1z"])

    assert hand.total_tiles() == 4
    assert hand.count_exact("0m") == 1
    assert hand.count_exact("5m") == 2
    assert hand.count_exact("1z") == 1


def test_remove_tile_updates_red_bookkeeping():
    hand = HandState.from_tiles(["0p", "5p", "2m"])
    hand.remove_tile("0p")

    assert hand.total_tiles() == 2
    assert hand.count_exact("0p") == 0
    assert hand.count_exact("5p") == 1


def test_remove_missing_tile_raises():
    hand = HandState.from_tiles(["1m"])

    with pytest.raises(ValueError):
        hand.remove_tile("2m")
