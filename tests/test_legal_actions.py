from __future__ import annotations

from mjai.rules.legal_actions import legal_discards
from mjai.rules.state import HandState


def test_legal_discards_include_red_and_normal_five_when_both_exist():
    hand = HandState.from_tiles(["0m", "5m", "6m", "1z"])

    discards = legal_discards(hand)
    tiles = [action.tile for action in discards]

    assert "0m" in tiles
    assert "5m" in tiles
    assert "6m" in tiles
    assert "1z" in tiles


def test_legal_discards_do_not_include_missing_tiles():
    hand = HandState.from_tiles(["1m", "1m", "3p"])

    discards = legal_discards(hand)
    tiles = {action.tile for action in discards}

    assert "1m" in tiles
    assert "3p" in tiles
    assert "2m" not in tiles
    assert "0m" not in tiles
