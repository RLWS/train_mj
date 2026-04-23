from __future__ import annotations

from mjai.rules.hand_checker import is_agari
from mjai.rules.state import HandState


def test_is_agari_for_standard_hand_with_incoming_tile():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"]
    )

    assert is_agari(hand, incoming_tile="5z") is True


def test_is_agari_for_chiitoitsu():
    hand = HandState.from_tiles(
        ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "6s", "6s", "7z", "7z"]
    )

    assert is_agari(hand) is True


def test_is_not_agari_for_incomplete_hand():
    hand = HandState.from_tiles(
        ["1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "2p", "3p", "4p", "5z", "6z"]
    )

    assert is_agari(hand) is False
