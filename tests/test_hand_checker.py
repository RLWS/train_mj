from __future__ import annotations

from mjai.rules.hand_checker import effective_tile_base_indices, is_agari, shanten_number
from mjai.rules.state import HandState
from mjai.rules.tiles import tile_to_base_index


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


def test_shanten_number_returns_minus_one_for_complete_hand():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z", "5z"]
    )

    assert shanten_number(hand) == -1


def test_effective_tile_base_indices_include_tenpai_waits():
    hand = HandState.from_tiles(
        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "2p", "3p", "4p", "5z"]
    )

    assert tile_to_base_index("5z") in effective_tile_base_indices(hand)
