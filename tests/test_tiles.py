from __future__ import annotations

import pytest

from mjai.rules.tiles import (
    base_index_to_tile,
    is_aka_tile,
    normalize_tile,
    parse_tile,
    tile_to_base_index,
)


def test_red_five_and_normal_five_share_base_index():
    assert tile_to_base_index("0m") == tile_to_base_index("5m") == 4
    assert tile_to_base_index("0p") == tile_to_base_index("5p") == 13
    assert tile_to_base_index("0s") == tile_to_base_index("5s") == 22


def test_parse_tile_marks_red_five():
    tile = parse_tile("0m")

    assert tile.base_index == 4
    assert tile.is_aka is True
    assert is_aka_tile(tile.text) is True


def test_normalize_tile_converts_red_five_to_base_tile():
    assert normalize_tile("0m") == "5m"
    assert normalize_tile("0p") == "5p"
    assert normalize_tile("7z") == "7z"


def test_base_index_to_tile_round_trip_for_non_red_tiles():
    assert base_index_to_tile(0) == "1m"
    assert base_index_to_tile(13) == "5p"
    assert base_index_to_tile(33) == "7z"


@pytest.mark.parametrize("tile", ["0z", "9z", "xm", "55m"])
def test_invalid_tiles_raise(tile: str):
    with pytest.raises(ValueError):
        tile_to_base_index(tile)
