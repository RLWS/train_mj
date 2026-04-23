from __future__ import annotations

from mjai.data.tenhou.meld_decoder import decode_meld


def _encode_chi(base_index: int, called_offset: int, copies: tuple[int, int, int], from_who: int) -> int:
    sequence_index = (base_index // 9) * 7 + (base_index % 9)
    value = from_who
    value |= 0x4
    value |= copies[0] << 3
    value |= copies[1] << 5
    value |= copies[2] << 7
    value |= (sequence_index * 3 + called_offset) << 10
    return value


def _encode_pon(tile_base: int, called_index: int, unused_copy: int, from_who: int) -> int:
    value = from_who
    value |= 0x8
    value |= unused_copy << 5
    value |= (tile_base * 3 + called_index) << 9
    return value


def test_decode_chi_meld():
    value = _encode_chi(base_index=3, called_offset=1, copies=(0, 1, 2), from_who=3)
    meld = decode_meld(actor=0, value=value)

    assert meld.kind == "chi"
    assert meld.source == 3
    assert meld.called_tile == "5m"
    assert meld.consumed_tiles == ("4m", "6m")


def test_decode_pon_meld_with_red_tile():
    value = _encode_pon(tile_base=4, called_index=0, unused_copy=3, from_who=1)
    meld = decode_meld(actor=0, value=value)

    assert meld.kind == "pon"
    assert meld.source == 1
    assert meld.called_tile in {"0m", "5m"}
    assert len(meld.consumed_tiles) == 2


def test_decode_ankan_values_from_real_logs():
    meld_a = decode_meld(actor=2, value=9216)
    meld_b = decode_meld(actor=3, value=17408)

    assert meld_a.kind == "ankan"
    assert meld_a.called_tile == "1p"
    assert meld_a.consumed_tiles == ("1p", "1p", "1p", "1p")

    assert meld_b.kind == "ankan"
    assert meld_b.called_tile == "9p"
    assert meld_b.consumed_tiles == ("9p", "9p", "9p", "9p")
