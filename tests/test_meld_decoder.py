from __future__ import annotations

from mjai.data.tenhou.meld_decoder import decode_meld


def _encode_chi(base_index: int, called_offset: int, copies: tuple[int, int, int], from_who: int) -> int:
    """构造用于测试吃牌副露解码的天凤压缩整数。
    
    参数:
        base_index (int): 顺子起始牌的基础索引。
        called_offset (int): 被鸣牌在顺子中的偏移位置。
        copies (tuple[int, int, int]): 顺子三张牌在天凤 136 牌编号中的副本索引。
        from_who (int): 天凤副露值中表示来源玩家的相对索引。
    
    返回:
        int: 编码后的天凤副露整数。
    
    异常:
        无。
    """
    sequence_index = (base_index // 9) * 7 + (base_index % 9)
    value = from_who
    value |= 0x4
    value |= copies[0] << 3
    value |= copies[1] << 5
    value |= copies[2] << 7
    value |= (sequence_index * 3 + called_offset) << 10
    return value


def _encode_pon(tile_base: int, called_index: int, unused_copy: int, from_who: int) -> int:
    """构造用于测试碰牌副露解码的天凤压缩整数。
    
    参数:
        tile_base (int): 碰牌牌型的基础索引。
        called_index (int): 被鸣牌在副露牌组中的索引。
        unused_copy (int): 碰牌编码中未使用的副本索引。
        from_who (int): 天凤副露值中表示来源玩家的相对索引。
    
    返回:
        int: 编码后的天凤副露整数。
    
    异常:
        无。
    """
    value = from_who
    value |= 0x8
    value |= unused_copy << 5
    value |= (tile_base * 3 + called_index) << 9
    return value


def test_decode_chi_meld():
    """验证吃牌副露解码结果。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    value = _encode_chi(base_index=3, called_offset=1, copies=(0, 1, 2), from_who=3)
    meld = decode_meld(actor=0, value=value)

    assert meld.kind == "chi"
    assert meld.source == 3
    assert meld.called_tile == "5m"
    assert meld.consumed_tiles == ("4m", "6m")


def test_decode_pon_meld_with_red_tile():
    """验证含赤牌碰牌副露的解码结果。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    value = _encode_pon(tile_base=4, called_index=0, unused_copy=3, from_who=1)
    meld = decode_meld(actor=0, value=value)

    assert meld.kind == "pon"
    assert meld.source == 1
    assert meld.called_tile in {"0m", "5m"}
    assert len(meld.consumed_tiles) == 2


def test_decode_ankan_values_from_real_logs():
    """验证真实日志中暗杠数值的解码结果。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    meld_a = decode_meld(actor=2, value=9216)
    meld_b = decode_meld(actor=3, value=17408)

    assert meld_a.kind == "ankan"
    assert meld_a.called_tile == "1p"
    assert meld_a.consumed_tiles == ("1p", "1p", "1p", "1p")

    assert meld_b.kind == "ankan"
    assert meld_b.called_tile == "9p"
    assert meld_b.consumed_tiles == ("9p", "9p", "9p", "9p")


def test_decode_daiminkan_uses_anchor_tile_as_called_tile():
    """验证大明杠解码使用锚点牌作为被鸣牌。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    value = (88 << 8) | 1
    meld = decode_meld(actor=2, value=value)

    assert meld.kind == "daiminkan"
    assert meld.source == 3
    assert meld.called_tile == "0s"
    assert meld.consumed_tiles == ("5s", "5s", "5s")


def test_decode_daiminkan_preserves_red_tile_when_it_is_consumed():
    """验证大明杠解码在赤牌被消耗时保留赤牌。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    value = (89 << 8) | 1
    meld = decode_meld(actor=2, value=value)

    assert meld.kind == "daiminkan"
    assert meld.called_tile == "5s"
    assert meld.consumed_tiles == ("0s", "5s", "5s")
