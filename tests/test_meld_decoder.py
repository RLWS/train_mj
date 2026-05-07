from __future__ import annotations

from mjai.data.tenhou.meld_decoder import decode_meld


def _encode_chi(base_index: int, called_offset: int, copies: tuple[int, int, int], from_who: int) -> int:
    """将输入数据编码为模型可用特征。
    
    参数:
        base_index (int): 三十四种牌基础索引。
        called_offset (int): 函数执行所需的输入参数。
        copies (tuple[int, int, int]): 相关数据列表。
        from_who (int): 函数执行所需的输入参数。
    
    返回:
        int: 整数结果。
    
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
    """将输入数据编码为模型可用特征。
    
    参数:
        tile_base (int): 函数执行所需的输入参数。
        called_index (int): 相关索引值。
        unused_copy (int): 函数执行所需的输入参数。
        from_who (int): 函数执行所需的输入参数。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    value = from_who
    value |= 0x8
    value |= unused_copy << 5
    value |= (tile_base * 3 + called_index) << 9
    return value


def test_decode_chi_meld():
    """验证目标测试场景的行为符合预期。
    
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
    """验证目标测试场景的行为符合预期。
    
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
    """验证目标测试场景的行为符合预期。
    
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
    """验证目标测试场景的行为符合预期。
    
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
    """验证目标测试场景的行为符合预期。
    
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
