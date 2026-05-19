from __future__ import annotations

import pytest

from mjai.rules.state import GameState, HandState, MeldState
from mjai.rules.tiles import tile_to_base_index


def test_hand_state_counts_red_and_normal_five_separately():
    """验证手牌状态分别统计赤五和普通五。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    hand = HandState.from_tiles(["0m", "5m", "5m", "1z"])

    assert hand.total_tiles() == 4
    assert hand.count_exact("0m") == 1
    assert hand.count_exact("5m") == 2
    assert hand.count_exact("1z") == 1


def test_remove_tile_updates_red_bookkeeping():
    """验证移除赤牌时手牌状态更新赤牌计数。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    hand = HandState.from_tiles(["0p", "5p", "2m"])
    hand.remove_tile("0p")

    assert hand.total_tiles() == 2
    assert hand.count_exact("0p") == 0
    assert hand.count_exact("5p") == 1


def test_remove_missing_tile_raises():
    """验证移除缺失牌时抛出错误。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    hand = HandState.from_tiles(["1m"])

    with pytest.raises(ValueError):
        hand.remove_tile("2m")


def test_riichi_pending_becomes_declared_on_step_two():
    """验证立直第二步会将待确认立直转为已宣告。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = GameState()

    state.apply_reach(actor=0, step=1)
    assert state.riichi_pending[0] is True
    assert state.riichi_declared[0] is False

    state.apply_reach(actor=0, step=2)
    assert state.riichi_pending[0] is False
    assert state.riichi_declared[0] is True
    assert state.riichi_turns[0] == 0


def test_riichi_turn_uses_player_discard_count_at_confirmation():
    state = GameState(
        hands=[HandState.from_tiles(["1m", "2m"]), HandState(), HandState(), HandState()]
    )

    state.apply_discard(actor=0, tile="1m")
    state.apply_reach(actor=0, step=1)
    state.apply_reach(actor=0, step=2)

    assert state.riichi_turns[0] == 1


def test_can_declare_riichi_is_blocked_while_pending():
    """验证立直待确认时不能重复宣告立直。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = GameState()

    assert state.can_declare_riichi(0) is True
    state.apply_reach(actor=0, step=1)
    assert state.can_declare_riichi(0) is False


def test_discard_records_furiten_base_tile():
    """验证弃牌会记录振听基础牌。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    hand = HandState.from_tiles(["5z"])
    state = GameState(hands=[hand, HandState(), HandState(), HandState()])

    state.apply_discard(actor=0, tile="5z")

    assert state.is_furiten(0, "5z") is True


def test_discard_history_records_global_order():
    state = GameState(
        hands=[
            HandState.from_tiles(["1m"]),
            HandState.from_tiles(["2m"]),
            HandState.from_tiles(["3m"]),
            HandState(),
        ]
    )

    state.apply_discard(actor=1, tile="2m")
    state.apply_discard(actor=0, tile="1m")
    state.apply_discard(actor=2, tile="3m")

    assert state.discard_history == [(1, "2m"), (0, "1m"), (2, "3m")]


def test_draw_clears_temporary_furiten_only():
    """验证摸牌只清除临时振听而不清除立直振听。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = GameState()
    state.temporary_furiten[0] = True
    state.riichi_furiten[0] = True

    state.apply_draw(actor=0, tile="1m")

    assert state.temporary_furiten[0] is False
    assert state.riichi_furiten[0] is True


def test_open_meld_clears_temporary_furiten_only():
    """验证明副露只清除临时振听而不清除立直振听。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = GameState(
        hands=[
            HandState.from_tiles(["5m", "5m"]),
            HandState.from_tiles(["5m"]),
            HandState(),
            HandState(),
        ]
    )
    state.apply_discard(actor=1, tile="5m")
    state.temporary_furiten[0] = True
    state.riichi_furiten[0] = True

    state.apply_meld(
        actor=0,
        kind="pon",
        source=1,
        called_tile="5m",
        consumed_tiles=["5m", "5m"],
    )

    assert state.temporary_furiten[0] is False
    assert state.riichi_furiten[0] is True


def test_mark_missed_ron_sets_riichi_furiten_after_riichi():
    """验证立直后错过荣和会标记立直振听。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = GameState()
    state.riichi_declared[0] = True

    state.mark_missed_ron(0)

    assert state.temporary_furiten[0] is True
    assert state.riichi_furiten[0] is True


def test_game_state_tracks_round_context_and_dora():
    """验证游戏状态跟踪局况上下文和宝牌。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = GameState.from_init_payload(
        0,
        {
            "seed": "0,2,1,0,0,16",
            "ten": "250,260,240,250",
            "oya": 2,
            "dora_indicator": "5m",
            "hai0": ["1m"] * 13,
            "hai1": ["1p"] * 13,
            "hai2": ["1s"] * 13,
            "hai3": ["1z"] * 13,
        },
    )

    assert state.dealer == 2
    assert state.honba == 2
    assert state.riichi_sticks == 1
    assert tile_to_base_index("6m") in state.dora_base_indices()


def test_kakan_upgrades_existing_pon_and_consumes_added_tile():
    """验证加杠会升级已有碰牌并消耗加杠牌。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = GameState(
        hands=[HandState.from_tiles(["5m"]), HandState(), HandState(), HandState()],
        melds=[
            [MeldState(kind="pon", source=1, called_tile="5m", consumed_tiles=("0m", "5m"))],
            [],
            [],
            [],
        ],
    )

    state.apply_meld(
        actor=0,
        kind="kakan",
        source=1,
        called_tile="5m",
        consumed_tiles=["0m", "5m", "5m"],
    )

    assert state.hands[0].count_exact("5m") == 0
    assert state.melds[0][0].kind == "kakan"
    assert state.melds[0][0].consumed_tiles == ("0m", "5m", "5m")
