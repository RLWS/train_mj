from __future__ import annotations

import pytest

from mjai.features.state_features import (
    ACTOR_DIM,
    ALL_DISCARD_COUNTS_DIM,
    HAND_COUNTS_DIM,
    LAST_DISCARD_ACTOR_DIM,
    LAST_DISCARD_TILE_DIM,
    RESPONSE_STATE_DIM,
    RIICHI_FLAG_DIM,
    TURN_STATE_DIM,
    encode_response_state,
    encode_turn_state,
)
from mjai.features.action_features import (
    RESPONSE_ACTION_DIM,
    RESPONSE_ACTION_KINDS,
    RESPONSE_CALL_FEATURE_DIM,
    TURN_ACTION_KINDS,
    TURN_DISCARD_FEATURE_DIM,
    encode_response_action,
    encode_turn_action,
)


def test_encode_turn_action_supports_tsumo():
    """验证自摸巡动作编码支持自摸。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    vector = encode_turn_action({"kind": "tsumo"})

    assert len(vector) > len(TURN_ACTION_KINDS) + TURN_DISCARD_FEATURE_DIM
    assert vector[TURN_ACTION_KINDS.index("tsumo")] == 1.0


def test_encode_turn_action_supports_ankan_and_kakan():
    """验证自摸巡动作编码支持暗杠and加杠。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    ankan_vector = encode_turn_action({"kind": "ankan", "tile": "1p"})
    kakan_vector = encode_turn_action({"kind": "kakan", "tile": "5m"})

    assert ankan_vector[TURN_ACTION_KINDS.index("ankan")] == 1.0
    assert kakan_vector[TURN_ACTION_KINDS.index("kakan")] == 1.0


def test_encode_turn_action_supports_kyuushu_kyuuhai():
    """验证自摸巡动作编码支持kyuushukyuuhai。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    vector = encode_turn_action({"kind": "kyuushu_kyuuhai"})

    assert vector[TURN_ACTION_KINDS.index("kyuushu_kyuuhai")] == 1.0


def test_encode_response_action_supports_chankan():
    """验证响应动作编码支持抢杠。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    vector = encode_response_action({"kind": "chankan", "tile": "5m", "source": 1})

    assert vector[RESPONSE_ACTION_KINDS.index("chankan")] == 1.0
    assert len(vector) == RESPONSE_ACTION_DIM


def test_encode_response_action_adds_call_shape_features():
    """验证响应动作编码增加鸣牌形状特征。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = {
        "actor": 0,
        "hand": {
            "2m": 1,
            "4m": 1,
            "5m": 1,
            "6m": 1,
            "7m": 1,
            "2p": 1,
            "3p": 1,
            "4p": 1,
            "5p": 2,
            "6p": 1,
            "5z": 2,
        },
        "all_discards": [[], [], [], []],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "all_melds": [[], [], [], []],
        "last_discard_actor": 3,
        "last_discard_tile": "3m",
        "response_target": {"source": 3, "tile": "3m"},
        "dora_base_indices": [4],
        "turn_number": 8,
        "remaining_draws_estimate": 10,
    }

    vector = encode_response_action(
        {"kind": "chi", "tile": "3m", "source": 3, "consumed_tiles": ["2m", "4m"]},
        state,
    )

    assert len(vector) == RESPONSE_ACTION_DIM
    call_features = vector[-RESPONSE_CALL_FEATURE_DIM:]
    assert call_features[0] == 1.0
    assert call_features[1] == 1.0
    assert call_features[4] != 0.0
    assert call_features[20] == 1.0
    assert call_features[27] == 8 / 18


def test_encode_response_action_marks_yakuhai_pon_and_red_cost():
    """验证响应动作编码标记役牌碰牌and赤牌代价。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = {
        "actor": 1,
        "hand": {"0p": 1, "5p": 1, "5z": 2, "1m": 1, "2m": 1, "3m": 1},
        "all_discards": [[], [], [], []],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "all_melds": [[], [], [], []],
        "last_discard_actor": 0,
        "last_discard_tile": "5z",
        "response_target": {"source": 0, "tile": "5z"},
        "dealer": 0,
        "seat_wind": 1,
        "dora_base_indices": [13],
        "turn_number": 3,
        "remaining_draws_estimate": 15,
    }

    pon_vector = encode_response_action(
        {"kind": "pon", "tile": "5z", "source": 0, "consumed_tiles": ["5z", "5z"]},
        state,
    )
    red_vector = encode_response_action(
        {"kind": "pon", "tile": "5p", "source": 0, "consumed_tiles": ["0p", "5p"]},
        state,
    )

    pon_features = pon_vector[-RESPONSE_CALL_FEATURE_DIM:]
    red_features = red_vector[-RESPONSE_CALL_FEATURE_DIM:]
    assert pon_features[16] == 1.0
    assert red_features[11] == 1.0
    assert red_features[12] == 1.0


def test_encode_response_action_marks_open_yaku_routes_and_no_yaku_risk():
    """验证响应动作编码标记开副露役路与无役风险。"""
    no_yaku_state = {
        "actor": 0,
        "hand": {
            "2m": 1,
            "4m": 1,
            "7m": 1,
            "8m": 1,
            "9m": 1,
            "2p": 1,
            "3p": 1,
            "4p": 1,
            "5z": 1,
            "6z": 1,
            "7z": 1,
        },
        "all_discards": [[], [], [], ["3m"]],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "all_melds": [[], [], [], []],
        "last_discard_actor": 3,
        "last_discard_tile": "3m",
        "response_target": {"source": 3, "tile": "3m"},
        "dealer": 0,
        "seat_wind": 0,
        "dora_base_indices": [],
        "turn_number": 5,
        "remaining_draws_estimate": 13,
    }
    tanyao_state = {
        **no_yaku_state,
        "hand": {
            "2m": 1,
            "4m": 1,
            "3p": 1,
            "4p": 1,
            "5p": 1,
            "6p": 1,
            "7p": 1,
            "3s": 1,
            "4s": 1,
            "5s": 1,
            "6s": 1,
        },
    }

    no_yaku_vector = encode_response_action(
        {"kind": "chi", "tile": "3m", "source": 3, "consumed_tiles": ["2m", "4m"]},
        no_yaku_state,
    )
    tanyao_vector = encode_response_action(
        {"kind": "chi", "tile": "3m", "source": 3, "consumed_tiles": ["2m", "4m"]},
        tanyao_state,
    )

    no_yaku_features = no_yaku_vector[-RESPONSE_CALL_FEATURE_DIM:]
    tanyao_features = tanyao_vector[-RESPONSE_CALL_FEATURE_DIM:]
    assert no_yaku_features[32] == 1.0
    assert no_yaku_features[33] == 1.0
    assert no_yaku_features[38] == 1.0
    assert tanyao_features[35] == 1.0
    assert tanyao_features[38] == 0.0


def test_encode_response_action_marks_chiitoitsu_route_break_and_visible_wait_copies():
    """验证响应动作编码标记七对子路线破坏与可见牌修正有效牌枚数。"""
    state = {
        "actor": 0,
        "hand": {
            "1m": 2,
            "2m": 2,
            "3m": 2,
            "4m": 2,
            "5p": 2,
            "6p": 2,
            "7z": 1,
        },
        "all_discards": [["7z"], ["7z"], ["7z"], ["1m"]],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "all_melds": [[], [], [], []],
        "last_discard_actor": 3,
        "last_discard_tile": "1m",
        "response_target": {"source": 3, "tile": "1m"},
        "dealer": 0,
        "seat_wind": 0,
        "dora_base_indices": [],
        "turn_number": 7,
        "remaining_draws_estimate": 11,
    }

    vector = encode_response_action(
        {"kind": "pon", "tile": "1m", "source": 3, "consumed_tiles": ["1m", "1m"]},
        state,
    )

    call_features = vector[-RESPONSE_CALL_FEATURE_DIM:]
    assert call_features[29] < call_features[8]
    assert call_features[30] <= call_features[9]
    assert call_features[39] == 0.0
    assert call_features[40] == 1.0


def test_encode_turn_state_includes_furiten_flags():
    """验证自摸巡状态编码包含振听标志。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = {
        "actor": 1,
        "hand": {"1m": 1},
        "all_discards": [[], [], [], []],
        "riichi_declared": True,
        "temporary_furiten": True,
        "riichi_furiten": False,
        "own_melds": [],
        "last_discard_actor": None,
        "last_discard_tile": None,
    }

    vector = encode_turn_state(state)

    assert len(vector) == TURN_STATE_DIM
    furiten_offset = _furiten_offset()
    assert vector[furiten_offset - RIICHI_FLAG_DIM] == 1.0
    assert vector[furiten_offset] == 1.0
    assert vector[furiten_offset + 1] == 0.0


def test_encode_response_state_includes_furiten_flags():
    """验证响应状态编码包含振听标志。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = {
        "actor": 2,
        "hand": {"1m": 1},
        "all_discards": [[], [], [], []],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": True,
        "own_melds": [],
        "last_discard_actor": None,
        "last_discard_tile": None,
        "response_target": {"source": 0, "tile": "5z"},
    }

    vector = encode_response_state(state)

    assert len(vector) == RESPONSE_STATE_DIM
    furiten_offset = _furiten_offset()
    assert vector[furiten_offset] == 0.0
    assert vector[furiten_offset + 1] == 1.0


def test_encode_turn_action_adds_discard_candidate_features():
    """验证自摸巡动作编码增加弃牌候选特征。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = {
        "actor": 0,
        "hand": {
            "1m": 1,
            "2m": 1,
            "3m": 1,
            "4m": 1,
            "5m": 1,
            "6m": 1,
            "7m": 1,
            "8m": 1,
            "9m": 1,
            "2p": 1,
            "3p": 1,
            "4p": 1,
            "5z": 2,
        },
        "all_discards": [["9m"], [], [], []],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "last_discard_actor": None,
        "last_discard_tile": None,
    }

    vector = encode_turn_action({"kind": "discard", "tile": "9m"}, state)

    assert len(vector) == len(TURN_ACTION_KINDS) + 37 + TURN_DISCARD_FEATURE_DIM
    assert vector[-TURN_DISCARD_FEATURE_DIM] == 1.0


def test_encode_turn_action_adds_discard_river_timing_features():
    """验证自摸巡弃牌候选编码包含牌河时序摘要。"""
    state = {
        "actor": 0,
        "hand": {
            "3m": 1,
            "4m": 1,
            "5m": 1,
            "6m": 1,
            "7m": 1,
            "2p": 1,
            "3p": 1,
            "4p": 1,
            "5p": 1,
            "6p": 1,
            "5z": 2,
            "6z": 1,
        },
        "all_discards": [
            ["3m", "9p"],
            ["1p", "2p", "4p", "5p", "6p", "7p", "3m"],
            ["3m", "1s", "2s", "3s", "4s", "5s", "6s"],
            ["9s", "8s", "7s", "6s", "5s", "4s", "3m"],
        ],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "last_discard_actor": 3,
        "last_discard_tile": "3m",
        "dora_base_indices": [],
    }

    vector = encode_turn_action({"kind": "discard", "tile": "3m"}, state)
    features = vector[-TURN_DISCARD_FEATURE_DIM:]

    assert features[25] == pytest.approx(1 / 4)
    assert features[26] == pytest.approx(3 / 4)
    assert features[27] == 1.0
    assert features[28] == pytest.approx(2 / 3)
    assert features[29] == pytest.approx(1 / 3)
    assert features[30] == pytest.approx(1 / 18)
    assert features[31] == pytest.approx(7 / 18)
    assert features[32] == 1.0
    assert features[33] == 1.0
    assert features[34] == 1.0
    assert features[35:38] == [1.0, 1.0, 1.0]
    assert features[38:41] == [1.0, 0.0, 1.0]


def test_encode_turn_state_includes_round_context():
    """验证自摸巡状态编码包含局况上下文。
    
    参数:
        无。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    state = {
        "actor": 0,
        "hand": {"1m": 1},
        "all_discards": [[], [], [], []],
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "last_discard_actor": None,
        "last_discard_tile": None,
        "dealer": 2,
        "seat_wind": 2,
        "scores": [250, 260, 240, 250],
        "dora_base_indices": [1],
        "turn_number": 9,
        "remaining_draws_estimate": 9,
        "honba": 2,
        "riichi_sticks": 1,
    }

    vector = encode_turn_state(state)

    assert len(vector) == TURN_STATE_DIM
    assert vector[-4:] == [0.5, 0.5, 0.25, 0.125]


def _furiten_offset() -> int:
    """计算振听标志在状态特征向量中的起始偏移。
    
    参数:
        无。
    
    返回:
        int: 振听标志特征起始偏移量。
    
    异常:
        无。
    """
    return (
        HAND_COUNTS_DIM
        + ALL_DISCARD_COUNTS_DIM
        + LAST_DISCARD_TILE_DIM
        + LAST_DISCARD_ACTOR_DIM
        + ACTOR_DIM
        + RIICHI_FLAG_DIM
    )
