from __future__ import annotations

from pathlib import Path

import torch

from mjai.features.action_features import RESPONSE_ACTION_DIM, TURN_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM
from mjai.models.policy_net import MahjongPolicyNet
from mjai.serving.action_adapter import select_response_action
from mjai.serving.predictor import build_predictor_from_config
from mjai.serving.validation import BadRequestError, CandidateLimitError


def test_serving_predictor_returns_turn_action_from_legal_set(tmp_path: Path):
    turn_ckpt = _write_checkpoint(tmp_path / "turn.pt", task="turn", hidden_size=16)
    response_ckpt = _write_checkpoint(tmp_path / "response.pt", task="response", hidden_size=16)
    predictor = build_predictor_from_config(
        _api_config(turn_ckpt, response_ckpt, max_legal_actions=64)
    )

    legal_actions = [
        {"kind": "discard", "tile": "1m"},
        {"kind": "discard", "tile": "2m"},
        {"kind": "discard", "tile": "3m"},
    ]
    result = predictor.predict_turn(_turn_state(), legal_actions, top_k=2)

    assert result.task == "turn"
    assert result.selection_mode == "raw"
    assert result.selected_action in legal_actions
    assert len(result.predictions) == 2
    assert result.model.state_dim == TURN_STATE_DIM
    assert result.model.action_dim == TURN_ACTION_DIM


def test_serving_predictor_returns_calibrated_response_action(tmp_path: Path):
    turn_ckpt = _write_checkpoint(tmp_path / "turn.pt", task="turn", hidden_size=16)
    response_ckpt = _write_checkpoint(tmp_path / "response.pt", task="response", hidden_size=16)
    predictor = build_predictor_from_config(
        _api_config(turn_ckpt, response_ckpt, pass_bias=0.4)
    )

    legal_actions = [
        {"kind": "pass"},
        {"kind": "chi", "tile": "3m", "consumed_tiles": ["1m", "2m"], "source": 0},
    ]
    result = predictor.predict_response(_response_state(), legal_actions, top_k=5)

    assert result.task == "response"
    assert result.selection_mode == "calibrated"
    assert result.pass_bias == 0.4
    assert result.selected_action in legal_actions
    assert result.raw_selected_action in legal_actions
    assert len(result.predictions) == len(legal_actions)
    assert result.model.state_dim == RESPONSE_STATE_DIM
    assert result.model.action_dim == RESPONSE_ACTION_DIM


def test_serving_predictor_preserves_zero_default_pass_bias(tmp_path: Path):
    turn_ckpt = _write_checkpoint(tmp_path / "turn.pt", task="turn", hidden_size=16)
    response_ckpt = _write_checkpoint(tmp_path / "response.pt", task="response", hidden_size=16)
    predictor = build_predictor_from_config(
        _api_config(
            turn_ckpt,
            response_ckpt,
            pass_bias=0.0,
            no_yaku_call_penalty=0.4,
        )
    )

    legal_actions = [
        {"kind": "pass"},
        {"kind": "chi", "tile": "3m", "consumed_tiles": ["1m", "2m"], "source": 0},
    ]
    result = predictor.predict_response(_response_state(), legal_actions, top_k=2)

    assert result.pass_bias == 0.0
    assert result.no_yaku_call_penalty == 0.4
    assert result.model.default_pass_bias == 0.0


def test_response_selection_uses_pass_bias_margin():
    logits = torch.tensor([0.50, 0.75, 0.10], dtype=torch.float32)
    legal_actions = [
        {"kind": "pass"},
        {"kind": "chi", "tile": "3m"},
        {"kind": "pon", "tile": "3m"},
    ]

    conservative = select_response_action(logits, legal_actions, pass_bias=0.30)
    recall = select_response_action(logits, legal_actions, pass_bias=0.10)

    assert conservative.raw_action == legal_actions[1]
    assert conservative.selected_action == legal_actions[0]
    assert recall.selected_action == legal_actions[1]


def test_response_selection_penalizes_no_yaku_call_candidates():
    logits = torch.tensor([0.50, 0.85, 0.20], dtype=torch.float32)
    legal_actions = [
        {"kind": "pass"},
        {"kind": "chi", "tile": "3m"},
        {"kind": "pon", "tile": "5z"},
    ]
    candidate_features = torch.zeros((3, RESPONSE_ACTION_DIM), dtype=torch.float32)
    candidate_features[1, 48 + 38] = 1.0

    without_penalty = select_response_action(
        logits,
        legal_actions,
        pass_bias=0.10,
        candidate_features=candidate_features,
    )
    with_penalty = select_response_action(
        logits,
        legal_actions,
        pass_bias=0.10,
        no_yaku_call_penalty=0.30,
        candidate_features=candidate_features,
    )

    assert without_penalty.selected_action == legal_actions[1]
    assert with_penalty.raw_action == legal_actions[1]
    assert with_penalty.selected_action == legal_actions[0]


def test_serving_predictor_validates_legal_action_count(tmp_path: Path):
    turn_ckpt = _write_checkpoint(tmp_path / "turn.pt", task="turn", hidden_size=16)
    response_ckpt = _write_checkpoint(tmp_path / "response.pt", task="response", hidden_size=16)
    predictor = build_predictor_from_config(
        _api_config(turn_ckpt, response_ckpt, max_legal_actions=1)
    )

    try:
        predictor.predict_turn(
            _turn_state(),
            [{"kind": "discard", "tile": "1m"}, {"kind": "discard", "tile": "2m"}],
        )
    except CandidateLimitError as exc:
        assert "configured limit" in str(exc)
    else:
        raise AssertionError("expected candidate limit validation to fail")

    try:
        predictor.predict_turn(_turn_state(), [])
    except BadRequestError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("expected empty legal_actions validation to fail")


def _write_checkpoint(path: Path, *, task: str, hidden_size: int) -> Path:
    if task == "turn":
        model = MahjongPolicyNet(
            state_dim=TURN_STATE_DIM,
            action_dim=TURN_ACTION_DIM,
            hidden_size=hidden_size,
            dropout=0.0,
        )
        candidate_limit = 64
    else:
        model = MahjongPolicyNet(
            state_dim=RESPONSE_STATE_DIM,
            action_dim=RESPONSE_ACTION_DIM,
            hidden_size=hidden_size,
            dropout=0.0,
        )
        candidate_limit = 16
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "task": {"task": task},
                "model": {"hidden_size": hidden_size, "dropout": 0.0},
                "data": {"candidate_limit": candidate_limit},
                "train": {"device": "cpu"},
            },
        },
        path,
    )
    return path


def _api_config(
    turn_ckpt: Path,
    response_ckpt: Path,
    *,
    max_legal_actions: int = 64,
    pass_bias: float = 2.0,
    no_yaku_call_penalty: float = 0.8,
) -> dict:
    return {
        "api": {"max_legal_actions": max_legal_actions},
        "device": "cpu",
        "models": {
            "turn": {"checkpoint": str(turn_ckpt)},
            "response": {
                "checkpoint": str(response_ckpt),
                "pass_bias": {"default": pass_bias},
                "no_yaku_call_penalty": no_yaku_call_penalty,
            },
        },
    }


def _turn_state() -> dict:
    return {
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
            "1p": 1,
            "1s": 1,
            "2s": 1,
            "3s": 1,
            "4s": 1,
        },
        "all_discards": [[], [], [], []],
        "last_discard_tile": None,
        "last_discard_actor": None,
        "riichi_declared": False,
        "temporary_furiten": False,
        "riichi_furiten": False,
        "own_melds": [],
        "dealer": 0,
        "round_wind": 0,
        "seat_wind": 0,
        "dora_indicators": [],
        "scores": [25000, 25000, 25000, 25000],
        "turn_number": 1,
        "remaining_draws_estimate": 69,
        "honba": 0,
        "riichi_sticks": 0,
    }


def _response_state() -> dict:
    state = _turn_state()
    state.update(
        {
            "actor": 1,
            "hand": {
                "1m": 1,
                "2m": 1,
                "4m": 1,
                "5m": 1,
                "6m": 1,
                "7m": 1,
                "2p": 1,
                "3p": 1,
                "4p": 1,
                "2s": 1,
                "3s": 1,
                "4s": 1,
                "5s": 1,
            },
            "all_discards": [["3m"], [], [], []],
            "last_discard_tile": "3m",
            "last_discard_actor": 0,
            "seat_wind": 1,
            "all_melds": [[], [], [], []],
            "response_target": {"source": 0, "tile": "3m"},
        }
    )
    return state
