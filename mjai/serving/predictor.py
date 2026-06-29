from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from mjai.features.action_features import (
    encode_response_candidate_actions,
    encode_turn_candidate_actions,
)
from mjai.features.state_features import encode_response_state, encode_turn_state
from mjai.serving.action_adapter import rank_candidates, select_response_action
from mjai.serving.checkpoint import load_policy_checkpoint
from mjai.serving.schemas import LoadedPolicy, PredictionResult
from mjai.serving.validation import (
    BadRequestError,
    ServingError,
    validate_and_normalize_state,
    validate_legal_actions,
)


class MahjongPredictor:
    def __init__(
        self,
        turn_policy: LoadedPolicy,
        response_policy: LoadedPolicy,
        *,
        max_legal_actions: int = 64,
    ) -> None:
        self.turn_policy = turn_policy
        self.response_policy = response_policy
        self.max_legal_actions = int(max_legal_actions)

    def predict_turn(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        top_k: int = 5,
    ) -> PredictionResult:
        if top_k <= 0:
            raise BadRequestError("top_k must be positive")
        normalized_state = validate_and_normalize_state("turn", state)
        checked_legal_actions = validate_legal_actions(
            legal_actions,
            candidate_limit=min(self.max_legal_actions, self.turn_policy.candidate_limit),
            task="turn",
        )
        return self._predict_turn(normalized_state, checked_legal_actions, top_k=top_k)

    def predict_response(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        top_k: int = 5,
        pass_bias: float | None = None,
    ) -> PredictionResult:
        if top_k <= 0:
            raise BadRequestError("top_k must be positive")
        normalized_state = validate_and_normalize_state("response", state)
        checked_legal_actions = validate_legal_actions(
            legal_actions,
            candidate_limit=min(self.max_legal_actions, self.response_policy.candidate_limit),
            task="response",
        )
        if pass_bias is None:
            if self.response_policy.default_pass_bias is None:
                pass_bias = 2.0
            else:
                pass_bias = self.response_policy.default_pass_bias
        if pass_bias < 0:
            raise BadRequestError("pass_bias must be non-negative")
        return self._predict_response(
            normalized_state,
            checked_legal_actions,
            top_k=top_k,
            pass_bias=float(pass_bias),
        )

    def model_metadata(self) -> dict[str, dict[str, Any]]:
        return {
            "turn": asdict(self.turn_policy.metadata()),
            "response": asdict(self.response_policy.metadata()),
        }

    def _predict_turn(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> PredictionResult:
        try:
            state_feats = torch.tensor(
                [encode_turn_state(state)],
                dtype=torch.float32,
                device=self.turn_policy.device,
            )
            candidate_feats = torch.tensor(
                [encode_turn_candidate_actions(legal_actions, state)],
                dtype=torch.float32,
                device=self.turn_policy.device,
            )
            candidate_mask = torch.ones(
                (1, len(legal_actions)),
                dtype=torch.bool,
                device=self.turn_policy.device,
            )
            batch = {
                "state_feats": state_feats,
                "candidate_feats": candidate_feats,
                "candidate_mask": candidate_mask,
            }
            with torch.no_grad():
                logits = self.turn_policy.model.forward_turn(batch)[0]
        except ServingError:
            raise
        except Exception as exc:
            raise RuntimeError(f"turn prediction failed: {exc}") from exc

        predictions = rank_candidates(logits, legal_actions, top_k)
        selected_index = int(torch.argmax(logits).item())
        return PredictionResult(
            task="turn",
            selection_mode="raw",
            selected_action=dict(legal_actions[selected_index]),
            predictions=predictions,
            model=self.turn_policy.metadata(),
        )

    def _predict_response(
        self,
        state: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        top_k: int,
        pass_bias: float,
    ) -> PredictionResult:
        try:
            state_feats = torch.tensor(
                [encode_response_state(state)],
                dtype=torch.float32,
                device=self.response_policy.device,
            )
            candidate_feats = torch.tensor(
                [encode_response_candidate_actions(legal_actions, state)],
                dtype=torch.float32,
                device=self.response_policy.device,
            )
            candidate_mask = torch.ones(
                (1, len(legal_actions)),
                dtype=torch.bool,
                device=self.response_policy.device,
            )
            batch = {
                "state_feats": state_feats,
                "candidate_feats": candidate_feats,
                "candidate_mask": candidate_mask,
            }
            with torch.no_grad():
                logits = self.response_policy.model.forward_response(batch)[0]
        except ServingError:
            raise
        except Exception as exc:
            raise RuntimeError(f"response prediction failed: {exc}") from exc

        predictions = rank_candidates(logits, legal_actions, top_k)
        raw_selection_index = int(torch.argmax(logits).item())
        selection = select_response_action(
            logits,
            legal_actions,
            pass_bias=pass_bias,
            no_yaku_call_penalty=self.response_policy.no_yaku_call_penalty,
            candidate_features=candidate_feats[0],
        )
        return PredictionResult(
            task="response",
            selection_mode="calibrated",
            selected_action=selection.selected_action,
            raw_selected_action=dict(legal_actions[raw_selection_index]),
            pass_bias=pass_bias,
            no_yaku_call_penalty=self.response_policy.no_yaku_call_penalty,
            predictions=predictions,
            model=self.response_policy.metadata(),
        )


def build_predictor_from_config(config: dict[str, Any], *, base_dir: str | Path | None = None) -> MahjongPredictor:
    base_path = Path(base_dir) if base_dir is not None else Path.cwd()
    models_config = config.get("models", {})
    turn_config = models_config.get("turn")
    response_config = models_config.get("response")
    if not isinstance(turn_config, dict):
        raise TypeError("models.turn must be configured")
    if not isinstance(response_config, dict):
        raise TypeError("models.response must be configured")

    device = config.get("device", config.get("train", {}).get("device"))
    turn_checkpoint = _resolve_checkpoint_path(base_path, turn_config.get("checkpoint"))
    response_checkpoint = _resolve_checkpoint_path(base_path, response_config.get("checkpoint"))
    turn_policy = load_policy_checkpoint(turn_checkpoint, task="turn", device=device)
    response_policy = load_policy_checkpoint(response_checkpoint, task="response", device=device)
    response_policy.default_pass_bias = _configured_pass_bias(response_config)
    response_policy.no_yaku_call_penalty = _configured_no_yaku_call_penalty(response_config)
    max_legal_actions = int(config.get("api", {}).get("max_legal_actions", 64))
    return MahjongPredictor(
        turn_policy=turn_policy,
        response_policy=response_policy,
        max_legal_actions=max_legal_actions,
    )


def _resolve_checkpoint_path(base_dir: Path, value: Any) -> Path:
    if value is None:
        raise TypeError("checkpoint path must be configured")
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    return path


def _configured_pass_bias(response_config: dict[str, Any]) -> float:
    pass_bias = response_config.get("pass_bias")
    if isinstance(pass_bias, dict):
        return float(pass_bias.get("default", 2.0))
    if pass_bias is not None:
        return float(pass_bias)
    return 2.0


def _configured_no_yaku_call_penalty(response_config: dict[str, Any]) -> float:
    value = response_config.get("no_yaku_call_penalty", 0.0)
    return max(0.0, float(value))
