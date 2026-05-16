from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mjai.features.action_features import RESPONSE_ACTION_DIM, TURN_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM
from mjai.models.multitask_policy_net import MultitaskMahjongPolicyNet
from mjai.models.policy_net import MahjongPolicyNet
from mjai.serving.schemas import LoadedPolicy
from mjai.utils.device import normalize_device_name


def load_policy_checkpoint(
    path: str | Path,
    *,
    task: str | None = None,
    device: str | None = None,
) -> LoadedPolicy:
    checkpoint_path = Path(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"checkpoint config is missing or invalid: {checkpoint_path}")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise TypeError(f"checkpoint model_state_dict is missing or invalid: {checkpoint_path}")

    checkpoint_task = str(config.get("task", {}).get("task", ""))
    selected_task = _resolve_task_name(checkpoint_task=checkpoint_task, requested_task=task)
    resolved_device = _resolve_device(device if device is not None else config.get("train", {}).get("device"))
    model_config = config.get("model", {})
    hidden_size = int(model_config.get("hidden_size", 512 if checkpoint_task == "multitask" else 128))
    dropout = float(model_config.get("dropout", 0.1))
    candidate_limit = _candidate_limit_for_task(config, selected_task)
    default_pass_bias = _default_pass_bias(config, selected_task)

    if checkpoint_task == "multitask":
        model = MultitaskMahjongPolicyNet(hidden_size=hidden_size, dropout=dropout).to(resolved_device)
        state_dim = TURN_STATE_DIM if selected_task == "turn" else RESPONSE_STATE_DIM
        action_dim = TURN_ACTION_DIM if selected_task == "turn" else RESPONSE_ACTION_DIM
    else:
        state_dim = TURN_STATE_DIM if selected_task == "turn" else RESPONSE_STATE_DIM
        action_dim = TURN_ACTION_DIM if selected_task == "turn" else RESPONSE_ACTION_DIM
        model = MahjongPolicyNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        ).to(resolved_device)

    model.load_state_dict(model_state_dict)
    model.eval()

    return LoadedPolicy(
        task=selected_task,
        checkpoint_path=checkpoint_path,
        checkpoint_task=checkpoint_task,
        device=resolved_device,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        dropout=dropout,
        candidate_limit=candidate_limit,
        default_pass_bias=default_pass_bias,
        model=model,
        config=config,
    )


def _resolve_task_name(*, checkpoint_task: str, requested_task: str | None) -> str:
    if requested_task is not None:
        if requested_task not in {"turn", "response"}:
            raise ValueError(f"task must be turn or response, got: {requested_task}")
        if checkpoint_task in {"turn", "response"} and checkpoint_task != requested_task:
            raise ValueError(
                f"checkpoint task {checkpoint_task} does not match requested task {requested_task}"
            )
        return requested_task
    if checkpoint_task not in {"turn", "response"}:
        raise ValueError(
            "multitask checkpoints require an explicit task name of turn or response"
        )
    return checkpoint_task


def _resolve_device(config_device: str | None) -> torch.device:
    normalized = normalize_device_name(config_device)
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized in {"auto", "xpu"} and _xpu_is_available():
        return torch.device("xpu")
    if normalized == "xpu":
        raise RuntimeError("device is xpu, but torch.xpu is not available")
    return torch.device("cpu")


def _xpu_is_available() -> bool:
    xpu = getattr(torch, "xpu", None)
    is_available = getattr(xpu, "is_available", None)
    return bool(callable(is_available) and is_available())


def _candidate_limit_for_task(config: dict[str, Any], task: str) -> int:
    data_config = config.get("data", {})
    task_key = f"{task}_candidate_limit"
    if task_key in data_config:
        return int(data_config[task_key])
    if "candidate_limit" in data_config:
        return int(data_config["candidate_limit"])
    return 16 if task == "response" else 64


def _default_pass_bias(config: dict[str, Any], task: str) -> float | None:
    if task != "response":
        return None
    response_config = config.get("models", {}).get("response", {})
    pass_bias = response_config.get("pass_bias")
    if isinstance(pass_bias, dict):
        if "default" in pass_bias:
            return float(pass_bias["default"])
        if pass_bias:
            first_key = next(iter(pass_bias))
            return float(pass_bias[first_key])
        return 0.3
    if pass_bias is not None:
        return float(pass_bias)
    train_config = config.get("train", {})
    if "response_pass_bias" in train_config:
        return float(train_config["response_pass_bias"])
    return 0.3
