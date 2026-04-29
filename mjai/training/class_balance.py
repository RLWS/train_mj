from __future__ import annotations

from typing import Any

import torch


MAJORITY_LABEL_BY_TASK = {
    "turn": "discard",
    "response": "pass",
}

NON_MAJORITY_GROUP_BY_TASK = {
    "turn": "non_discard",
    "response": "non_pass",
}


def class_weight_map(task_name: str, train_config: dict[str, Any]) -> dict[str, float]:
    weight_key = f"{task_name}_class_weights"
    raw_weights = train_config.get(weight_key, {})
    if raw_weights is None:
        return {}
    if not isinstance(raw_weights, dict):
        raise TypeError(f"train.{weight_key} must be a mapping, got: {type(raw_weights).__name__}")
    return {str(label_kind): float(weight) for label_kind, weight in raw_weights.items()}


def build_sample_weights(
    label_kinds: list[str],
    weights: dict[str, float],
    device: torch.device,
) -> torch.Tensor | None:
    if not weights:
        return None
    return torch.tensor(
        [float(weights.get(label_kind, 1.0)) for label_kind in label_kinds],
        dtype=torch.float32,
        device=device,
    )


def class_group_name(task_name: str, label_kind: str) -> str:
    majority_label = MAJORITY_LABEL_BY_TASK.get(task_name)
    if majority_label is None:
        return label_kind
    if label_kind == majority_label:
        return majority_label
    return NON_MAJORITY_GROUP_BY_TASK[task_name]

