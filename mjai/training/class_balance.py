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
    """读取指定任务的类别权重配置。
    
    参数:
        task_name (str): 任务名称。
        train_config (dict[str, Any]): 训练配置字典。
    
    返回:
        dict[str, float]: 字典结果。
    
    异常:
        TypeError: 当输入类型不符合要求时抛出。
    """
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
    """根据标签类型构建样本权重张量。
    
    参数:
        label_kinds (list[str]): 标签动作类型列表。
        weights (dict[str, float]): 相关数据列表。
        device (torch.device): 计算设备。
    
    返回:
        torch.Tensor | None: 张量结果。
    
    异常:
        无。
    """
    if not weights:
        return None
    return torch.tensor(
        [float(weights.get(label_kind, 1.0)) for label_kind in label_kinds],
        dtype=torch.float32,
        device=device,
    )


def class_group_name(task_name: str, label_kind: str) -> str:
    """将标签类型映射到类别分组名称。
    
    参数:
        task_name (str): 任务名称。
        label_kind (str): 标签动作类型。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    majority_label = MAJORITY_LABEL_BY_TASK.get(task_name)
    if majority_label is None:
        return label_kind
    if label_kind == majority_label:
        return majority_label
    return NON_MAJORITY_GROUP_BY_TASK[task_name]

