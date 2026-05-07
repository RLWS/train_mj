from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset

from mjai.data.datasets.parquet_dataset import (
    ManifestResponseDataset,
    ManifestTurnDataset,
    RESPONSE_LABEL_KIND_ORDER,
    ResponseBatchCollator,
    StreamingManifestResponseDataset,
    StreamingManifestTurnDataset,
    TURN_LABEL_KIND_ORDER,
    TurnBatchCollator,
)
from mjai.features.action_features import RESPONSE_ACTION_DIM, TURN_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM
from mjai.models.policy_net import MahjongPolicyNet
from mjai.training.class_balance import build_sample_weights, class_group_name, class_weight_map
from mjai.training.losses import policy_loss
from mjai.training.metrics import (
    ResponseDiagnostics,
    ResponseMetricsAccumulator,
    response_selection_score,
)
from mjai.utils.config import dump_yaml
from mjai.utils.device import normalize_device_name
from mjai.utils.paths import ensure_dir
from mjai.utils.seed import seed_everything


@dataclass(slots=True, frozen=True)
class LabelMetricSummary:
    samples: int
    loss: float
    accuracy: float


@dataclass(slots=True, frozen=True)
class EpochSummary:
    epoch: int
    split: str
    loss: float
    accuracy: float
    macro_accuracy: float
    batches: int
    samples: int
    per_kind: dict[str, LabelMetricSummary]
    class_groups: dict[str, LabelMetricSummary]
    response: ResponseDiagnostics | None = None


@dataclass(slots=True, frozen=True)
class TrainSummary:
    task: str
    epochs: int
    train_loss: float
    train_accuracy: float
    train_macro_accuracy: float
    valid_loss: float | None
    valid_accuracy: float | None
    valid_macro_accuracy: float | None
    train_samples: int
    valid_samples: int
    original_train_samples: int
    filtered_train_samples: int
    best_checkpoint: str
    best_response_checkpoint: str | None
    best_loss_checkpoint: str
    best_macro_checkpoint: str
    final_checkpoint: str
    used_device: str


def train_turn_model(config: dict[str, Any]) -> TrainSummary:
    """训练自摸巡或响应候选排序策略模型。
    
    参数:
        config (dict[str, Any]): 配置字典。
    
    返回:
        TrainSummary: 函数执行结果。
    
    异常:
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    task_name = str(config.get("task", {}).get("task", ""))
    if task_name not in {"turn", "response"}:
        raise ValueError(
            f"minimal training path currently supports only task=turn/response, got: {task_name}"
        )
    seed_everything(int(config.get("seed", 20260423)))

    output_dir = ensure_dir(config.get("output_dir", "outputs/checkpoints/turn_smoke"))
    log_dir = ensure_dir(config.get("log_dir", "outputs/logs/turn_smoke"))
    dump_yaml(output_dir / "config.resolved.yaml", config)
    print(
        "[train] preparing datasets "
        f"task={task_name} "
        f"train_manifest={config['data']['train_manifest']} "
        f"valid_manifest={config.get('data', {}).get('valid_manifest')}",
        flush=True,
    )

    train_dataset = _build_dataset(
        task_name,
        config["data"]["train_manifest"],
        train_config=config.get("train", {}),
        split="train",
    )
    valid_manifest = config.get("data", {}).get("valid_manifest")
    valid_dataset = (
        _build_dataset(
            task_name,
            valid_manifest,
            train_config=config.get("train", {}),
            split="valid",
        )
        if valid_manifest and Path(valid_manifest).exists()
        else None
    )
    candidate_limit = int(config.get("data", {}).get("candidate_limit", 64))
    collate_fn = _build_collate_fn(task_name, candidate_limit)
    train_config = config.get("train", {})
    device = _resolve_training_device(train_config.get("device"))
    precision_config = _resolve_precision_config(train_config, device)
    _configure_matmul_precision(train_config, device)
    train_num_workers = int(train_config.get("num_workers", 0))
    valid_num_workers = int(train_config.get("valid_num_workers", 0))
    pin_memory = bool(train_config.get("pin_memory", device.type == "cuda"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_size", 32)),
        shuffle=_dataloader_shuffle(train_dataset, train=True),
        num_workers=train_num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        **_worker_loader_kwargs(train_config, train_num_workers),
    )
    valid_loader = None
    if valid_dataset is not None and len(valid_dataset) > 0:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(train_config.get("batch_size", 32)),
            shuffle=False,
            num_workers=valid_num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            **_worker_loader_kwargs(train_config, valid_num_workers),
        )

    model = MahjongPolicyNet(
        state_dim=TURN_STATE_DIM if task_name == "turn" else RESPONSE_STATE_DIM,
        action_dim=TURN_ACTION_DIM if task_name == "turn" else RESPONSE_ACTION_DIM,
        hidden_size=int(config.get("model", {}).get("hidden_size", 128)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("train", {}).get("lr", 1e-3)),
        weight_decay=float(config.get("train", {}).get("weight_decay", 0.0)),
    )
    weight_map = class_weight_map(task_name, train_config)
    log_every_n_steps = int(train_config.get("log_every_n_steps", 0))

    epochs = int(config.get("train", {}).get("epochs", 1))
    _print_training_start(
        task_name=task_name,
        device=device,
        output_dir=output_dir,
        log_dir=log_dir,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        train_loader=train_loader,
        valid_loader=valid_loader,
        train_config=train_config,
        model_config=config.get("model", {}),
        candidate_limit=candidate_limit,
        pin_memory=pin_memory,
        train_num_workers=train_num_workers,
        valid_num_workers=valid_num_workers,
        log_every_n_steps=log_every_n_steps,
        epochs=epochs,
        precision_config=precision_config,
    )
    epoch_summaries: list[dict[str, Any]] = []
    best_valid_accuracy = float("-inf")
    best_valid_loss = float("inf")
    best_valid_macro_accuracy = float("-inf")
    best_response_selection_score = float("-inf")
    best_checkpoint_path = output_dir / "best.pt"
    best_accuracy_checkpoint_path = output_dir / "best_accuracy.pt"
    best_loss_checkpoint_path = output_dir / "best_loss.pt"
    best_macro_checkpoint_path = output_dir / "best_macro.pt"
    best_response_checkpoint_path = output_dir / "best_response.pt"
    final_checkpoint_path = output_dir / "final.pt"

    last_train_summary: EpochSummary | None = None
    last_valid_summary: EpochSummary | None = None

    for epoch in range(1, epochs + 1):
        last_train_summary = _run_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            split="train",
            task_name=task_name,
            train=True,
            weight_map=weight_map,
            log_every_n_steps=log_every_n_steps,
            precision_config=precision_config,
            collect_response_diagnostics=_collect_response_diagnostics(
                task_name,
                train_config,
                split="train",
            ),
        )
        epoch_summaries.append(asdict(last_train_summary))

        if valid_loader is not None:
            last_valid_summary = _run_epoch(
                model=model,
                data_loader=valid_loader,
                optimizer=None,
                device=device,
                epoch=epoch,
                split="valid",
                task_name=task_name,
                train=False,
                weight_map=weight_map,
                log_every_n_steps=0,
                precision_config=precision_config,
                collect_response_diagnostics=_collect_response_diagnostics(
                    task_name,
                    train_config,
                    split="valid",
                ),
            )
            epoch_summaries.append(asdict(last_valid_summary))

            if last_valid_summary.accuracy >= best_valid_accuracy:
                best_valid_accuracy = last_valid_summary.accuracy
                _save_checkpoint(
                    best_accuracy_checkpoint_path,
                    model,
                    optimizer,
                    config,
                    last_valid_summary,
                )
                if task_name != "response":
                    _save_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        config,
                        last_valid_summary,
                    )
            if last_valid_summary.loss <= best_valid_loss:
                best_valid_loss = last_valid_summary.loss
                _save_checkpoint(best_loss_checkpoint_path, model, optimizer, config, last_valid_summary)
            if last_valid_summary.macro_accuracy >= best_valid_macro_accuracy:
                best_valid_macro_accuracy = last_valid_summary.macro_accuracy
                _save_checkpoint(best_macro_checkpoint_path, model, optimizer, config, last_valid_summary)
            if task_name == "response":
                current_response_score = response_selection_score(last_valid_summary.response)
                if (
                    current_response_score is not None
                    and current_response_score >= best_response_selection_score
                ):
                    best_response_selection_score = current_response_score
                    _save_checkpoint(
                        best_response_checkpoint_path,
                        model,
                        optimizer,
                        config,
                        last_valid_summary,
                    )
                    _save_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        config,
                        last_valid_summary,
                    )
        else:
            _save_checkpoint(best_checkpoint_path, model, optimizer, config, last_train_summary)
            _save_checkpoint(best_accuracy_checkpoint_path, model, optimizer, config, last_train_summary)
            _save_checkpoint(best_loss_checkpoint_path, model, optimizer, config, last_train_summary)
            _save_checkpoint(best_macro_checkpoint_path, model, optimizer, config, last_train_summary)
            if task_name == "response":
                _save_checkpoint(best_response_checkpoint_path, model, optimizer, config, last_train_summary)

    assert last_train_summary is not None
    _save_checkpoint(final_checkpoint_path, model, optimizer, config, last_train_summary)
    observed_train_samples = int(last_train_summary.samples)
    observed_valid_samples = 0 if last_valid_summary is None else int(last_valid_summary.samples)
    original_train_samples = _dataset_original_rows(train_dataset)
    filtered_train_samples = _observed_filtered_rows(train_dataset, observed_train_samples)

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "epoch_summaries": epoch_summaries,
                "train_samples": observed_train_samples,
                "valid_samples": observed_valid_samples,
                "original_train_samples": original_train_samples,
                "filtered_train_samples": filtered_train_samples,
                "best_response_selection_score": (
                    None
                    if task_name != "response"
                    or best_response_selection_score == float("-inf")
                    else best_response_selection_score
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (log_dir / "epoch_metrics.jsonl").write_text(
        "".join(json.dumps(summary, ensure_ascii=False) + "\n" for summary in epoch_summaries),
        encoding="utf-8",
    )

    return TrainSummary(
        task=task_name,
        epochs=epochs,
        train_loss=last_train_summary.loss,
        train_accuracy=last_train_summary.accuracy,
        train_macro_accuracy=last_train_summary.macro_accuracy,
        valid_loss=None if last_valid_summary is None else last_valid_summary.loss,
        valid_accuracy=None if last_valid_summary is None else last_valid_summary.accuracy,
        valid_macro_accuracy=None if last_valid_summary is None else last_valid_summary.macro_accuracy,
        train_samples=observed_train_samples,
        valid_samples=observed_valid_samples,
        original_train_samples=original_train_samples,
        filtered_train_samples=filtered_train_samples,
        best_checkpoint=str(best_checkpoint_path),
        best_response_checkpoint=(
            str(best_response_checkpoint_path) if task_name == "response" else None
        ),
        best_loss_checkpoint=str(best_loss_checkpoint_path),
        best_macro_checkpoint=str(best_macro_checkpoint_path),
        final_checkpoint=str(final_checkpoint_path),
        used_device=str(device),
    )


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: AdamW | None,
    device: torch.device,
    epoch: int,
    split: str,
    task_name: str,
    train: bool,
    weight_map: dict[str, float],
    log_every_n_steps: int,
    precision_config: dict[str, Any],
    collect_response_diagnostics: bool,
) -> EpochSummary:
    """运行与函数名对应的处理流程。
    
    参数:
        model (nn.Module): 模型对象。
        data_loader (DataLoader): 数据加载器。
        optimizer (AdamW | None): 优化器对象。
        device (torch.device): 计算设备。
        epoch (int): 训练轮次编号。
        split (str): 数据划分名称。
        task_name (str): 任务名称。
        train (bool): 是否处于训练模式。
        weight_map (dict[str, float]): 函数执行所需的输入参数。
        log_every_n_steps (int): 相关数据列表。
        precision_config (dict[str, Any]): 精度配置字典。
        collect_response_diagnostics (bool): 相关数据列表。
    
    返回:
        EpochSummary: 进程退出码。
    
    异常:
        无。
    """
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batches = 0
    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    running_time = 0.0
    running_data_wait_time = 0.0
    running_compute_time = 0.0
    kind_totals: dict[str, int] = {}
    kind_correct: dict[str, int] = {}
    kind_loss_totals: dict[str, float] = {}
    group_totals: dict[str, int] = {}
    group_correct: dict[str, int] = {}
    group_loss_totals: dict[str, float] = {}
    response_metrics = (
        ResponseMetricsAccumulator()
        if task_name == "response" and collect_response_diagnostics
        else None
    )

    model.train(train)
    total_batches = len(data_loader)
    epoch_start = time.perf_counter()
    next_batch_start = epoch_start
    for batch in data_loader:
        batch_ready = time.perf_counter()
        data_wait_elapsed = batch_ready - next_batch_start
        batch_start = batch_ready
        batches += 1
        prepared_batch = {
            key: value.to(device, non_blocking=device.type == "cuda")
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

        with _autocast_context(device, precision_config):
            if task_name == "turn":
                logits = model.forward_turn(prepared_batch)
            else:
                logits = model.forward_response(prepared_batch)
            sample_weights = build_sample_weights(prepared_batch["label_kinds"], weight_map, device)
            loss = policy_loss(logits, prepared_batch["labels"], sample_weights=sample_weights)
            per_sample_losses = F.cross_entropy(logits, prepared_batch["labels"], reduction="none")

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = prepared_batch["labels"].shape[0]
        predictions = logits.argmax(dim=1)
        label_ranks = (
            _compute_label_ranks(logits, prepared_batch["labels"])
            if response_metrics is not None
            else None
        )
        correct = predictions == prepared_batch["labels"]
        loss_value = float(loss.detach().item())
        if response_metrics is not None:
            assert label_ranks is not None
            correct_list = correct.detach().to("cpu").tolist()
            per_sample_loss_list = per_sample_losses.detach().to("cpu").tolist()
            prediction_list = predictions.detach().to("cpu").tolist()
            label_rank_list = label_ranks.detach().to("cpu").tolist()
        else:
            correct_list = []
            per_sample_loss_list = []
            prediction_list = []
            label_rank_list = []
        batch_correct = int(correct.sum().detach().item())
        total_loss += loss_value * batch_size
        total_correct += batch_correct
        total_samples += batch_size
        compute_elapsed = time.perf_counter() - batch_start
        batch_elapsed = data_wait_elapsed + compute_elapsed
        running_loss += loss_value * batch_size
        running_correct += batch_correct
        running_samples += batch_size
        running_time += batch_elapsed
        running_data_wait_time += data_wait_elapsed
        running_compute_time += compute_elapsed
        if response_metrics is None:
            _update_fast_label_summaries(
                task_name=task_name,
                label_kind_ids=prepared_batch["label_kind_ids"],
                correct=correct,
                per_sample_losses=per_sample_losses,
                kind_totals=kind_totals,
                kind_correct=kind_correct,
                kind_loss_totals=kind_loss_totals,
                group_totals=group_totals,
                group_correct=group_correct,
                group_loss_totals=group_loss_totals,
            )
        else:
            for batch_index, label_kind in enumerate(prepared_batch["label_kinds"]):
                is_correct = int(correct_list[batch_index])
                sample_loss = float(per_sample_loss_list[batch_index])
                kind_totals[label_kind] = kind_totals.get(label_kind, 0) + 1
                kind_correct[label_kind] = kind_correct.get(label_kind, 0) + is_correct
                kind_loss_totals[label_kind] = kind_loss_totals.get(label_kind, 0.0) + sample_loss

                group_name = class_group_name(task_name, label_kind)
                group_totals[group_name] = group_totals.get(group_name, 0) + 1
                group_correct[group_name] = group_correct.get(group_name, 0) + is_correct
                group_loss_totals[group_name] = group_loss_totals.get(group_name, 0.0) + sample_loss
                legal_actions = prepared_batch["legal_actions"][batch_index]
                predicted_index = int(prediction_list[batch_index])
                predicted_action = legal_actions[predicted_index]
                response_metrics.update(
                    label_kind=label_kind,
                    predicted_kind=str(predicted_action.get("kind", "unknown")),
                    candidate_kinds=[str(action.get("kind", "unknown")) for action in legal_actions],
                    candidate_count=int(prepared_batch["candidate_counts"][batch_index]),
                    label_rank=int(label_rank_list[batch_index]),
                )

        if (
            train
            and log_every_n_steps > 0
            and (batches == 1 or batches % log_every_n_steps == 0 or batches == total_batches)
        ):
            _print_batch_progress(
                epoch=epoch,
                split=split,
                batch_index=batches,
                total_batches=total_batches,
                total_samples=total_samples,
                batch_samples=batch_size,
                batch_elapsed=batch_elapsed,
                data_wait_elapsed=data_wait_elapsed,
                compute_elapsed=compute_elapsed,
                running_time=running_time,
                running_data_wait_time=running_data_wait_time,
                running_compute_time=running_compute_time,
                running_loss=running_loss,
                running_correct=running_correct,
                running_samples=running_samples,
                epoch_start=epoch_start,
            )
            running_loss = 0.0
            running_correct = 0
            running_samples = 0
            running_time = 0.0
            running_data_wait_time = 0.0
            running_compute_time = 0.0
        next_batch_start = time.perf_counter()

    average_loss = 0.0 if total_samples == 0 else total_loss / total_samples
    accuracy = 0.0 if total_samples == 0 else total_correct / total_samples
    per_kind = _build_label_metric_summary(kind_totals, kind_correct, kind_loss_totals)
    class_groups = _build_label_metric_summary(group_totals, group_correct, group_loss_totals)
    return EpochSummary(
        epoch=epoch,
        split=split,
        loss=average_loss,
        accuracy=accuracy,
        macro_accuracy=_macro_accuracy(per_kind),
        batches=batches,
        samples=total_samples,
        per_kind=per_kind,
        class_groups=class_groups,
        response=None if response_metrics is None else response_metrics.summary(),
    )


def _resolve_training_device(config_device: str | None) -> torch.device:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        config_device (str | None): 函数执行所需的输入参数。
    
    返回:
        torch.device: 函数执行结果。
    
    异常:
        RuntimeError: 当运行环境或后端状态不满足要求时抛出。
    """
    normalized = normalize_device_name(config_device)
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized in {"auto", "xpu"} and _xpu_is_available():
        return torch.device("xpu")
    if normalized == "xpu":
        raise RuntimeError("train.device is xpu, but torch.xpu is not available")
    return torch.device("cpu")


def _compute_label_ranks(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """执行与函数名对应的业务逻辑。
    
    参数:
        logits (torch.Tensor): 候选动作分数张量。
        labels (torch.Tensor): 标签索引张量。
    
    返回:
        torch.Tensor: 张量结果。
    
    异常:
        无。
    """
    label_scores = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    higher_scores = (logits > label_scores.unsqueeze(1)).sum(dim=1)
    return higher_scores + 1


def _update_fast_label_summaries(
    *,
    task_name: str,
    label_kind_ids: torch.Tensor,
    correct: torch.Tensor,
    per_sample_losses: torch.Tensor,
    kind_totals: dict[str, int],
    kind_correct: dict[str, int],
    kind_loss_totals: dict[str, float],
    group_totals: dict[str, int],
    group_correct: dict[str, int],
    group_loss_totals: dict[str, float],
) -> None:
    """更新统计或状态累积结果。
    
    参数:
        task_name (str): 任务名称。
        label_kind_ids (torch.Tensor): 相关数据列表。
        correct (torch.Tensor): 函数执行所需的输入参数。
        per_sample_losses (torch.Tensor): 相关数据列表。
        kind_totals (dict[str, int]): 相关数据列表。
        kind_correct (dict[str, int]): 函数执行所需的输入参数。
        kind_loss_totals (dict[str, float]): 相关数据列表。
        group_totals (dict[str, int]): 相关数据列表。
        group_correct (dict[str, int]): 函数执行所需的输入参数。
        group_loss_totals (dict[str, float]): 相关数据列表。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    kind_order = _label_kind_order(task_name)
    label_kind_ids_cpu = label_kind_ids.detach().to("cpu", dtype=torch.long)
    correct_cpu = correct.detach().to("cpu", dtype=torch.float32)
    losses_cpu = per_sample_losses.detach().to("cpu", dtype=torch.float32)
    minlength = len(kind_order)

    kind_count_values = torch.bincount(label_kind_ids_cpu, minlength=minlength).tolist()
    kind_correct_values = torch.bincount(
        label_kind_ids_cpu,
        weights=correct_cpu,
        minlength=minlength,
    ).tolist()
    kind_loss_values = torch.bincount(
        label_kind_ids_cpu,
        weights=losses_cpu,
        minlength=minlength,
    ).tolist()
    for kind_index, label_kind in enumerate(kind_order):
        count = int(kind_count_values[kind_index])
        if count == 0:
            continue
        correct_count = int(kind_correct_values[kind_index])
        loss_total = float(kind_loss_values[kind_index])
        kind_totals[label_kind] = kind_totals.get(label_kind, 0) + count
        kind_correct[label_kind] = kind_correct.get(label_kind, 0) + correct_count
        kind_loss_totals[label_kind] = kind_loss_totals.get(label_kind, 0.0) + loss_total

        group_name = class_group_name(task_name, label_kind)
        group_totals[group_name] = group_totals.get(group_name, 0) + count
        group_correct[group_name] = group_correct.get(group_name, 0) + correct_count
        group_loss_totals[group_name] = group_loss_totals.get(group_name, 0.0) + loss_total


def _label_kind_order(task_name: str) -> tuple[str, ...]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        task_name (str): 任务名称。
    
    返回:
        tuple[str, ...]: 元组结果。
    
    异常:
        无。
    """
    if task_name == "response":
        return RESPONSE_LABEL_KIND_ORDER
    return TURN_LABEL_KIND_ORDER


def _xpu_is_available() -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        无。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    xpu = getattr(torch, "xpu", None)
    is_available = getattr(xpu, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return bool(is_available())
    except RuntimeError:
        return False


def _resolve_precision_config(train_config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        train_config (dict[str, Any]): 训练配置字典。
        device (torch.device): 计算设备。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    precision = str(train_config.get("precision", "fp32")).lower()
    if precision in {"float32", "32", "full"}:
        precision = "fp32"
    if precision in {"float16", "16"}:
        precision = "fp16"
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"unsupported train.precision: {precision}")

    autocast_enabled = device.type in {"cuda", "xpu"} and precision in {"fp16", "bf16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return {
        "precision": precision,
        "autocast_enabled": autocast_enabled,
        "autocast_dtype": autocast_dtype,
    }


def _autocast_context(device: torch.device, precision_config: dict[str, Any]):
    """执行与函数名对应的业务逻辑。
    
    参数:
        device (torch.device): 计算设备。
        precision_config (dict[str, Any]): 精度配置字典。
    
    返回:
        Any: 函数执行结果。
    
    异常:
        无。
    """
    if not bool(precision_config.get("autocast_enabled", False)):
        return nullcontext()
    return torch.autocast(
        device_type=device.type,
        dtype=precision_config["autocast_dtype"],
    )


def _configure_matmul_precision(train_config: dict[str, Any], device: torch.device) -> None:
    """执行与函数名对应的业务逻辑。
    
    参数:
        train_config (dict[str, Any]): 训练配置字典。
        device (torch.device): 计算设备。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    if device.type != "cuda":
        return
    matmul_precision = str(train_config.get("float32_matmul_precision", "high")).lower()
    if matmul_precision in {"highest", "high", "medium"}:
        torch.set_float32_matmul_precision(matmul_precision)
    allow_tf32 = bool(train_config.get("allow_tf32", True))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    config: dict[str, Any],
    summary: EpochSummary,
) -> None:
    """执行与函数名对应的业务逻辑。
    
    参数:
        path (Path): 文件或目录路径。
        model (nn.Module): 模型对象。
        optimizer (AdamW): 优化器对象。
        config (dict[str, Any]): 配置字典。
        summary (EpochSummary): 函数执行所需的输入参数。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "summary": asdict(summary),
        },
        path,
    )


def _build_dataset(
    task_name: str,
    manifest_path: str | Path,
    train_config: dict[str, Any] | None = None,
    split: str | None = None,
):
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        task_name (str): 任务名称。
        manifest_path (str | Path): 清单文件路径。
        train_config (dict[str, Any] | None): 训练配置字典。
        split (str | None): 数据划分名称。
    
    返回:
        Any: 函数执行结果。
    
    异常:
        无。
    """
    train_config = train_config or {}
    if task_name == "turn":
        if bool(train_config.get("streaming_dataset", True)):
            shuffle = split == "train"
            return StreamingManifestTurnDataset(
                manifest_path,
                shuffle_shards=shuffle and bool(train_config.get("shuffle_shards", True)),
                shuffle_rows=shuffle and bool(train_config.get("shuffle_rows", True)),
                shuffle_buffer_size=int(train_config.get("shuffle_buffer_size", 8192)),
                seed=int(train_config.get("dataset_seed", 20260423)),
            )
        return ManifestTurnDataset(manifest_path)
    response_min_candidate_count = _response_min_candidate_count(train_config, split)
    if bool(train_config.get("streaming_dataset", True)):
        shuffle = split == "train"
        return StreamingManifestResponseDataset(
            manifest_path,
            min_candidate_count=response_min_candidate_count,
            shuffle_shards=shuffle and bool(train_config.get("shuffle_shards", True)),
            shuffle_rows=shuffle and bool(train_config.get("shuffle_rows", True)),
            shuffle_buffer_size=int(train_config.get("shuffle_buffer_size", 8192)),
            include_legal_actions=_collect_response_diagnostics(
                task_name,
                train_config,
                split=split,
            ),
            seed=int(train_config.get("dataset_seed", 20260423)),
        )
    return ManifestResponseDataset(
        manifest_path,
        min_candidate_count=response_min_candidate_count,
    )


def _dataloader_shuffle(dataset, train: bool) -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        dataset (Any): 数据集对象。
        train (bool): 是否处于训练模式。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if not train or isinstance(dataset, IterableDataset):
        return False
    return True


def _response_min_candidate_count(
    train_config: dict[str, Any],
    split: str | None,
) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        train_config (dict[str, Any]): 训练配置字典。
        split (str | None): 数据划分名称。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    if split != "train":
        return 1
    return int(train_config.get("response_min_train_candidate_count", 1))


def _collect_response_diagnostics(
    task_name: str,
    train_config: dict[str, Any],
    split: str | None,
) -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        task_name (str): 任务名称。
        train_config (dict[str, Any]): 训练配置字典。
        split (str | None): 数据划分名称。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if task_name != "response":
        return False
    if split == "train":
        return bool(train_config.get("response_train_diagnostics", False))
    return bool(train_config.get("response_eval_diagnostics", True))


def _build_collate_fn(task_name: str, candidate_limit: int):
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        task_name (str): 任务名称。
        candidate_limit (int): 批次允许的最大候选数量。
    
    返回:
        Any: 函数执行结果。
    
    异常:
        无。
    """
    if task_name == "turn":
        return TurnBatchCollator(candidate_limit=candidate_limit)
    return ResponseBatchCollator(candidate_limit=candidate_limit)


def _dataset_original_rows(dataset) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        dataset (Any): 数据集对象。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    return int(getattr(dataset, "original_rows", len(dataset)))


def _dataset_filtered_rows(dataset) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        dataset (Any): 数据集对象。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    return int(getattr(dataset, "filtered_rows", 0))


def _observed_filtered_rows(dataset, observed_rows: int) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        dataset (Any): 数据集对象。
        observed_rows (int): 相关数据列表。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    filtered_rows = _dataset_filtered_rows(dataset)
    if filtered_rows > 0:
        return filtered_rows
    original_rows = _dataset_original_rows(dataset)
    if original_rows > observed_rows:
        return original_rows - observed_rows
    return 0


def _print_training_start(
    task_name: str,
    device: torch.device,
    output_dir: Path,
    log_dir: Path,
    train_dataset,
    valid_dataset,
    train_loader: DataLoader,
    valid_loader: DataLoader | None,
    train_config: dict[str, Any],
    model_config: dict[str, Any],
    candidate_limit: int,
    pin_memory: bool,
    train_num_workers: int,
    valid_num_workers: int,
    log_every_n_steps: int,
    epochs: int,
    precision_config: dict[str, Any],
) -> None:
    """打印与函数名对应的进度或诊断信息。
    
    参数:
        task_name (str): 任务名称。
        device (torch.device): 计算设备。
        output_dir (Path): 输出目录路径。
        log_dir (Path): 相关目录路径。
        train_dataset (Any): 函数执行所需的输入参数。
        valid_dataset (Any): 函数执行所需的输入参数。
        train_loader (DataLoader): 函数执行所需的输入参数。
        valid_loader (DataLoader | None): 函数执行所需的输入参数。
        train_config (dict[str, Any]): 训练配置字典。
        model_config (dict[str, Any]): 模型配置字典。
        candidate_limit (int): 批次允许的最大候选数量。
        pin_memory (bool): 函数执行所需的输入参数。
        train_num_workers (int): 相关数据列表。
        valid_num_workers (int): 相关数据列表。
        log_every_n_steps (int): 相关数据列表。
        epochs (int): 相关数据列表。
        precision_config (dict[str, Any]): 精度配置字典。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    valid_samples = 0 if valid_dataset is None else len(valid_dataset)
    filtered_train_samples = _dataset_filtered_rows(train_dataset)
    original_train_samples = _dataset_original_rows(train_dataset)
    valid_batches = 0 if valid_loader is None else len(valid_loader)
    print(
        "[train] starting "
        f"task={task_name} "
        f"device={device} "
        f"cuda_available={torch.cuda.is_available()} "
        f"xpu_available={_xpu_is_available()} "
        f"epochs={epochs} "
        f"train_samples={len(train_dataset)} "
        f"original_train_samples={original_train_samples} "
        f"filtered_train_samples={filtered_train_samples} "
        f"valid_samples={valid_samples} "
        f"train_batches={len(train_loader)} "
        f"valid_batches={valid_batches} "
        f"batch_size={int(train_config.get('batch_size', 32))} "
        f"candidate_limit={candidate_limit} "
        f"num_workers={train_num_workers} "
        f"valid_num_workers={valid_num_workers} "
        f"pin_memory={pin_memory} "
        f"precision={precision_config['precision']} "
        f"autocast={bool(precision_config['autocast_enabled'])} "
        f"autocast_dtype={_format_autocast_dtype(precision_config)} "
        f"allow_tf32={_cuda_tf32_enabled()} "
        f"float32_matmul_precision={torch.get_float32_matmul_precision()} "
        f"persistent_workers={bool(train_config.get('persistent_workers', True)) if train_num_workers > 0 else False} "
        f"prefetch_factor={int(train_config.get('prefetch_factor', 2)) if train_num_workers > 0 else 0} "
        f"hidden_size={int(model_config.get('hidden_size', 128))} "
        f"dropout={float(model_config.get('dropout', 0.1))} "
        f"lr={float(train_config.get('lr', 1e-3))} "
        f"weight_decay={float(train_config.get('weight_decay', 0.0))} "
        f"log_every_n_steps={log_every_n_steps} "
        f"response_train_diagnostics={_collect_response_diagnostics(task_name, train_config, 'train')} "
        f"response_eval_diagnostics={_collect_response_diagnostics(task_name, train_config, 'valid')} "
        f"output_dir={output_dir} "
        f"log_dir={log_dir}",
        flush=True,
    )


def _worker_loader_kwargs(train_config: dict[str, Any], num_workers: int) -> dict[str, Any]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        train_config (dict[str, Any]): 训练配置字典。
        num_workers (int): 相关数据列表。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        无。
    """
    if num_workers <= 0:
        return {}
    return {
        "persistent_workers": bool(train_config.get("persistent_workers", True)),
        "prefetch_factor": int(train_config.get("prefetch_factor", 2)),
    }


def _format_autocast_dtype(precision_config: dict[str, Any]) -> str:
    """执行与函数名对应的业务逻辑。
    
    参数:
        precision_config (dict[str, Any]): 精度配置字典。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    if not bool(precision_config.get("autocast_enabled", False)):
        return "none"
    dtype = precision_config.get("autocast_dtype")
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    return str(dtype)


def _cuda_tf32_enabled() -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        无。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if not torch.cuda.is_available():
        return False
    return bool(torch.backends.cuda.matmul.allow_tf32)


def _print_batch_progress(
    epoch: int,
    split: str,
    batch_index: int,
    total_batches: int,
    total_samples: int,
    batch_samples: int,
    batch_elapsed: float,
    data_wait_elapsed: float,
    compute_elapsed: float,
    running_time: float,
    running_data_wait_time: float,
    running_compute_time: float,
    running_loss: float,
    running_correct: int,
    running_samples: int,
    epoch_start: float,
) -> None:
    """打印与函数名对应的进度或诊断信息。
    
    参数:
        epoch (int): 训练轮次编号。
        split (str): 数据划分名称。
        batch_index (int): 相关索引值。
        total_batches (int): 相关数据列表。
        total_samples (int): 相关数据列表。
        batch_samples (int): 相关数据列表。
        batch_elapsed (float): 函数执行所需的输入参数。
        data_wait_elapsed (float): 函数执行所需的输入参数。
        compute_elapsed (float): 函数执行所需的输入参数。
        running_time (float): 函数执行所需的输入参数。
        running_data_wait_time (float): 函数执行所需的输入参数。
        running_compute_time (float): 函数执行所需的输入参数。
        running_loss (float): 相关数据列表。
        running_correct (int): 函数执行所需的输入参数。
        running_samples (int): 相关数据列表。
        epoch_start (float): 函数执行所需的输入参数。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    batches_in_window = max(1.0, running_samples / max(1, batch_samples))
    average_batch_time = running_time / batches_in_window
    average_data_wait_time = running_data_wait_time / batches_in_window
    average_compute_time = running_compute_time / batches_in_window
    samples_per_second = running_samples / running_time if running_time > 0 else 0.0
    window_loss = running_loss / running_samples if running_samples > 0 else 0.0
    window_accuracy = running_correct / running_samples if running_samples > 0 else 0.0
    elapsed = time.perf_counter() - epoch_start
    batches_per_second = batch_index / elapsed if elapsed > 0 else 0.0
    remaining_batches = max(0, total_batches - batch_index)
    eta_seconds = remaining_batches / batches_per_second if batches_per_second > 0 else 0.0
    progress = batch_index / total_batches if total_batches > 0 else 0.0

    print(
        "[train] "
        f"epoch={epoch} split={split} "
        f"batch={batch_index}/{total_batches} ({progress:.1%}) "
        f"samples={total_samples} "
        f"batch_time={batch_elapsed:.3f}s "
        f"data_wait={data_wait_elapsed:.3f}s "
        f"compute_time={compute_elapsed:.3f}s "
        f"avg_batch_time={average_batch_time:.3f}s "
        f"avg_data_wait={average_data_wait_time:.3f}s "
        f"avg_compute_time={average_compute_time:.3f}s "
        f"samples_per_s={samples_per_second:.1f} "
        f"loss={window_loss:.4f} "
        f"accuracy={window_accuracy:.4f} "
        f"eta={_format_duration(eta_seconds)}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
    """执行与函数名对应的业务逻辑。
    
    参数:
        seconds (float): 相关数据列表。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _build_label_metric_summary(
    totals: dict[str, int],
    correct: dict[str, int],
    loss_totals: dict[str, float],
) -> dict[str, LabelMetricSummary]:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        totals (dict[str, int]): 相关数据列表。
        correct (dict[str, int]): 函数执行所需的输入参数。
        loss_totals (dict[str, float]): 相关数据列表。
    
    返回:
        dict[str, LabelMetricSummary]: 字典结果。
    
    异常:
        无。
    """
    return {
        label: LabelMetricSummary(
            samples=count,
            loss=0.0 if count == 0 else loss_totals.get(label, 0.0) / count,
            accuracy=0.0 if count == 0 else correct.get(label, 0) / count,
        )
        for label, count in sorted(totals.items())
    }


def _macro_accuracy(per_kind: dict[str, LabelMetricSummary]) -> float:
    """执行与函数名对应的业务逻辑。
    
    参数:
        per_kind (dict[str, LabelMetricSummary]): 函数执行所需的输入参数。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    if not per_kind:
        return 0.0
    return sum(summary.accuracy for summary in per_kind.values()) / len(per_kind)
