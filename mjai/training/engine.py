from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
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
from mjai.models.multitask_policy_net import MultitaskMahjongPolicyNet
from mjai.models.policy_net import MahjongPolicyNet
from mjai.training.class_balance import build_sample_weights, class_group_name, class_weight_map
from mjai.training.losses import policy_loss
from mjai.training.metrics import (
    ResponseDiagnostics,
    ResponseMetricsAccumulator,
    combine_response_diagnostics,
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


@dataclass
class EpochAccumulator:
    task_name: str
    collect_response_diagnostics: bool
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0
    batches: int = 0
    running_loss: float = 0.0
    running_correct: int = 0
    running_samples: int = 0
    running_time: float = 0.0
    running_data_wait_time: float = 0.0
    running_compute_time: float = 0.0
    kind_totals: dict[str, int] = field(default_factory=dict)
    kind_correct: dict[str, int] = field(default_factory=dict)
    kind_loss_totals: dict[str, float] = field(default_factory=dict)
    group_totals: dict[str, int] = field(default_factory=dict)
    group_correct: dict[str, int] = field(default_factory=dict)
    group_loss_totals: dict[str, float] = field(default_factory=dict)
    response_metrics: ResponseMetricsAccumulator | None = None

    def __post_init__(self) -> None:
        if self.task_name == "response" and self.collect_response_diagnostics:
            self.response_metrics = ResponseMetricsAccumulator()


@dataclass(slots=True, frozen=True)
class BatchStepResult:
    batch_size: int
    batch_correct: int
    loss_value: float
    correct: torch.Tensor
    predictions: torch.Tensor
    per_sample_losses: torch.Tensor
    label_ranks: torch.Tensor | None


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


@dataclass(slots=True, frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str | None
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


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
    if task_name == "multitask":
        return _train_multitask_model(config)
    if task_name not in {"turn", "response"}:
        raise ValueError(
            f"minimal training path currently supports only task=turn/response/multitask, got: {task_name}"
        )
    train_config = config.get("train", {})
    distributed_context = _initialize_distributed_context(train_config)
    seed_everything(int(config.get("seed", 20260423)))

    output_dir = ensure_dir(config.get("output_dir", "outputs/checkpoints/turn_smoke"))
    log_dir = ensure_dir(config.get("log_dir", "outputs/logs/turn_smoke"))
    if distributed_context.is_main:
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
        train_config=train_config,
        split="train",
        distributed_context=distributed_context,
    )
    valid_manifest = config.get("data", {}).get("valid_manifest")
    valid_dataset = (
        _build_dataset(
            task_name,
            valid_manifest,
            train_config=train_config,
            split="valid",
            distributed_context=distributed_context,
        )
        if valid_manifest and Path(valid_manifest).exists()
        else None
    )
    candidate_limit = int(config.get("data", {}).get("candidate_limit", 64))
    collate_fn = _build_collate_fn(task_name, candidate_limit)
    device = distributed_context.device
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
    model = _wrap_distributed_model(model, distributed_context, train_config, task_name=task_name)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("train", {}).get("lr", 1e-3)),
        weight_decay=float(config.get("train", {}).get("weight_decay", 0.0)),
    )
    weight_map = class_weight_map(task_name, train_config)
    log_every_n_steps = int(train_config.get("log_every_n_steps", 0))

    epochs = int(config.get("train", {}).get("epochs", 1))
    if distributed_context.is_main:
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
            distributed_context=distributed_context,
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
            distributed_context=distributed_context,
        )

        if distributed_context.is_main:
            epoch_summaries.append(asdict(last_train_summary))

        if valid_loader is not None:
            last_valid_summary = _run_epoch(
                model=_unwrap_model(model),
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
                distributed_context=distributed_context,
            )
            if distributed_context.is_main:
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
        elif distributed_context.is_main:
            _save_checkpoint(best_checkpoint_path, model, optimizer, config, last_train_summary)
            _save_checkpoint(best_accuracy_checkpoint_path, model, optimizer, config, last_train_summary)
            _save_checkpoint(best_loss_checkpoint_path, model, optimizer, config, last_train_summary)
            _save_checkpoint(best_macro_checkpoint_path, model, optimizer, config, last_train_summary)
            if task_name == "response":
                _save_checkpoint(
                    best_response_checkpoint_path,
                    model,
                    optimizer,
                    config,
                    last_train_summary,
                )
        _distributed_barrier(distributed_context)

    assert last_train_summary is not None
    if distributed_context.is_main:
        _save_checkpoint(final_checkpoint_path, model, optimizer, config, last_train_summary)
    observed_train_samples = int(last_train_summary.samples)
    observed_valid_samples = 0 if last_valid_summary is None else int(last_valid_summary.samples)
    original_train_samples = _dataset_original_rows(train_dataset)
    filtered_train_samples = _observed_filtered_rows(train_dataset, observed_train_samples)

    if distributed_context.is_main:
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
    _shutdown_distributed_context(distributed_context)

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


def _train_multitask_model(config: dict[str, Any]) -> TrainSummary:
    train_config = config.get("train", {})
    distributed_context = _initialize_distributed_context(train_config)
    seed_everything(int(config.get("seed", 20260423)))

    output_dir = ensure_dir(config.get("output_dir", "outputs/checkpoints/multitask_p800_v1_1"))
    log_dir = ensure_dir(config.get("log_dir", "outputs/logs/multitask_p800_v1_1"))
    if distributed_context.is_main:
        dump_yaml(output_dir / "config.resolved.yaml", config)
    data_config = config.get("data", {})
    train_manifest = data_config["train_manifest"]
    valid_manifest = data_config.get("valid_manifest")
    if distributed_context.is_main:
        print(
            "[train] preparing datasets "
            f"task=multitask train_manifest={train_manifest} valid_manifest={valid_manifest}",
            flush=True,
        )

    turn_train_dataset = _build_dataset(
        "turn",
        train_manifest,
        train_config=train_config,
        split="train",
        distributed_context=distributed_context,
    )
    response_train_dataset = _build_dataset(
        "response",
        train_manifest,
        train_config=train_config,
        split="train",
        distributed_context=distributed_context,
    )
    turn_valid_dataset = None
    response_valid_dataset = None
    if valid_manifest and Path(valid_manifest).exists():
        turn_valid_dataset = _build_dataset(
            "turn",
            valid_manifest,
            train_config=train_config,
            split="valid",
            distributed_context=distributed_context,
        )
        response_valid_dataset = _build_dataset(
            "response",
            valid_manifest,
            train_config=train_config,
            split="valid",
            distributed_context=distributed_context,
        )

    device = distributed_context.device
    precision_config = _resolve_precision_config(train_config, device)
    _configure_matmul_precision(train_config, device)
    train_num_workers = int(train_config.get("num_workers", 0))
    valid_num_workers = int(train_config.get("valid_num_workers", 0))
    pin_memory = bool(train_config.get("pin_memory", device.type == "cuda"))
    turn_candidate_limit = _candidate_limit_for_task(config, "turn")
    response_candidate_limit = _candidate_limit_for_task(config, "response")
    turn_batch_size = _batch_size_for_task(train_config, "turn")
    response_batch_size = _batch_size_for_task(train_config, "response")

    turn_train_loader = DataLoader(
        turn_train_dataset,
        batch_size=turn_batch_size,
        shuffle=_dataloader_shuffle(turn_train_dataset, train=True),
        num_workers=train_num_workers,
        collate_fn=_build_collate_fn("turn", turn_candidate_limit),
        pin_memory=pin_memory,
        **_worker_loader_kwargs(train_config, train_num_workers),
    )
    response_train_loader = DataLoader(
        response_train_dataset,
        batch_size=response_batch_size,
        shuffle=_dataloader_shuffle(response_train_dataset, train=True),
        num_workers=train_num_workers,
        collate_fn=_build_collate_fn("response", response_candidate_limit),
        pin_memory=pin_memory,
        **_worker_loader_kwargs(train_config, train_num_workers),
    )
    turn_valid_loader = None
    if turn_valid_dataset is not None and len(turn_valid_dataset) > 0:
        turn_valid_loader = DataLoader(
            turn_valid_dataset,
            batch_size=turn_batch_size,
            shuffle=False,
            num_workers=valid_num_workers,
            collate_fn=_build_collate_fn("turn", turn_candidate_limit),
            pin_memory=pin_memory,
            **_worker_loader_kwargs(train_config, valid_num_workers),
        )
    response_valid_loader = None
    if response_valid_dataset is not None and len(response_valid_dataset) > 0:
        response_valid_loader = DataLoader(
            response_valid_dataset,
            batch_size=response_batch_size,
            shuffle=False,
            num_workers=valid_num_workers,
            collate_fn=_build_collate_fn("response", response_candidate_limit),
            pin_memory=pin_memory,
            **_worker_loader_kwargs(train_config, valid_num_workers),
        )

    model = MultitaskMahjongPolicyNet(
        hidden_size=int(config.get("model", {}).get("hidden_size", 512)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    model = _wrap_distributed_model(model, distributed_context, train_config, task_name="multitask")
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config.get("lr", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
    )
    turn_weight_map = class_weight_map("turn", train_config)
    response_weight_map = class_weight_map("response", train_config)
    task_config = config.get("task", {})
    losses_config = task_config.get("losses", {})
    turn_loss_scale = float(losses_config.get("turn_ce", 1.0))
    response_loss_scale = float(losses_config.get("response_ce", 1.0))
    multitask_schedule = _resolve_multitask_schedule(task_config)
    interleave_batch_totals = _resolve_interleave_batch_totals(
        task_config,
        turn_loader=turn_train_loader,
        response_loader=response_train_loader,
    )
    log_every_n_steps = int(train_config.get("log_every_n_steps", 0))
    epochs = int(train_config.get("epochs", 1))

    if distributed_context.is_main:
        _print_multitask_training_start(
            device=device,
            output_dir=output_dir,
            log_dir=log_dir,
            turn_train_dataset=turn_train_dataset,
            response_train_dataset=response_train_dataset,
            turn_valid_dataset=turn_valid_dataset,
            response_valid_dataset=response_valid_dataset,
            turn_train_loader=turn_train_loader,
            response_train_loader=response_train_loader,
            turn_valid_loader=turn_valid_loader,
            response_valid_loader=response_valid_loader,
            train_config=train_config,
            model_config=config.get("model", {}),
            turn_candidate_limit=turn_candidate_limit,
            response_candidate_limit=response_candidate_limit,
            turn_batch_size=turn_batch_size,
            response_batch_size=response_batch_size,
            pin_memory=pin_memory,
            train_num_workers=train_num_workers,
            valid_num_workers=valid_num_workers,
            log_every_n_steps=log_every_n_steps,
            epochs=epochs,
            precision_config=precision_config,
            turn_loss_scale=turn_loss_scale,
            response_loss_scale=response_loss_scale,
            schedule=multitask_schedule,
            interleave_batch_totals=interleave_batch_totals,
            distributed_context=distributed_context,
        )

    epoch_summaries: list[dict[str, Any]] = []
    multitask_epoch_summaries: list[dict[str, Any]] = []
    best_turn_accuracy = float("-inf")
    best_response_selection_score = float("-inf")
    best_multitask_score = float("-inf")
    best_valid_accuracy = float("-inf")
    best_valid_loss = float("inf")
    best_valid_macro_accuracy = float("-inf")
    best_checkpoint_path = output_dir / "best.pt"
    best_multitask_checkpoint_path = output_dir / "best_multitask.pt"
    best_turn_checkpoint_path = output_dir / "best_turn.pt"
    best_response_checkpoint_path = output_dir / "best_response.pt"
    best_accuracy_checkpoint_path = output_dir / "best_accuracy.pt"
    best_loss_checkpoint_path = output_dir / "best_loss.pt"
    best_macro_checkpoint_path = output_dir / "best_macro.pt"
    final_checkpoint_path = output_dir / "final.pt"

    last_turn_train_summary: EpochSummary | None = None
    last_response_train_summary: EpochSummary | None = None
    last_train_summary: EpochSummary | None = None
    last_valid_summary: EpochSummary | None = None

    for epoch in range(1, epochs + 1):
        if multitask_schedule == "batch_interleave":
            last_turn_train_summary, last_response_train_summary = _run_interleaved_multitask_epoch(
                model=model,
                turn_loader=turn_train_loader,
                response_loader=response_train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                turn_weight_map=turn_weight_map,
                response_weight_map=response_weight_map,
                log_every_n_steps=log_every_n_steps,
                precision_config=precision_config,
                collect_response_diagnostics=_collect_response_diagnostics(
                    "response",
                    train_config,
                    split="train",
                ),
                turn_loss_scale=turn_loss_scale,
                response_loss_scale=response_loss_scale,
                interleave_batch_totals=interleave_batch_totals,
                distributed_context=distributed_context,
            )
        else:
            last_turn_train_summary = _run_epoch(
                model=model,
                data_loader=turn_train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                split="train_turn",
                task_name="turn",
                train=True,
                weight_map=turn_weight_map,
                log_every_n_steps=log_every_n_steps,
                precision_config=precision_config,
                collect_response_diagnostics=False,
                loss_scale=turn_loss_scale,
                distributed_context=distributed_context,
            )

            last_response_train_summary = _run_epoch(
                model=model,
                data_loader=response_train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                split="train_response",
                task_name="response",
                train=True,
                weight_map=response_weight_map,
                log_every_n_steps=log_every_n_steps,
                precision_config=precision_config,
                collect_response_diagnostics=_collect_response_diagnostics(
                    "response",
                    train_config,
                    split="train",
                ),
                loss_scale=response_loss_scale,
                distributed_context=distributed_context,
            )
        last_train_summary = _combine_epoch_summaries(
            epoch=epoch,
            split="train_multitask",
            summaries=[last_turn_train_summary, last_response_train_summary],
            response_summary=last_response_train_summary,
        )

        if distributed_context.is_main:
            epoch_summaries.append({"task": "turn", **asdict(last_turn_train_summary)})
            epoch_summaries.append({"task": "response", **asdict(last_response_train_summary)})
            multitask_epoch_summaries.append(asdict(last_train_summary))

        turn_valid_summary = None
        response_valid_summary = None
        validation_model = _unwrap_model(model)
        if turn_valid_loader is not None:
            turn_valid_summary = _run_epoch(
                model=validation_model,
                data_loader=turn_valid_loader,
                optimizer=None,
                device=device,
                epoch=epoch,
                split="valid_turn",
                task_name="turn",
                train=False,
                weight_map=turn_weight_map,
                log_every_n_steps=0,
                precision_config=precision_config,
                collect_response_diagnostics=False,
                distributed_context=distributed_context,
            )
            if distributed_context.is_main:
                epoch_summaries.append({"task": "turn", **asdict(turn_valid_summary)})
                if turn_valid_summary.accuracy >= best_turn_accuracy:
                    best_turn_accuracy = turn_valid_summary.accuracy
                    _save_checkpoint(best_turn_checkpoint_path, model, optimizer, config, turn_valid_summary)

        if response_valid_loader is not None:
            response_valid_summary = _run_epoch(
                model=validation_model,
                data_loader=response_valid_loader,
                optimizer=None,
                device=device,
                epoch=epoch,
                split="valid_response",
                task_name="response",
                train=False,
                weight_map=response_weight_map,
                log_every_n_steps=0,
                precision_config=precision_config,
                collect_response_diagnostics=_collect_response_diagnostics(
                    "response",
                    train_config,
                    split="valid",
                ),
                distributed_context=distributed_context,
            )
            if distributed_context.is_main:
                epoch_summaries.append({"task": "response", **asdict(response_valid_summary)})
                current_response_score = response_selection_score(response_valid_summary.response)
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
                        response_valid_summary,
                    )

        if distributed_context.is_main:
            if turn_valid_summary is not None or response_valid_summary is not None:
                valid_summaries = [
                    summary
                    for summary in (turn_valid_summary, response_valid_summary)
                    if summary is not None
                ]
                last_valid_summary = _combine_epoch_summaries(
                    epoch=epoch,
                    split="valid_multitask",
                    summaries=valid_summaries,
                    response_summary=response_valid_summary,
                )
                multitask_epoch_summaries.append(asdict(last_valid_summary))
                multitask_score = _multitask_selection_score(
                    turn_valid_summary=turn_valid_summary,
                    response_valid_summary=response_valid_summary,
                )
                if multitask_score >= best_multitask_score:
                    best_multitask_score = multitask_score
                    _save_checkpoint(
                        best_multitask_checkpoint_path,
                        model,
                        optimizer,
                        config,
                        last_valid_summary,
                    )
                    _save_checkpoint(best_checkpoint_path, model, optimizer, config, last_valid_summary)
                if last_valid_summary.accuracy >= best_valid_accuracy:
                    best_valid_accuracy = last_valid_summary.accuracy
                    _save_checkpoint(
                        best_accuracy_checkpoint_path,
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
            else:
                _save_checkpoint(best_checkpoint_path, model, optimizer, config, last_train_summary)
                _save_checkpoint(
                    best_multitask_checkpoint_path,
                    model,
                    optimizer,
                    config,
                    last_train_summary,
                )
                _save_checkpoint(best_turn_checkpoint_path, model, optimizer, config, last_turn_train_summary)
                _save_checkpoint(
                    best_response_checkpoint_path,
                    model,
                    optimizer,
                    config,
                    last_response_train_summary,
                )
                _save_checkpoint(best_accuracy_checkpoint_path, model, optimizer, config, last_train_summary)
                _save_checkpoint(best_loss_checkpoint_path, model, optimizer, config, last_train_summary)
                _save_checkpoint(best_macro_checkpoint_path, model, optimizer, config, last_train_summary)
        _distributed_barrier(distributed_context)

    assert last_turn_train_summary is not None
    assert last_response_train_summary is not None
    assert last_train_summary is not None
    if distributed_context.is_main:
        _save_checkpoint(final_checkpoint_path, model, optimizer, config, last_train_summary)

    turn_train_samples = int(last_turn_train_summary.samples)
    response_train_samples = int(last_response_train_summary.samples)
    observed_train_samples = int(last_train_summary.samples)
    observed_valid_samples = 0 if last_valid_summary is None else int(last_valid_summary.samples)
    turn_original_train_samples = _dataset_original_rows(turn_train_dataset)
    response_original_train_samples = _dataset_original_rows(response_train_dataset)
    turn_filtered_train_samples = _observed_filtered_rows(
        turn_train_dataset,
        turn_train_samples,
    )
    response_filtered_train_samples = _observed_filtered_rows(
        response_train_dataset,
        response_train_samples,
    )
    original_train_samples = turn_original_train_samples + response_original_train_samples
    filtered_train_samples = turn_filtered_train_samples + response_filtered_train_samples

    metrics_payload = {
        "schedule": multitask_schedule,
        "interleave_batch_totals": {
            "turn": interleave_batch_totals[0],
            "response": interleave_batch_totals[1],
        },
        "epoch_summaries": epoch_summaries,
        "multitask_epoch_summaries": multitask_epoch_summaries,
        "train_samples": observed_train_samples,
        "valid_samples": observed_valid_samples,
        "turn_train_samples": turn_train_samples,
        "response_train_samples": response_train_samples,
        "original_train_samples": original_train_samples,
        "filtered_train_samples": filtered_train_samples,
        "turn_original_train_samples": turn_original_train_samples,
        "response_original_train_samples": response_original_train_samples,
        "turn_filtered_train_samples": turn_filtered_train_samples,
        "response_filtered_train_samples": response_filtered_train_samples,
        "best_turn_accuracy": None if best_turn_accuracy == float("-inf") else best_turn_accuracy,
        "best_response_selection_score": (
            None
            if best_response_selection_score == float("-inf")
            else best_response_selection_score
        ),
        "best_multitask_selection_score": (
            None if best_multitask_score == float("-inf") else best_multitask_score
        ),
        "best_turn_checkpoint": str(best_turn_checkpoint_path),
        "best_response_checkpoint": str(best_response_checkpoint_path),
        "best_multitask_checkpoint": str(best_multitask_checkpoint_path),
    }
    if distributed_context.is_main:
        (output_dir / "train_metrics.json").write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (log_dir / "epoch_metrics.jsonl").write_text(
            "".join(
                json.dumps(summary, ensure_ascii=False) + "\n"
                for summary in [*epoch_summaries, *multitask_epoch_summaries]
            ),
            encoding="utf-8",
        )
    _shutdown_distributed_context(distributed_context)

    return TrainSummary(
        task="multitask",
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
        best_response_checkpoint=str(best_response_checkpoint_path),
        best_loss_checkpoint=str(best_loss_checkpoint_path),
        best_macro_checkpoint=str(best_macro_checkpoint_path),
        final_checkpoint=str(final_checkpoint_path),
        used_device=str(device),
    )


def _prepare_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _run_policy_batch(
    *,
    model: nn.Module,
    prepared_batch: dict[str, Any],
    optimizer: AdamW | None,
    device: torch.device,
    task_name: str,
    train: bool,
    weight_map: dict[str, float],
    precision_config: dict[str, Any],
    collect_response_diagnostics: bool,
    loss_scale: float,
) -> BatchStepResult:
    with _autocast_context(device, precision_config):
        logits = model(prepared_batch, task_name=task_name)
        sample_weights = build_sample_weights(prepared_batch["label_kinds"], weight_map, device)
        loss = policy_loss(logits, prepared_batch["labels"], sample_weights=sample_weights)
        per_sample_losses = F.cross_entropy(logits, prepared_batch["labels"], reduction="none")

    if train:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
        (loss * float(loss_scale)).backward()
        optimizer.step()

    predictions = logits.argmax(dim=1)
    label_ranks = (
        _compute_label_ranks(logits, prepared_batch["labels"])
        if task_name == "response" and collect_response_diagnostics
        else None
    )
    correct = predictions == prepared_batch["labels"]
    return BatchStepResult(
        batch_size=int(prepared_batch["labels"].shape[0]),
        batch_correct=int(correct.sum().detach().item()),
        loss_value=float(loss.detach().item()),
        correct=correct,
        predictions=predictions,
        per_sample_losses=per_sample_losses,
        label_ranks=label_ranks,
    )


def _update_epoch_accumulator(
    accumulator: EpochAccumulator,
    prepared_batch: dict[str, Any],
    result: BatchStepResult,
    *,
    data_wait_elapsed: float,
    compute_elapsed: float,
) -> None:
    batch_elapsed = data_wait_elapsed + compute_elapsed
    accumulator.batches += 1
    accumulator.total_loss += result.loss_value * result.batch_size
    accumulator.total_correct += result.batch_correct
    accumulator.total_samples += result.batch_size
    accumulator.running_loss += result.loss_value * result.batch_size
    accumulator.running_correct += result.batch_correct
    accumulator.running_samples += result.batch_size
    accumulator.running_time += batch_elapsed
    accumulator.running_data_wait_time += data_wait_elapsed
    accumulator.running_compute_time += compute_elapsed

    response_metrics = accumulator.response_metrics
    if response_metrics is None:
        _update_fast_label_summaries(
            task_name=accumulator.task_name,
            label_kind_ids=prepared_batch["label_kind_ids"],
            correct=result.correct,
            per_sample_losses=result.per_sample_losses,
            kind_totals=accumulator.kind_totals,
            kind_correct=accumulator.kind_correct,
            kind_loss_totals=accumulator.kind_loss_totals,
            group_totals=accumulator.group_totals,
            group_correct=accumulator.group_correct,
            group_loss_totals=accumulator.group_loss_totals,
        )
        return

    assert result.label_ranks is not None
    correct_list = result.correct.detach().to("cpu").tolist()
    per_sample_loss_list = result.per_sample_losses.detach().to("cpu").tolist()
    prediction_list = result.predictions.detach().to("cpu").tolist()
    label_rank_list = result.label_ranks.detach().to("cpu").tolist()
    for batch_index, label_kind in enumerate(prepared_batch["label_kinds"]):
        is_correct = int(correct_list[batch_index])
        sample_loss = float(per_sample_loss_list[batch_index])
        accumulator.kind_totals[label_kind] = accumulator.kind_totals.get(label_kind, 0) + 1
        accumulator.kind_correct[label_kind] = accumulator.kind_correct.get(label_kind, 0) + is_correct
        accumulator.kind_loss_totals[label_kind] = (
            accumulator.kind_loss_totals.get(label_kind, 0.0) + sample_loss
        )

        group_name = class_group_name(accumulator.task_name, label_kind)
        accumulator.group_totals[group_name] = accumulator.group_totals.get(group_name, 0) + 1
        accumulator.group_correct[group_name] = (
            accumulator.group_correct.get(group_name, 0) + is_correct
        )
        accumulator.group_loss_totals[group_name] = (
            accumulator.group_loss_totals.get(group_name, 0.0) + sample_loss
        )
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


def _reset_running_accumulator(accumulator: EpochAccumulator) -> None:
    accumulator.running_loss = 0.0
    accumulator.running_correct = 0
    accumulator.running_samples = 0
    accumulator.running_time = 0.0
    accumulator.running_data_wait_time = 0.0
    accumulator.running_compute_time = 0.0


def _summary_from_accumulator(
    accumulator: EpochAccumulator,
    *,
    epoch: int,
    split: str,
) -> EpochSummary:
    average_loss = (
        0.0 if accumulator.total_samples == 0 else accumulator.total_loss / accumulator.total_samples
    )
    accuracy = (
        0.0
        if accumulator.total_samples == 0
        else accumulator.total_correct / accumulator.total_samples
    )
    per_kind = _build_label_metric_summary(
        accumulator.kind_totals,
        accumulator.kind_correct,
        accumulator.kind_loss_totals,
    )
    class_groups = _build_label_metric_summary(
        accumulator.group_totals,
        accumulator.group_correct,
        accumulator.group_loss_totals,
    )
    return EpochSummary(
        epoch=epoch,
        split=split,
        loss=average_loss,
        accuracy=accuracy,
        macro_accuracy=_macro_accuracy(per_kind),
        batches=accumulator.batches,
        samples=accumulator.total_samples,
        per_kind=per_kind,
        class_groups=class_groups,
        response=None
        if accumulator.response_metrics is None
        else accumulator.response_metrics.summary(),
    )


def _run_interleaved_multitask_epoch(
    *,
    model: nn.Module,
    turn_loader: DataLoader,
    response_loader: DataLoader,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
    turn_weight_map: dict[str, float],
    response_weight_map: dict[str, float],
    log_every_n_steps: int,
    precision_config: dict[str, Any],
    collect_response_diagnostics: bool,
    turn_loss_scale: float,
    response_loss_scale: float,
    interleave_batch_totals: tuple[float, float],
    distributed_context: DistributedContext | None = None,
) -> tuple[EpochSummary, EpochSummary]:
    model.train(True)
    turn_accumulator = EpochAccumulator(
        task_name="turn",
        collect_response_diagnostics=False,
    )
    response_accumulator = EpochAccumulator(
        task_name="response",
        collect_response_diagnostics=collect_response_diagnostics,
    )
    turn_schedule_batches, response_schedule_batches = interleave_batch_totals
    total_batches = max(1, int(round(turn_schedule_batches + response_schedule_batches)))
    epoch_start = time.perf_counter()
    next_batch_start = epoch_start
    executed_batches = 0

    turn_iter = iter(turn_loader)
    response_iter = iter(response_loader)
    turn_done = 0
    response_done = 0
    turn_exhausted = False
    response_exhausted = False

    with _ddp_join_context(model, train=True):
        while not (turn_exhausted and response_exhausted):
            task_name = _next_interleaved_task(
                turn_done=turn_done,
                response_done=response_done,
                turn_total=turn_schedule_batches,
                response_total=response_schedule_batches,
                turn_exhausted=turn_exhausted,
                response_exhausted=response_exhausted,
            )
            if task_name == "turn":
                try:
                    batch = next(turn_iter)
                except StopIteration:
                    turn_exhausted = True
                    continue
                turn_done += 1
                accumulator = turn_accumulator
                weight_map = turn_weight_map
                loss_scale = turn_loss_scale
            else:
                try:
                    batch = next(response_iter)
                except StopIteration:
                    response_exhausted = True
                    continue
                response_done += 1
                accumulator = response_accumulator
                weight_map = response_weight_map
                loss_scale = response_loss_scale

            batch_ready = time.perf_counter()
            data_wait_elapsed = batch_ready - next_batch_start
            batch_start = batch_ready
            prepared_batch = _prepare_batch(batch, device)
            result = _run_policy_batch(
                model=model,
                prepared_batch=prepared_batch,
                optimizer=optimizer,
                device=device,
                task_name=task_name,
                train=True,
                weight_map=weight_map,
                precision_config=precision_config,
                collect_response_diagnostics=task_name == "response" and collect_response_diagnostics,
                loss_scale=loss_scale,
            )
            compute_elapsed = time.perf_counter() - batch_start
            batch_elapsed = data_wait_elapsed + compute_elapsed
            _update_epoch_accumulator(
                accumulator,
                prepared_batch,
                result,
                data_wait_elapsed=data_wait_elapsed,
                compute_elapsed=compute_elapsed,
            )
            executed_batches += 1

            if (
                _is_main_process(distributed_context)
                and log_every_n_steps > 0
                and (
                    executed_batches == 1
                    or executed_batches % log_every_n_steps == 0
                    or executed_batches == total_batches
                )
            ):
                _print_batch_progress(
                    epoch=epoch,
                    split=f"train_interleave_{task_name}",
                    batch_index=executed_batches,
                    total_batches=total_batches,
                    total_samples=turn_accumulator.total_samples + response_accumulator.total_samples,
                    batch_samples=result.batch_size,
                    batch_elapsed=batch_elapsed,
                    data_wait_elapsed=data_wait_elapsed,
                    compute_elapsed=compute_elapsed,
                    running_time=turn_accumulator.running_time + response_accumulator.running_time,
                    running_data_wait_time=(
                        turn_accumulator.running_data_wait_time
                        + response_accumulator.running_data_wait_time
                    ),
                    running_compute_time=(
                        turn_accumulator.running_compute_time
                        + response_accumulator.running_compute_time
                    ),
                    running_loss=turn_accumulator.running_loss + response_accumulator.running_loss,
                    running_correct=(
                        turn_accumulator.running_correct + response_accumulator.running_correct
                    ),
                    running_samples=(
                        turn_accumulator.running_samples + response_accumulator.running_samples
                    ),
                    epoch_start=epoch_start,
                )
                _reset_running_accumulator(turn_accumulator)
                _reset_running_accumulator(response_accumulator)

            next_batch_start = time.perf_counter()

    turn_summary = _summary_from_accumulator(turn_accumulator, epoch=epoch, split="train_turn")
    response_summary = _summary_from_accumulator(
        response_accumulator,
        epoch=epoch,
        split="train_response",
    )
    return (
        _sync_epoch_summary(turn_summary, distributed_context),
        _sync_epoch_summary(response_summary, distributed_context),
    )


def _next_interleaved_task(
    *,
    turn_done: int,
    response_done: int,
    turn_total: float,
    response_total: float,
    turn_exhausted: bool,
    response_exhausted: bool,
) -> str:
    if turn_exhausted:
        return "response"
    if response_exhausted:
        return "turn"
    turn_progress = turn_done / max(1, turn_total)
    response_progress = response_done / max(1, response_total)
    return "turn" if turn_progress <= response_progress else "response"


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
    loss_scale: float = 1.0,
    distributed_context: DistributedContext | None = None,
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
    accumulator = EpochAccumulator(
        task_name=task_name,
        collect_response_diagnostics=collect_response_diagnostics,
    )

    model.train(train)
    total_batches = len(data_loader)
    epoch_start = time.perf_counter()
    next_batch_start = epoch_start
    with _ddp_join_context(model, train=train):
        for batch in data_loader:
            batch_ready = time.perf_counter()
            data_wait_elapsed = batch_ready - next_batch_start
            batch_start = batch_ready
            prepared_batch = _prepare_batch(batch, device)
            result = _run_policy_batch(
                model=model,
                prepared_batch=prepared_batch,
                optimizer=optimizer,
                device=device,
                task_name=task_name,
                train=train,
                weight_map=weight_map,
                precision_config=precision_config,
                collect_response_diagnostics=collect_response_diagnostics,
                loss_scale=loss_scale,
            )
            compute_elapsed = time.perf_counter() - batch_start
            batch_elapsed = data_wait_elapsed + compute_elapsed
            _update_epoch_accumulator(
                accumulator,
                prepared_batch,
                result,
                data_wait_elapsed=data_wait_elapsed,
                compute_elapsed=compute_elapsed,
            )

            if (
                train
                and _is_main_process(distributed_context)
                and log_every_n_steps > 0
                and (
                    accumulator.batches == 1
                    or accumulator.batches % log_every_n_steps == 0
                    or accumulator.batches == total_batches
                )
            ):
                _print_batch_progress(
                    epoch=epoch,
                    split=split,
                    batch_index=accumulator.batches,
                    total_batches=total_batches,
                    total_samples=accumulator.total_samples,
                    batch_samples=result.batch_size,
                    batch_elapsed=batch_elapsed,
                    data_wait_elapsed=data_wait_elapsed,
                    compute_elapsed=compute_elapsed,
                    running_time=accumulator.running_time,
                    running_data_wait_time=accumulator.running_data_wait_time,
                    running_compute_time=accumulator.running_compute_time,
                    running_loss=accumulator.running_loss,
                    running_correct=accumulator.running_correct,
                    running_samples=accumulator.running_samples,
                    epoch_start=epoch_start,
                )
                _reset_running_accumulator(accumulator)
            next_batch_start = time.perf_counter()

    summary = _summary_from_accumulator(
        accumulator,
        epoch=epoch,
        split=split,
    )
    return _sync_epoch_summary(summary, distributed_context)


def _initialize_distributed_context(train_config: dict[str, Any]) -> DistributedContext:
    requested = bool(train_config.get("distributed", False))
    env_world_size = _env_int("WORLD_SIZE", 1)
    env_rank = _env_int("RANK", 0)
    env_local_rank = _env_int("LOCAL_RANK", 0)
    if not requested:
        if env_world_size > 1:
            raise RuntimeError(
                "torchrun launched multiple processes, but train.distributed is false"
            )
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            backend=None,
            device=_resolve_training_device(train_config.get("device")),
        )
    if env_world_size <= 1:
        raise RuntimeError(
            "train.distributed=true requires torchrun with WORLD_SIZE > 1, "
            "for example: torchrun --standalone --nproc_per_node=2 -m mjai.cli train --config <config>"
        )
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    device = _resolve_training_device(train_config.get("device"), local_rank=env_local_rank)
    backend = _resolve_distributed_backend(train_config, device)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistributedContext(
        enabled=True,
        rank=dist.get_rank() if dist.is_initialized() else env_rank,
        local_rank=env_local_rank,
        world_size=dist.get_world_size() if dist.is_initialized() else env_world_size,
        backend=backend,
        device=device,
    )


def _resolve_distributed_backend(train_config: dict[str, Any], device: torch.device) -> str:
    backend = str(train_config.get("backend", "auto")).lower()
    if backend in {"auto", "vendor"}:
        if device.type == "cuda":
            return "nccl"
        if device.type == "xpu":
            return "ccl"
        return "gloo"
    return backend


def _wrap_distributed_model(
    model: nn.Module,
    distributed_context: DistributedContext,
    train_config: dict[str, Any],
    *,
    task_name: str,
) -> nn.Module:
    if not distributed_context.enabled:
        return model
    find_unused = train_config.get("find_unused_parameters")
    if find_unused is None:
        find_unused = task_name == "multitask"
    if task_name == "multitask" and not bool(find_unused):
        raise ValueError("multitask DDP requires train.find_unused_parameters=true")
    ddp_kwargs: dict[str, Any] = {
        "find_unused_parameters": bool(find_unused),
    }
    if distributed_context.device.type == "cuda":
        ddp_kwargs["device_ids"] = [distributed_context.local_rank]
        ddp_kwargs["output_device"] = distributed_context.local_rank
    return DistributedDataParallel(model, **ddp_kwargs)


def _unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _ddp_join_context(model: nn.Module, *, train: bool):
    if train and isinstance(model, DistributedDataParallel):
        return model.join()
    return nullcontext()


def _is_main_process(distributed_context: DistributedContext | None) -> bool:
    return distributed_context is None or distributed_context.is_main


def _distributed_enabled(distributed_context: DistributedContext | None) -> bool:
    return bool(distributed_context is not None and distributed_context.enabled)


def _distributed_rank(distributed_context: DistributedContext | None) -> int:
    return 0 if distributed_context is None else distributed_context.rank


def _distributed_world_size(distributed_context: DistributedContext | None) -> int:
    return 1 if distributed_context is None else distributed_context.world_size


def _distributed_backend(distributed_context: DistributedContext | None) -> str | None:
    return None if distributed_context is None else distributed_context.backend


def _distributed_barrier(distributed_context: DistributedContext | None) -> None:
    if (
        distributed_context is not None
        and distributed_context.enabled
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.barrier()


def _shutdown_distributed_context(distributed_context: DistributedContext | None) -> None:
    if (
        distributed_context is not None
        and distributed_context.enabled
        and dist.is_available()
        and dist.is_initialized()
    ):
        dist.destroy_process_group()


def _sync_epoch_summary(
    summary: EpochSummary,
    distributed_context: DistributedContext | None,
) -> EpochSummary:
    if (
        distributed_context is None
        or not distributed_context.enabled
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return summary
    gathered: list[EpochSummary | None] = [None for _ in range(distributed_context.world_size)]
    dist.all_gather_object(gathered, summary)
    return _combine_distributed_epoch_summaries(
        [item for item in gathered if item is not None],
        response_summary=summary if distributed_context.is_main else None,
    )


def _combine_distributed_epoch_summaries(
    summaries: list[EpochSummary],
    *,
    response_summary: EpochSummary | None,
) -> EpochSummary:
    if not summaries:
        raise ValueError("cannot combine empty distributed epoch summaries")
    first = summaries[0]
    total_samples = sum(summary.samples for summary in summaries)
    total_batches = sum(summary.batches for summary in summaries)
    if total_samples <= 0:
        return EpochSummary(
            epoch=first.epoch,
            split=first.split,
            loss=0.0,
            accuracy=0.0,
            macro_accuracy=0.0,
            batches=total_batches,
            samples=0,
            per_kind={},
            class_groups={},
            response=None,
        )
    per_kind = _combine_label_metric_maps([summary.per_kind for summary in summaries])
    class_groups = _combine_label_metric_maps([summary.class_groups for summary in summaries])
    response = combine_response_diagnostics(summary.response for summary in summaries)
    if response is None and response_summary is not None:
        response = response_summary.response
    return EpochSummary(
        epoch=first.epoch,
        split=first.split,
        loss=sum(summary.loss * summary.samples for summary in summaries) / total_samples,
        accuracy=sum(summary.accuracy * summary.samples for summary in summaries) / total_samples,
        macro_accuracy=_macro_accuracy(per_kind),
        batches=total_batches,
        samples=total_samples,
        per_kind=per_kind,
        class_groups=class_groups,
        response=response,
    )


def _combine_label_metric_maps(
    metric_maps: list[dict[str, LabelMetricSummary]],
) -> dict[str, LabelMetricSummary]:
    combined: dict[str, LabelMetricSummary] = {}
    keys = sorted({key for metric_map in metric_maps for key in metric_map})
    for key in keys:
        metrics = [metric_map[key] for metric_map in metric_maps if key in metric_map]
        samples = sum(metric.samples for metric in metrics)
        if samples <= 0:
            combined[key] = LabelMetricSummary(samples=0, loss=0.0, accuracy=0.0)
            continue
        combined[key] = LabelMetricSummary(
            samples=samples,
            loss=sum(metric.loss * metric.samples for metric in metrics) / samples,
            accuracy=sum(metric.accuracy * metric.samples for metric in metrics) / samples,
        )
    return combined


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _resolve_training_device(
    config_device: str | None,
    local_rank: int | None = None,
) -> torch.device:
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
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cuda")
    if normalized in {"auto", "xpu"} and _xpu_is_available():
        if local_rank is not None:
            xpu = getattr(torch, "xpu")
            set_device = getattr(xpu, "set_device", None)
            if callable(set_device):
                set_device(local_rank)
            return torch.device("xpu", local_rank)
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
            "model_state_dict": _unwrap_model(model).state_dict(),
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
    distributed_context: DistributedContext | None = None,
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
    distributed_kwargs = (
        {
            "distributed_rank": distributed_context.rank,
            "distributed_world_size": distributed_context.world_size,
        }
        if split in {"train", "valid"} and distributed_context is not None and distributed_context.enabled
        else {}
    )
    if task_name == "turn":
        if bool(train_config.get("streaming_dataset", True)):
            shuffle = split == "train"
            return StreamingManifestTurnDataset(
                manifest_path,
                shuffle_shards=shuffle and bool(train_config.get("shuffle_shards", True)),
                shuffle_rows=shuffle and bool(train_config.get("shuffle_rows", True)),
                shuffle_buffer_size=int(train_config.get("shuffle_buffer_size", 8192)),
                seed=int(train_config.get("dataset_seed", 20260423)),
                **distributed_kwargs,
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
            **distributed_kwargs,
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
    distributed_context: DistributedContext | None = None,
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
        f"distributed={_distributed_enabled(distributed_context)} "
        f"rank={_distributed_rank(distributed_context)} "
        f"world_size={_distributed_world_size(distributed_context)} "
        f"backend={_distributed_backend(distributed_context)} "
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


def _print_multitask_training_start(
    *,
    device: torch.device,
    output_dir: Path,
    log_dir: Path,
    turn_train_dataset,
    response_train_dataset,
    turn_valid_dataset,
    response_valid_dataset,
    turn_train_loader: DataLoader,
    response_train_loader: DataLoader,
    turn_valid_loader: DataLoader | None,
    response_valid_loader: DataLoader | None,
    train_config: dict[str, Any],
    model_config: dict[str, Any],
    turn_candidate_limit: int,
    response_candidate_limit: int,
    turn_batch_size: int,
    response_batch_size: int,
    pin_memory: bool,
    train_num_workers: int,
    valid_num_workers: int,
    log_every_n_steps: int,
    epochs: int,
    precision_config: dict[str, Any],
    turn_loss_scale: float,
    response_loss_scale: float,
    schedule: str,
    interleave_batch_totals: tuple[float, float],
    distributed_context: DistributedContext | None = None,
) -> None:
    turn_valid_samples = 0 if turn_valid_dataset is None else len(turn_valid_dataset)
    response_valid_samples = 0 if response_valid_dataset is None else len(response_valid_dataset)
    turn_valid_batches = 0 if turn_valid_loader is None else len(turn_valid_loader)
    response_valid_batches = 0 if response_valid_loader is None else len(response_valid_loader)
    print(
        "[train] starting "
        "task=multitask "
        f"device={device} "
        f"distributed={_distributed_enabled(distributed_context)} "
        f"rank={_distributed_rank(distributed_context)} "
        f"world_size={_distributed_world_size(distributed_context)} "
        f"backend={_distributed_backend(distributed_context)} "
        f"cuda_available={torch.cuda.is_available()} "
        f"xpu_available={_xpu_is_available()} "
        f"epochs={epochs} "
        f"turn_train_samples={len(turn_train_dataset)} "
        f"response_train_samples={len(response_train_dataset)} "
        f"turn_valid_samples={turn_valid_samples} "
        f"response_valid_samples={response_valid_samples} "
        f"turn_train_batches={len(turn_train_loader)} "
        f"response_train_batches={len(response_train_loader)} "
        f"turn_valid_batches={turn_valid_batches} "
        f"response_valid_batches={response_valid_batches} "
        f"turn_batch_size={turn_batch_size} "
        f"response_batch_size={response_batch_size} "
        f"turn_candidate_limit={turn_candidate_limit} "
        f"response_candidate_limit={response_candidate_limit} "
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
        f"hidden_size={int(model_config.get('hidden_size', 512))} "
        f"dropout={float(model_config.get('dropout', 0.1))} "
        f"lr={float(train_config.get('lr', 1e-3))} "
        f"weight_decay={float(train_config.get('weight_decay', 0.0))} "
        f"schedule={schedule} "
        f"interleave_turn_batches={interleave_batch_totals[0]:.1f} "
        f"interleave_response_batches={interleave_batch_totals[1]:.1f} "
        f"turn_loss_scale={turn_loss_scale} "
        f"response_loss_scale={response_loss_scale} "
        f"log_every_n_steps={log_every_n_steps} "
        f"response_train_diagnostics={_collect_response_diagnostics('response', train_config, 'train')} "
        f"response_eval_diagnostics={_collect_response_diagnostics('response', train_config, 'valid')} "
        f"output_dir={output_dir} "
        f"log_dir={log_dir}",
        flush=True,
    )


def _candidate_limit_for_task(config: dict[str, Any], task_name: str) -> int:
    data_config = config.get("data", {})
    task_key = f"{task_name}_candidate_limit"
    if task_key in data_config:
        return int(data_config[task_key])
    if "candidate_limit" in data_config:
        return int(data_config["candidate_limit"])
    return 16 if task_name == "response" else 64


def _resolve_multitask_schedule(task_config: dict[str, Any]) -> str:
    schedule = str(task_config.get("schedule", "epoch_sequential")).lower()
    aliases = {
        "epoch": "epoch_sequential",
        "sequential": "epoch_sequential",
        "epoch_sequential": "epoch_sequential",
        "interleave": "batch_interleave",
        "batch_interleave": "batch_interleave",
        "batch_interleaved": "batch_interleave",
    }
    if schedule not in aliases:
        raise ValueError(
            "unsupported multitask schedule: "
            f"{schedule}; expected epoch_sequential or batch_interleave"
        )
    return aliases[schedule]


def _resolve_interleave_batch_totals(
    task_config: dict[str, Any],
    *,
    turn_loader: DataLoader,
    response_loader: DataLoader,
) -> tuple[float, float]:
    interleave_config = task_config.get("interleave_batches", {})
    if interleave_config is None:
        interleave_config = {}
    if not isinstance(interleave_config, dict):
        raise TypeError("task.interleave_batches must be a mapping when provided")

    turn_total = float(interleave_config.get("turn", len(turn_loader)))
    response_total = float(interleave_config.get("response", len(response_loader)))
    if turn_total <= 0 or response_total <= 0:
        raise ValueError(
            "task.interleave_batches turn/response must be positive "
            f"when provided, got turn={turn_total} response={response_total}"
        )
    return turn_total, response_total


def _batch_size_for_task(train_config: dict[str, Any], task_name: str) -> int:
    return int(train_config.get(f"{task_name}_batch_size", train_config.get("batch_size", 32)))


def _combine_epoch_summaries(
    *,
    epoch: int,
    split: str,
    summaries: list[EpochSummary],
    response_summary: EpochSummary | None,
) -> EpochSummary:
    total_samples = sum(summary.samples for summary in summaries)
    total_batches = sum(summary.batches for summary in summaries)
    if total_samples <= 0:
        return EpochSummary(
            epoch=epoch,
            split=split,
            loss=0.0,
            accuracy=0.0,
            macro_accuracy=0.0,
            batches=total_batches,
            samples=0,
            per_kind={},
            class_groups={},
            response=None,
        )

    average_loss = sum(summary.loss * summary.samples for summary in summaries) / total_samples
    accuracy = sum(summary.accuracy * summary.samples for summary in summaries) / total_samples
    per_kind: dict[str, LabelMetricSummary] = {}
    class_groups: dict[str, LabelMetricSummary] = {}
    for summary in summaries:
        task_prefix = _summary_task_prefix(summary)
        per_kind.update(
            {
                f"{task_prefix}/{kind}": metric
                for kind, metric in summary.per_kind.items()
            }
        )
        class_groups.update(
            {
                f"{task_prefix}/{group}": metric
                for group, metric in summary.class_groups.items()
            }
        )
    response = combine_response_diagnostics(summary.response for summary in summaries)
    if response is None and response_summary is not None:
        response = response_summary.response
    return EpochSummary(
        epoch=epoch,
        split=split,
        loss=average_loss,
        accuracy=accuracy,
        macro_accuracy=_macro_accuracy(per_kind),
        batches=total_batches,
        samples=total_samples,
        per_kind=per_kind,
        class_groups=class_groups,
        response=response,
    )


def _summary_task_prefix(summary: EpochSummary) -> str:
    if "response" in summary.split:
        return "response"
    if "turn" in summary.split:
        return "turn"
    return "task"


def _multitask_selection_score(
    *,
    turn_valid_summary: EpochSummary | None,
    response_valid_summary: EpochSummary | None,
) -> float:
    scores: list[float] = []
    if turn_valid_summary is not None:
        scores.append(turn_valid_summary.accuracy)
    if response_valid_summary is not None:
        response_score = response_selection_score(response_valid_summary.response)
        scores.append(response_valid_summary.accuracy if response_score is None else response_score)
    if not scores:
        return float("-inf")
    return sum(scores) / len(scores)


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
