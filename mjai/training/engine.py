from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mjai.data.datasets.parquet_dataset import (
    ManifestResponseDataset,
    ManifestTurnDataset,
    ResponseBatchCollator,
    TurnBatchCollator,
)
from mjai.features.action_features import RESPONSE_ACTION_DIM, TURN_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM
from mjai.models.policy_net import MahjongPolicyNet
from mjai.training.class_balance import build_sample_weights, class_group_name, class_weight_map
from mjai.training.losses import policy_loss
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
    best_checkpoint: str
    final_checkpoint: str
    used_device: str


def train_turn_model(config: dict[str, Any]) -> TrainSummary:
    task_name = str(config.get("task", {}).get("task", ""))
    if task_name not in {"turn", "response"}:
        raise ValueError(
            f"minimal training path currently supports only task=turn/response, got: {task_name}"
        )
    seed_everything(int(config.get("seed", 20260423)))

    output_dir = ensure_dir(config.get("output_dir", "outputs/checkpoints/turn_smoke"))
    log_dir = ensure_dir(config.get("log_dir", "outputs/logs/turn_smoke"))
    dump_yaml(output_dir / "config.resolved.yaml", config)

    train_dataset = _build_dataset(task_name, config["data"]["train_manifest"])
    valid_manifest = config.get("data", {}).get("valid_manifest")
    valid_dataset = (
        _build_dataset(task_name, valid_manifest)
        if valid_manifest and Path(valid_manifest).exists()
        else None
    )
    candidate_limit = int(config.get("data", {}).get("candidate_limit", 64))
    collate_fn = _build_collate_fn(task_name, candidate_limit)
    train_config = config.get("train", {})
    device = _resolve_training_device(train_config.get("device"))
    train_num_workers = int(train_config.get("num_workers", 0))
    valid_num_workers = int(train_config.get("valid_num_workers", 0))
    pin_memory = bool(train_config.get("pin_memory", device.type == "cuda"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_config.get("batch_size", 32)),
        shuffle=True,
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
    epoch_summaries: list[dict[str, Any]] = []
    best_valid_accuracy = float("-inf")
    best_checkpoint_path = output_dir / "best.pt"
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
            )
            epoch_summaries.append(asdict(last_valid_summary))

            if last_valid_summary.accuracy >= best_valid_accuracy:
                best_valid_accuracy = last_valid_summary.accuracy
                _save_checkpoint(best_checkpoint_path, model, optimizer, config, last_valid_summary)
        else:
            _save_checkpoint(best_checkpoint_path, model, optimizer, config, last_train_summary)

    assert last_train_summary is not None
    _save_checkpoint(final_checkpoint_path, model, optimizer, config, last_train_summary)

    metrics_path = output_dir / "train_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "epoch_summaries": epoch_summaries,
                "train_samples": len(train_dataset),
                "valid_samples": 0 if valid_dataset is None else len(valid_dataset),
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
        train_samples=len(train_dataset),
        valid_samples=0 if valid_dataset is None else len(valid_dataset),
        best_checkpoint=str(best_checkpoint_path),
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
) -> EpochSummary:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batches = 0
    running_loss = 0.0
    running_correct = 0
    running_samples = 0
    running_time = 0.0
    kind_totals: dict[str, int] = {}
    kind_correct: dict[str, int] = {}
    kind_loss_totals: dict[str, float] = {}
    group_totals: dict[str, int] = {}
    group_correct: dict[str, int] = {}
    group_loss_totals: dict[str, float] = {}

    model.train(train)
    total_batches = len(data_loader)
    epoch_start = time.perf_counter()
    for batch in data_loader:
        batch_start = time.perf_counter()
        batches += 1
        prepared_batch = {
            key: value.to(device, non_blocking=device.type == "cuda")
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

        if task_name == "turn":
            logits = model.forward_turn(prepared_batch)
        else:
            logits = model.forward_response(prepared_batch)
        sample_weights = build_sample_weights(prepared_batch["label_kinds"], weight_map, device)
        loss = policy_loss(logits, prepared_batch["labels"], sample_weights=sample_weights)
        per_sample_losses = F.cross_entropy(logits, prepared_batch["labels"], reduction="none")

        if train:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = prepared_batch["labels"].shape[0]
        predictions = logits.argmax(dim=1)
        correct = predictions == prepared_batch["labels"]
        total_loss += float(loss.item()) * batch_size
        total_correct += int(correct.sum().item())
        total_samples += batch_size
        batch_elapsed = time.perf_counter() - batch_start
        running_loss += float(loss.item()) * batch_size
        running_correct += int(correct.sum().item())
        running_samples += batch_size
        running_time += batch_elapsed

        for batch_index, label_kind in enumerate(prepared_batch["label_kinds"]):
            is_correct = int(correct[batch_index].item())
            sample_loss = float(per_sample_losses[batch_index].item())
            kind_totals[label_kind] = kind_totals.get(label_kind, 0) + 1
            kind_correct[label_kind] = kind_correct.get(label_kind, 0) + is_correct
            kind_loss_totals[label_kind] = kind_loss_totals.get(label_kind, 0.0) + sample_loss

            group_name = class_group_name(task_name, label_kind)
            group_totals[group_name] = group_totals.get(group_name, 0) + 1
            group_correct[group_name] = group_correct.get(group_name, 0) + is_correct
            group_loss_totals[group_name] = group_loss_totals.get(group_name, 0.0) + sample_loss

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
                running_time=running_time,
                running_loss=running_loss,
                running_correct=running_correct,
                running_samples=running_samples,
                epoch_start=epoch_start,
            )
            running_loss = 0.0
            running_correct = 0
            running_samples = 0
            running_time = 0.0

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
    )


def _resolve_training_device(config_device: str | None) -> torch.device:
    normalized = normalize_device_name(config_device)
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    config: dict[str, Any],
    summary: EpochSummary,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "summary": asdict(summary),
        },
        path,
    )


def _build_dataset(task_name: str, manifest_path: str | Path):
    if task_name == "turn":
        return ManifestTurnDataset(manifest_path)
    return ManifestResponseDataset(manifest_path)


def _build_collate_fn(task_name: str, candidate_limit: int):
    if task_name == "turn":
        return TurnBatchCollator(candidate_limit=candidate_limit)
    return ResponseBatchCollator(candidate_limit=candidate_limit)


def _worker_loader_kwargs(train_config: dict[str, Any], num_workers: int) -> dict[str, Any]:
    if num_workers <= 0:
        return {}
    return {
        "persistent_workers": bool(train_config.get("persistent_workers", True)),
        "prefetch_factor": int(train_config.get("prefetch_factor", 2)),
    }


def _print_batch_progress(
    epoch: int,
    split: str,
    batch_index: int,
    total_batches: int,
    total_samples: int,
    batch_samples: int,
    batch_elapsed: float,
    running_time: float,
    running_loss: float,
    running_correct: int,
    running_samples: int,
    epoch_start: float,
) -> None:
    batches_in_window = max(1.0, running_samples / max(1, batch_samples))
    average_batch_time = running_time / batches_in_window
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
        f"avg_batch_time={average_batch_time:.3f}s "
        f"samples_per_s={samples_per_second:.1f} "
        f"loss={window_loss:.4f} "
        f"accuracy={window_accuracy:.4f} "
        f"eta={_format_duration(eta_seconds)}",
        flush=True,
    )


def _format_duration(seconds: float) -> str:
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
    return {
        label: LabelMetricSummary(
            samples=count,
            loss=0.0 if count == 0 else loss_totals.get(label, 0.0) / count,
            accuracy=0.0 if count == 0 else correct.get(label, 0) / count,
        )
        for label, count in sorted(totals.items())
    }


def _macro_accuracy(per_kind: dict[str, LabelMetricSummary]) -> float:
    if not per_kind:
        return 0.0
    return sum(summary.accuracy for summary in per_kind.values()) / len(per_kind)
