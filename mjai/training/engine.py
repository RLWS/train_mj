from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from mjai.data.datasets.parquet_dataset import ManifestTurnDataset, build_turn_batch
from mjai.features.action_features import TURN_ACTION_DIM
from mjai.features.state_features import TURN_STATE_DIM
from mjai.models.policy_net import MahjongPolicyNet
from mjai.training.losses import turn_policy_loss
from mjai.training.metrics import top1_accuracy
from mjai.utils.config import dump_yaml
from mjai.utils.device import normalize_device_name
from mjai.utils.paths import ensure_dir
from mjai.utils.seed import seed_everything


@dataclass(slots=True, frozen=True)
class EpochSummary:
    epoch: int
    split: str
    loss: float
    accuracy: float
    batches: int
    samples: int


@dataclass(slots=True, frozen=True)
class TrainSummary:
    task: str
    epochs: int
    train_loss: float
    train_accuracy: float
    valid_loss: float | None
    valid_accuracy: float | None
    train_samples: int
    valid_samples: int
    best_checkpoint: str
    final_checkpoint: str
    used_device: str


def train_turn_model(config: dict[str, Any]) -> TrainSummary:
    task_name = str(config.get("task", {}).get("task", ""))
    if task_name != "turn":
        raise ValueError(f"minimal training path currently supports only task=turn, got: {task_name}")

    seed_everything(int(config.get("seed", 20260423)))

    output_dir = ensure_dir(config.get("output_dir", "outputs/checkpoints/turn_smoke"))
    log_dir = ensure_dir(config.get("log_dir", "outputs/logs/turn_smoke"))
    dump_yaml(output_dir / "config.resolved.yaml", config)

    train_dataset = ManifestTurnDataset(config["data"]["train_manifest"])
    valid_manifest = config.get("data", {}).get("valid_manifest")
    valid_dataset = ManifestTurnDataset(valid_manifest) if valid_manifest and Path(valid_manifest).exists() else None
    candidate_limit = int(config.get("data", {}).get("candidate_limit", 64))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.get("train", {}).get("batch_size", 32)),
        shuffle=True,
        num_workers=int(config.get("train", {}).get("num_workers", 0)),
        collate_fn=lambda rows: build_turn_batch(rows, candidate_limit=candidate_limit),
    )
    valid_loader = None
    if valid_dataset is not None and len(valid_dataset) > 0:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(config.get("train", {}).get("batch_size", 32)),
            shuffle=False,
            num_workers=0,
            collate_fn=lambda rows: build_turn_batch(rows, candidate_limit=candidate_limit),
        )

    device = _resolve_training_device(config.get("train", {}).get("device"))
    model = MahjongPolicyNet(
        state_dim=TURN_STATE_DIM,
        action_dim=TURN_ACTION_DIM,
        hidden_size=int(config.get("model", {}).get("hidden_size", 128)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(config.get("train", {}).get("lr", 1e-3)),
        weight_decay=float(config.get("train", {}).get("weight_decay", 0.0)),
    )

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
            train=True,
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
                train=False,
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
        task="turn",
        epochs=epochs,
        train_loss=last_train_summary.loss,
        train_accuracy=last_train_summary.accuracy,
        valid_loss=None if last_valid_summary is None else last_valid_summary.loss,
        valid_accuracy=None if last_valid_summary is None else last_valid_summary.accuracy,
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
    train: bool,
) -> EpochSummary:
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batches = 0

    model.train(train)
    for batch in data_loader:
        batches += 1
        prepared_batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

        logits = model.forward_turn(prepared_batch)
        loss = turn_policy_loss(logits, prepared_batch["labels"])

        if train:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_size = prepared_batch["labels"].shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == prepared_batch["labels"]).sum().item())
        total_samples += batch_size

    average_loss = 0.0 if total_samples == 0 else total_loss / total_samples
    accuracy = 0.0 if total_samples == 0 else total_correct / total_samples
    return EpochSummary(
        epoch=epoch,
        split=split,
        loss=average_loss,
        accuracy=accuracy,
        batches=batches,
        samples=total_samples,
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
