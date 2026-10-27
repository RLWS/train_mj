from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from mjai.features.compatibility import require_checkpoint_schema

import torch
from torch.utils.data import DataLoader

from mjai.data.datasets.parquet_dataset import StreamingManifestTurnDataset, TurnBatchCollator
from mjai.eval.offline import (
    _batch_size_for_task,
    _build_top_predictions,
    _candidate_limit_for_task,
    _compute_label_ranks,
    _merge_dicts,
    _resolve_bronze_dir,
    _resolve_dataset_dir,
    _resolve_device,
    _resolve_eval_num_workers,
    _resolve_manifest_path,
    _worker_loader_kwargs,
)
from mjai.features.action_features import TURN_ACTION_DIM
from mjai.features.state_features import TURN_STATE_DIM
from mjai.models.multitask_policy_net import MultitaskMahjongPolicyNet
from mjai.models.policy_net import MahjongPolicyNet
from mjai.utils.paths import ensure_dir


DEFAULT_TURN_ERROR_LABEL_KINDS = ("discard",)
DEFAULT_TURN_ERROR_RANKS = (2, 3)


@dataclass(slots=True, frozen=True)
class TurnErrorSample:
    sample_id: str
    target: str
    label_rank: int
    label_kind: str
    predicted_kind: str
    candidate_count: int
    label_index: int
    predicted_index: int
    game_id: str
    kyoku_index: int
    step_index: int
    actor: int
    decision_type: str
    label_action: dict[str, Any]
    predicted_action: dict[str, Any]
    top_predictions: list[dict[str, Any]]
    legal_actions: list[dict[str, Any]]
    inspect_sample_command: str
    dump_case_command: str


@dataclass(slots=True, frozen=True)
class TurnErrorSamplingSummary:
    task: str
    split: str
    checkpoint_path: str
    manifest_path: str
    output_path: str
    used_device: str
    seed: int
    per_rank: int
    label_kinds: list[str]
    ranks: list[int]
    targets: list[str]
    samples_seen: int
    matches_seen: dict[str, int]
    sampled_counts: dict[str, int]


def sample_turn_errors(
    ckpt_path: str | Path,
    split: str,
    output_path: str | Path,
    *,
    label_kinds: list[str] | None = None,
    ranks: list[int] | None = None,
    per_rank: int = 100,
    seed: int | None = None,
    config_override: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    progress_every_n_batches: int = 0,
) -> TurnErrorSamplingSummary:
    checkpoint_path = Path(ckpt_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require_checkpoint_schema(checkpoint)
    base_config = checkpoint.get("config")
    if not isinstance(base_config, dict):
        raise TypeError(f"checkpoint config is missing or invalid: {checkpoint_path}")

    config = _merge_dicts(base_config, config_override or {})
    checkpoint_task_name = str(base_config.get("task", {}).get("task", ""))
    if checkpoint_task_name not in {"turn", "multitask"}:
        raise ValueError(
            "turn error sampling requires task=turn or task=multitask checkpoint, "
            f"got: {checkpoint_task_name}"
        )
    task_name = "turn"

    resolved_label_kinds = _normalize_label_kinds(label_kinds)
    resolved_ranks = _normalize_ranks(ranks)
    resolved_per_rank = _normalize_per_rank(per_rank)
    resolved_seed = int(config.get("seed", 20260423) if seed is None else seed)
    resolved_manifest_path = _resolve_sampling_manifest_path(config, split, manifest_path)
    resolved_output_path = Path(output_path)
    candidate_limit = _candidate_limit_for_task(config, "turn")
    batch_size = _batch_size_for_task(config, "turn")
    device = _resolve_device(config.get("train", {}).get("device"))
    num_workers = _resolve_eval_num_workers(config)
    pin_memory = bool(config.get("eval", {}).get("pin_memory", device.type == "cuda"))
    non_blocking = bool(pin_memory and device.type == "cuda")
    dataset_dir = _resolve_dataset_dir(resolved_manifest_path, "turn")
    bronze_dir = _resolve_bronze_dir(config)

    dataset = StreamingManifestTurnDataset(
        resolved_manifest_path,
        shuffle_shards=False,
        shuffle_rows=False,
        shuffle_buffer_size=0,
        include_legal_actions=True,
        include_label_action=True,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"manifest has 0 turn samples for split '{split}': {resolved_manifest_path}"
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=TurnBatchCollator(candidate_limit=candidate_limit),
        pin_memory=pin_memory,
        **_worker_loader_kwargs(config, num_workers),
    )

    if checkpoint_task_name == "multitask":
        model = MultitaskMahjongPolicyNet(
            hidden_size=int(config.get("model", {}).get("hidden_size", 512)),
            dropout=float(config.get("model", {}).get("dropout", 0.1)),
        ).to(device)
    else:
        model = MahjongPolicyNet(
            state_dim=TURN_STATE_DIM,
            action_dim=TURN_ACTION_DIM,
            hidden_size=int(config.get("model", {}).get("hidden_size", 128)),
            dropout=float(config.get("model", {}).get("dropout", 0.1)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = random.Random(resolved_seed)
    sampler = _TurnErrorReservoirSampler(
        label_kinds=resolved_label_kinds,
        ranks=resolved_ranks,
        per_rank=resolved_per_rank,
        rng=rng,
    )

    total_batches = len(data_loader)
    samples_seen = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, start=1):
            prepared_batch = {
                key: value.to(device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            logits = model.forward_turn(prepared_batch)
            samples_seen += int(prepared_batch["labels"].shape[0])
            sampler.update_batch(
                logits=logits,
                batch=prepared_batch,
                dataset_dir=dataset_dir,
                bronze_dir=bronze_dir,
            )
            if progress_callback is not None and _should_report_progress(
                batch_index,
                total_batches,
                progress_every_n_batches,
            ):
                progress_callback(batch_index, total_batches, samples_seen)

    ensure_dir(resolved_output_path.parent)
    samples = sampler.samples()
    with resolved_output_path.open("w", encoding="utf-8") as output_file:
        for sample in samples:
            output_file.write(json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True))
            output_file.write("\n")

    return TurnErrorSamplingSummary(
        task=task_name,
        split=split,
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(resolved_manifest_path),
        output_path=str(resolved_output_path),
        used_device=str(device),
        seed=resolved_seed,
        per_rank=resolved_per_rank,
        label_kinds=list(resolved_label_kinds),
        ranks=list(resolved_ranks),
        targets=list(sampler.targets),
        samples_seen=samples_seen,
        matches_seen=sampler.matches_seen(),
        sampled_counts=sampler.sampled_counts(),
    )


class _TurnErrorReservoirSampler:
    def __init__(
        self,
        *,
        label_kinds: tuple[str, ...],
        ranks: tuple[int, ...],
        per_rank: int,
        rng: random.Random,
    ) -> None:
        self.label_kinds = set(label_kinds)
        self.ranks = set(ranks)
        self.targets = tuple(
            _target_name(label_kind, rank) for label_kind in label_kinds for rank in ranks
        )
        self.per_rank = per_rank
        self.rng = rng
        self._matches_seen = {target: 0 for target in self.targets}
        self._samples = {target: [] for target in self.targets}

    def update_batch(
        self,
        *,
        logits: torch.Tensor,
        batch: dict[str, Any],
        dataset_dir: Path,
        bronze_dir: Path,
    ) -> None:
        label_ranks = _compute_label_ranks(logits, batch["labels"])
        predicted_indices = logits.argmax(dim=1)
        for batch_index, label_kind in enumerate(batch["label_kinds"]):
            label_kind = str(label_kind)
            label_rank = int(label_ranks[batch_index].item())
            if label_kind not in self.label_kinds or label_rank not in self.ranks:
                continue

            target = _target_name(label_kind, label_rank)
            legal_actions = batch["legal_actions"][batch_index]
            predicted_index = int(predicted_indices[batch_index].item())
            predicted_action = legal_actions[predicted_index]
            self._matches_seen[target] += 1
            sample = _build_error_sample(
                target=target,
                label_rank=label_rank,
                label_kind=label_kind,
                predicted_kind=str(predicted_action.get("kind", "unknown")),
                predicted_index=predicted_index,
                logits=logits[batch_index],
                batch=batch,
                batch_index=batch_index,
                legal_actions=legal_actions,
                predicted_action=predicted_action,
                dataset_dir=dataset_dir,
                bronze_dir=bronze_dir,
            )
            reservoir = self._samples[target]
            if len(reservoir) < self.per_rank:
                reservoir.append(sample)
                continue
            replacement_index = self.rng.randrange(self._matches_seen[target])
            if replacement_index < self.per_rank:
                reservoir[replacement_index] = sample

    def samples(self) -> list[TurnErrorSample]:
        samples: list[TurnErrorSample] = []
        for target in self.targets:
            samples.extend(self._samples[target])
        return samples

    def matches_seen(self) -> dict[str, int]:
        return dict(self._matches_seen)

    def sampled_counts(self) -> dict[str, int]:
        return {target: len(samples) for target, samples in self._samples.items()}


def _build_error_sample(
    *,
    target: str,
    label_rank: int,
    label_kind: str,
    predicted_kind: str,
    predicted_index: int,
    logits: torch.Tensor,
    batch: dict[str, Any],
    batch_index: int,
    legal_actions: list[dict[str, Any]],
    predicted_action: dict[str, Any],
    dataset_dir: Path,
    bronze_dir: Path,
) -> TurnErrorSample:
    sample_id = str(batch["sample_ids"][batch_index])
    game_id = str(batch["game_ids"][batch_index])
    kyoku_index = int(batch["kyoku_indices"][batch_index])
    step_index = int(batch["step_indices"][batch_index])
    return TurnErrorSample(
        sample_id=sample_id,
        target=target,
        label_rank=label_rank,
        label_kind=label_kind,
        predicted_kind=predicted_kind,
        candidate_count=int(batch["candidate_counts"][batch_index]),
        label_index=int(batch["labels"][batch_index].item()),
        predicted_index=predicted_index,
        game_id=game_id,
        kyoku_index=kyoku_index,
        step_index=step_index,
        actor=int(batch["actors"][batch_index]),
        decision_type=str(batch["decision_types"][batch_index]),
        label_action=dict(batch["label_actions"][batch_index]),
        predicted_action=dict(predicted_action),
        top_predictions=_build_top_predictions(logits, legal_actions),
        legal_actions=list(legal_actions),
        inspect_sample_command=(
            f"python -m mjai.cli inspect-sample --dataset {dataset_dir.as_posix()} "
            f"--sample-id {sample_id}"
        ),
        dump_case_command=(
            f"python -m mjai.cli dump-case --bronze-dir {bronze_dir.as_posix()} "
            f"--game-id {game_id} --kyoku-id {kyoku_index} --step-id {step_index}"
        ),
    )


def _normalize_label_kinds(label_kinds: list[str] | None) -> tuple[str, ...]:
    if label_kinds is None:
        return DEFAULT_TURN_ERROR_LABEL_KINDS
    normalized = tuple(
        dict.fromkeys(label_kind.strip() for label_kind in label_kinds if label_kind.strip())
    )
    if not normalized:
        raise ValueError("turn error sampling requires at least one label kind")
    return normalized


def _normalize_ranks(ranks: list[int] | None) -> tuple[int, ...]:
    if ranks is None:
        return DEFAULT_TURN_ERROR_RANKS
    normalized = tuple(dict.fromkeys(int(rank) for rank in ranks))
    if not normalized:
        raise ValueError("turn error sampling requires at least one label rank")
    for rank in normalized:
        if rank <= 0:
            raise ValueError(f"label ranks must be positive, got: {rank}")
    return normalized


def _normalize_per_rank(per_rank: int) -> int:
    value = int(per_rank)
    if value <= 0:
        raise ValueError(f"per_rank must be positive, got: {value}")
    return value


def _resolve_sampling_manifest_path(
    config: dict[str, Any],
    split: str,
    manifest_path: str | Path | None,
) -> Path:
    if manifest_path is None:
        return _resolve_manifest_path(config, split)
    resolved = Path(manifest_path)
    if not resolved.exists():
        raise FileNotFoundError(f"manifest file not found for split '{split}': {resolved}")
    return resolved


def _should_report_progress(batch_index: int, total_batches: int, every_n_batches: int) -> bool:
    if every_n_batches <= 0:
        return False
    return batch_index == 1 or batch_index == total_batches or batch_index % every_n_batches == 0


def _target_name(label_kind: str, rank: int) -> str:
    return f"{label_kind}@rank{rank}"
