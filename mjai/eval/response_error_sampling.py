from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from mjai.data.datasets.parquet_dataset import (
    ResponseBatchCollator,
    StreamingManifestResponseDataset,
)
from mjai.eval.offline import (
    _build_top_predictions,
    _merge_dicts,
    _resolve_bronze_dir,
    _resolve_dataset_dir,
    _resolve_device,
    _resolve_eval_num_workers,
    _resolve_manifest_path,
    _worker_loader_kwargs,
)
from mjai.eval.response_calibration import _candidate_kind_ids
from mjai.features.action_features import RESPONSE_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM
from mjai.models.policy_net import MahjongPolicyNet
from mjai.training.metrics import RESPONSE_KIND_RANK
from mjai.utils.paths import ensure_dir


DEFAULT_RESPONSE_ERROR_TARGETS = ("pass->chi", "pass->pon", "chi->pass", "pon->pass")


@dataclass(slots=True, frozen=True)
class ResponseErrorSample:
    sample_id: str
    confusion: str
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
class ResponseErrorSamplingSummary:
    task: str
    split: str
    checkpoint_path: str
    manifest_path: str
    output_path: str
    used_device: str
    seed: int
    pass_bias: float
    per_target: int
    targets: list[str]
    samples_seen: int
    matches_seen: dict[str, int]
    sampled_counts: dict[str, int]


def sample_response_errors(
    ckpt_path: str | Path,
    split: str,
    output_path: str | Path,
    *,
    targets: list[str] | None = None,
    per_target: int = 100,
    pass_bias: float = 0.0,
    seed: int | None = None,
    config_override: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    progress_every_n_batches: int = 0,
) -> ResponseErrorSamplingSummary:
    checkpoint_path = Path(ckpt_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_config = checkpoint.get("config")
    if not isinstance(base_config, dict):
        raise TypeError(f"checkpoint config is missing or invalid: {checkpoint_path}")

    config = _merge_dicts(base_config, config_override or {})
    task_name = str(config.get("task", {}).get("task", ""))
    if task_name != "response":
        raise ValueError(
            f"response error sampling requires task=response checkpoint, got: {task_name}"
        )

    resolved_targets = _normalize_targets(targets)
    resolved_per_target = _normalize_per_target(per_target)
    resolved_pass_bias = _normalize_pass_bias(pass_bias)
    resolved_seed = int(config.get("seed", 20260423) if seed is None else seed)
    resolved_manifest_path = _resolve_sampling_manifest_path(config, split, manifest_path)
    resolved_output_path = Path(output_path)
    candidate_limit = int(config.get("data", {}).get("candidate_limit", 16))
    batch_size = int(config.get("train", {}).get("batch_size", 32))
    device = _resolve_device(config.get("train", {}).get("device"))
    num_workers = _resolve_eval_num_workers(config)
    pin_memory = bool(config.get("eval", {}).get("pin_memory", device.type == "cuda"))
    non_blocking = bool(pin_memory and device.type == "cuda")
    dataset_dir = _resolve_dataset_dir(resolved_manifest_path, "response")
    bronze_dir = _resolve_bronze_dir(config)

    dataset = StreamingManifestResponseDataset(
        resolved_manifest_path,
        min_candidate_count=1,
        shuffle_shards=False,
        shuffle_rows=False,
        shuffle_buffer_size=0,
        include_legal_actions=True,
        include_label_action=True,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"manifest has 0 response samples for split '{split}': {resolved_manifest_path}"
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ResponseBatchCollator(candidate_limit=candidate_limit),
        pin_memory=pin_memory,
        **_worker_loader_kwargs(config, num_workers),
    )

    model = MahjongPolicyNet(
        state_dim=RESPONSE_STATE_DIM,
        action_dim=RESPONSE_ACTION_DIM,
        hidden_size=int(config.get("model", {}).get("hidden_size", 128)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = random.Random(resolved_seed)
    sampler = _ResponseErrorReservoirSampler(
        targets=resolved_targets,
        per_target=resolved_per_target,
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
            logits = model.forward_response(prepared_batch)
            predicted_indices = _select_response_predictions(
                logits=logits,
                batch=prepared_batch,
                pass_bias=resolved_pass_bias,
            )
            samples_seen += int(prepared_batch["labels"].shape[0])
            sampler.update_batch(
                logits=logits,
                batch=prepared_batch,
                predicted_indices=predicted_indices,
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

    return ResponseErrorSamplingSummary(
        task=task_name,
        split=split,
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(resolved_manifest_path),
        output_path=str(resolved_output_path),
        used_device=str(device),
        seed=resolved_seed,
        pass_bias=resolved_pass_bias,
        per_target=resolved_per_target,
        targets=list(resolved_targets),
        samples_seen=samples_seen,
        matches_seen=sampler.matches_seen(),
        sampled_counts=sampler.sampled_counts(),
    )


class _ResponseErrorReservoirSampler:
    def __init__(self, *, targets: tuple[str, ...], per_target: int, rng: random.Random) -> None:
        self.targets = targets
        self.per_target = per_target
        self.rng = rng
        self._matches_seen = {target: 0 for target in targets}
        self._samples = {target: [] for target in targets}

    def update_batch(
        self,
        *,
        logits: torch.Tensor,
        batch: dict[str, Any],
        predicted_indices: torch.Tensor,
        dataset_dir: Path,
        bronze_dir: Path,
    ) -> None:
        for batch_index, label_kind in enumerate(batch["label_kinds"]):
            legal_actions = batch["legal_actions"][batch_index]
            predicted_index = int(predicted_indices[batch_index].item())
            predicted_action = legal_actions[predicted_index]
            predicted_kind = str(predicted_action.get("kind", "unknown"))
            confusion = f"{label_kind}->{predicted_kind}"
            if confusion not in self._samples:
                continue

            self._matches_seen[confusion] += 1
            sample = _build_error_sample(
                confusion=confusion,
                label_kind=str(label_kind),
                predicted_kind=predicted_kind,
                predicted_index=predicted_index,
                logits=logits[batch_index],
                batch=batch,
                batch_index=batch_index,
                legal_actions=legal_actions,
                predicted_action=predicted_action,
                dataset_dir=dataset_dir,
                bronze_dir=bronze_dir,
            )
            reservoir = self._samples[confusion]
            if len(reservoir) < self.per_target:
                reservoir.append(sample)
                continue
            replacement_index = self.rng.randrange(self._matches_seen[confusion])
            if replacement_index < self.per_target:
                reservoir[replacement_index] = sample

    def samples(self) -> list[ResponseErrorSample]:
        samples: list[ResponseErrorSample] = []
        for target in self.targets:
            samples.extend(self._samples[target])
        return samples

    def matches_seen(self) -> dict[str, int]:
        return dict(self._matches_seen)

    def sampled_counts(self) -> dict[str, int]:
        return {target: len(samples) for target, samples in self._samples.items()}


def _select_response_predictions(
    *,
    logits: torch.Tensor,
    batch: dict[str, Any],
    pass_bias: float,
) -> torch.Tensor:
    if pass_bias == 0.0:
        return logits.argmax(dim=1)

    device = logits.device
    candidate_mask = batch["candidate_mask"]
    candidate_kind_ids = _candidate_kind_ids(batch["legal_actions"], logits.shape[1], device=device)
    pass_kind_id = RESPONSE_KIND_RANK["pass"]
    pass_mask = candidate_mask & (candidate_kind_ids == pass_kind_id)
    non_pass_mask = candidate_mask & ~pass_mask
    pass_exists = pass_mask.any(dim=1)
    non_pass_exists = non_pass_mask.any(dim=1)

    negative_inf = torch.finfo(logits.dtype).min
    pass_scores, pass_indices = logits.masked_fill(~pass_mask, negative_inf).max(dim=1)
    non_pass_scores, non_pass_indices = logits.masked_fill(~non_pass_mask, negative_inf).max(dim=1)
    choose_non_pass = non_pass_exists & (
        ~pass_exists | ((non_pass_scores - pass_scores) > pass_bias)
    )
    return torch.where(choose_non_pass, non_pass_indices, pass_indices)


def _build_error_sample(
    *,
    confusion: str,
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
) -> ResponseErrorSample:
    sample_id = str(batch["sample_ids"][batch_index])
    game_id = str(batch["game_ids"][batch_index])
    kyoku_index = int(batch["kyoku_indices"][batch_index])
    step_index = int(batch["step_indices"][batch_index])
    return ResponseErrorSample(
        sample_id=sample_id,
        confusion=confusion,
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


def _normalize_targets(targets: list[str] | None) -> tuple[str, ...]:
    if targets is None:
        return DEFAULT_RESPONSE_ERROR_TARGETS
    normalized = tuple(target.strip() for target in targets if target.strip())
    if not normalized:
        raise ValueError("response error sampling requires at least one target")
    for target in normalized:
        if "->" not in target:
            raise ValueError(f"target must use label->pred format, got: {target}")
        label_kind, predicted_kind = target.split("->", maxsplit=1)
        if not label_kind or not predicted_kind:
            raise ValueError(f"target must use label->pred format, got: {target}")
    return normalized


def _normalize_per_target(per_target: int) -> int:
    value = int(per_target)
    if value <= 0:
        raise ValueError(f"per_target must be positive, got: {value}")
    return value


def _normalize_pass_bias(pass_bias: float) -> float:
    value = float(pass_bias)
    if not math.isfinite(value):
        raise ValueError(f"pass_bias must be finite, got: {value}")
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
