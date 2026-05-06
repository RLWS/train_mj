from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from mjai.data.datasets.parquet_dataset import ResponseBatchCollator, StreamingManifestResponseDataset
from mjai.eval.offline import _merge_dicts, _resolve_device, _resolve_manifest_path
from mjai.features.action_features import RESPONSE_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM
from mjai.models.policy_net import MahjongPolicyNet
from mjai.training.metrics import RESPONSE_KIND_RANK
from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class ResponseCalibrationPoint:
    pass_bias: float
    equivalent_call_margin: float
    samples: int
    top1_accuracy: float
    multi_candidate_samples: int
    multi_candidate_top1_accuracy: float
    actual_pass: int
    actual_non_pass: int
    predicted_pass: int
    predicted_non_pass: int
    pass_precision: float
    pass_recall: float
    pass_f1: float
    non_pass_precision: float
    non_pass_recall: float
    non_pass_f1: float
    false_call_count: int
    missed_call_count: int
    false_call_rate: float
    missed_call_rate: float
    per_kind_samples: dict[str, int]
    per_kind_top1_accuracy: dict[str, float]


@dataclass(slots=True, frozen=True)
class ResponseCalibrationSelection:
    false_call_rate_target: float
    selected_pass_bias: float | None
    selected_equivalent_call_margin: float | None
    selected_non_pass_f1: float | None
    selected_non_pass_precision: float | None
    selected_non_pass_recall: float | None
    selected_false_call_rate: float | None
    selected_missed_call_rate: float | None


@dataclass(slots=True, frozen=True)
class ResponseCalibrationSummary:
    task: str
    split: str
    checkpoint_path: str
    manifest_path: str
    report_path: str
    used_device: str
    pass_biases: list[float]
    false_call_rate_targets: list[float]
    points: list[ResponseCalibrationPoint]
    selections: list[ResponseCalibrationSelection]


def calibrate_response_checkpoint(
    ckpt_path: str | Path,
    split: str,
    pass_biases: list[float],
    false_call_rate_targets: list[float],
    config_override: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    progress_every_n_batches: int = 0,
) -> ResponseCalibrationSummary:
    pass_biases = _normalize_pass_biases(pass_biases)
    false_call_rate_targets = _normalize_false_call_rate_targets(false_call_rate_targets)

    checkpoint_path = Path(ckpt_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_config = checkpoint.get("config")
    if not isinstance(base_config, dict):
        raise TypeError(f"checkpoint config is missing or invalid: {checkpoint_path}")

    config = _merge_dicts(base_config, config_override or {})
    task_name = str(config.get("task", {}).get("task", ""))
    if task_name != "response":
        raise ValueError(f"response calibration requires task=response checkpoint, got: {task_name}")

    resolved_manifest_path = _resolve_calibration_manifest_path(config, split, manifest_path)
    candidate_limit = int(config.get("data", {}).get("candidate_limit", 16))
    batch_size = int(config.get("train", {}).get("batch_size", 32))
    device = _resolve_device(config.get("train", {}).get("device"))

    dataset = _build_calibration_dataset(resolved_manifest_path)
    if len(dataset) == 0:
        raise ValueError(
            f"manifest has 0 response samples for split '{split}': {resolved_manifest_path}"
        )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ResponseBatchCollator(candidate_limit=candidate_limit),
    )

    model = MahjongPolicyNet(
        state_dim=RESPONSE_STATE_DIM,
        action_dim=RESPONSE_ACTION_DIM,
        hidden_size=int(config.get("model", {}).get("hidden_size", 128)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    resolved_report_path = _resolve_calibration_report_path(checkpoint_path, split, report_path)
    scanner = _ResponseCalibrationScanner(pass_biases=pass_biases)

    total_batches = len(data_loader)
    samples_processed = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, start=1):
            prepared_batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            logits = model.forward_response(prepared_batch)
            scanner.update(logits=logits, batch=prepared_batch)
            samples_processed += int(prepared_batch["labels"].shape[0])
            if progress_callback is not None and _should_report_progress(
                batch_index,
                total_batches,
                progress_every_n_batches,
            ):
                progress_callback(batch_index, total_batches, samples_processed)

    points = scanner.points()
    selections = _select_operating_points(points, false_call_rate_targets)
    summary = ResponseCalibrationSummary(
        task=task_name,
        split=split,
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(resolved_manifest_path),
        report_path=str(resolved_report_path),
        used_device=str(device),
        pass_biases=[float(bias) for bias in pass_biases],
        false_call_rate_targets=[float(target) for target in false_call_rate_targets],
        points=points,
        selections=selections,
    )
    ensure_dir(resolved_report_path.parent)
    resolved_report_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


class _ResponseCalibrationScanner:
    def __init__(self, *, pass_biases: list[float]) -> None:
        self.pass_biases = [float(bias) for bias in pass_biases]
        self._bias_count = len(pass_biases)
        self._samples = 0
        self._multi_candidate_samples = 0
        self._actual_pass = 0
        self._actual_non_pass = 0
        self._top1_hits = [0] * self._bias_count
        self._multi_top1_hits = [0] * self._bias_count
        self._predicted_pass = [0] * self._bias_count
        self._predicted_non_pass = [0] * self._bias_count
        self._pass_tp = [0] * self._bias_count
        self._non_pass_tp = [0] * self._bias_count
        self._false_call = [0] * self._bias_count
        self._missed_call = [0] * self._bias_count
        self._per_kind_samples: dict[str, int] = {}
        self._per_kind_top1_hits: dict[str, list[int]] = {}

    def update(self, *, logits: torch.Tensor, batch: dict[str, Any]) -> None:
        device = logits.device
        labels = batch["labels"]
        candidate_mask = batch["candidate_mask"]
        candidate_counts = torch.as_tensor(batch["candidate_counts"], device=device)
        label_kind_ids = batch["label_kind_ids"]
        candidate_kind_ids = _candidate_kind_ids(batch["legal_actions"], logits.shape[1], device=device)

        pass_kind_id = RESPONSE_KIND_RANK["pass"]
        pass_mask = candidate_mask & (candidate_kind_ids == pass_kind_id)
        non_pass_mask = candidate_mask & ~pass_mask
        pass_exists = pass_mask.any(dim=1)
        non_pass_exists = non_pass_mask.any(dim=1)

        negative_inf = torch.finfo(logits.dtype).min
        pass_scores, pass_indices = logits.masked_fill(~pass_mask, negative_inf).max(dim=1)
        non_pass_scores, non_pass_indices = logits.masked_fill(~non_pass_mask, negative_inf).max(dim=1)
        score_delta = non_pass_scores - pass_scores

        bias_tensor = torch.tensor(self.pass_biases, dtype=logits.dtype, device=device).unsqueeze(1)
        choose_non_pass = non_pass_exists.unsqueeze(0) & (
            (~pass_exists).unsqueeze(0) | (score_delta.unsqueeze(0) > bias_tensor)
        )
        predicted_indices = torch.where(
            choose_non_pass,
            non_pass_indices.unsqueeze(0),
            pass_indices.unsqueeze(0),
        )

        pass_pred_kind_ids = candidate_kind_ids.gather(1, pass_indices.unsqueeze(1)).squeeze(1)
        non_pass_pred_kind_ids = candidate_kind_ids.gather(1, non_pass_indices.unsqueeze(1)).squeeze(1)
        predicted_kind_ids = torch.where(
            choose_non_pass,
            non_pass_pred_kind_ids.unsqueeze(0),
            pass_pred_kind_ids.unsqueeze(0),
        )

        label_is_pass = label_kind_ids == pass_kind_id
        label_is_non_pass = ~label_is_pass
        predicted_is_pass = predicted_kind_ids == pass_kind_id
        predicted_is_non_pass = ~predicted_is_pass
        exact = predicted_indices == labels.unsqueeze(0)
        multi_candidate = candidate_counts > 1

        batch_size = int(labels.shape[0])
        self._samples += batch_size
        self._multi_candidate_samples += int(multi_candidate.sum().item())
        self._actual_pass += int(label_is_pass.sum().item())
        self._actual_non_pass += int(label_is_non_pass.sum().item())

        self._add_vector(self._top1_hits, exact.sum(dim=1))
        self._add_vector(self._multi_top1_hits, (exact & multi_candidate.unsqueeze(0)).sum(dim=1))
        self._add_vector(self._predicted_pass, predicted_is_pass.sum(dim=1))
        self._add_vector(self._predicted_non_pass, predicted_is_non_pass.sum(dim=1))
        self._add_vector(self._pass_tp, (label_is_pass.unsqueeze(0) & predicted_is_pass).sum(dim=1))
        self._add_vector(
            self._non_pass_tp,
            (label_is_non_pass.unsqueeze(0) & predicted_is_non_pass).sum(dim=1),
        )
        self._add_vector(
            self._false_call,
            (label_is_pass.unsqueeze(0) & predicted_is_non_pass).sum(dim=1),
        )
        self._add_vector(
            self._missed_call,
            (label_is_non_pass.unsqueeze(0) & predicted_is_pass).sum(dim=1),
        )

        for kind, kind_id in RESPONSE_KIND_RANK.items():
            if kind == "unknown":
                continue
            kind_mask = label_kind_ids == kind_id
            kind_samples = int(kind_mask.sum().item())
            if kind_samples == 0:
                continue
            self._per_kind_samples[kind] = self._per_kind_samples.get(kind, 0) + kind_samples
            self._per_kind_top1_hits.setdefault(kind, [0] * self._bias_count)
            self._add_vector(
                self._per_kind_top1_hits[kind],
                (exact & kind_mask.unsqueeze(0)).sum(dim=1),
            )

    def points(self) -> list[ResponseCalibrationPoint]:
        points: list[ResponseCalibrationPoint] = []
        for index, pass_bias in enumerate(self.pass_biases):
            pass_precision = _safe_divide(self._pass_tp[index], self._predicted_pass[index])
            pass_recall = _safe_divide(self._pass_tp[index], self._actual_pass)
            non_pass_precision = _safe_divide(
                self._non_pass_tp[index],
                self._predicted_non_pass[index],
            )
            non_pass_recall = _safe_divide(self._non_pass_tp[index], self._actual_non_pass)
            points.append(
                ResponseCalibrationPoint(
                    pass_bias=pass_bias,
                    equivalent_call_margin=pass_bias,
                    samples=self._samples,
                    top1_accuracy=_safe_divide(self._top1_hits[index], self._samples),
                    multi_candidate_samples=self._multi_candidate_samples,
                    multi_candidate_top1_accuracy=_safe_divide(
                        self._multi_top1_hits[index],
                        self._multi_candidate_samples,
                    ),
                    actual_pass=self._actual_pass,
                    actual_non_pass=self._actual_non_pass,
                    predicted_pass=self._predicted_pass[index],
                    predicted_non_pass=self._predicted_non_pass[index],
                    pass_precision=pass_precision,
                    pass_recall=pass_recall,
                    pass_f1=_f1(pass_precision, pass_recall),
                    non_pass_precision=non_pass_precision,
                    non_pass_recall=non_pass_recall,
                    non_pass_f1=_f1(non_pass_precision, non_pass_recall),
                    false_call_count=self._false_call[index],
                    missed_call_count=self._missed_call[index],
                    false_call_rate=_safe_divide(self._false_call[index], self._actual_pass),
                    missed_call_rate=_safe_divide(self._missed_call[index], self._actual_non_pass),
                    per_kind_samples=dict(sorted(self._per_kind_samples.items())),
                    per_kind_top1_accuracy={
                        kind: _safe_divide(hits[index], self._per_kind_samples[kind])
                        for kind, hits in sorted(self._per_kind_top1_hits.items())
                    },
                )
            )
        return points

    @staticmethod
    def _add_vector(target: list[int], values: torch.Tensor) -> None:
        for index, value in enumerate(values.detach().to("cpu").tolist()):
            target[index] += int(value)


def _candidate_kind_ids(
    legal_actions: list[list[dict[str, Any]]],
    candidate_width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    unknown_id = RESPONSE_KIND_RANK["unknown"]
    rows: list[list[int]] = []
    for actions in legal_actions:
        row = [
            RESPONSE_KIND_RANK.get(str(action.get("kind", "unknown")), unknown_id)
            for action in actions
        ]
        row.extend([unknown_id] * (candidate_width - len(row)))
        rows.append(row[:candidate_width])
    return torch.tensor(rows, dtype=torch.long, device=device)


def _select_operating_points(
    points: list[ResponseCalibrationPoint],
    false_call_rate_targets: list[float],
) -> list[ResponseCalibrationSelection]:
    selections: list[ResponseCalibrationSelection] = []
    for target in false_call_rate_targets:
        eligible = [point for point in points if point.false_call_rate <= target]
        if not eligible:
            selections.append(
                ResponseCalibrationSelection(
                    false_call_rate_target=float(target),
                    selected_pass_bias=None,
                    selected_equivalent_call_margin=None,
                    selected_non_pass_f1=None,
                    selected_non_pass_precision=None,
                    selected_non_pass_recall=None,
                    selected_false_call_rate=None,
                    selected_missed_call_rate=None,
                )
            )
            continue
        selected = max(
            eligible,
            key=lambda point: (
                point.non_pass_f1,
                point.non_pass_recall,
                -point.false_call_rate,
            ),
        )
        selections.append(
            ResponseCalibrationSelection(
                false_call_rate_target=float(target),
                selected_pass_bias=selected.pass_bias,
                selected_equivalent_call_margin=selected.equivalent_call_margin,
                selected_non_pass_f1=selected.non_pass_f1,
                selected_non_pass_precision=selected.non_pass_precision,
                selected_non_pass_recall=selected.non_pass_recall,
                selected_false_call_rate=selected.false_call_rate,
                selected_missed_call_rate=selected.missed_call_rate,
            )
        )
    return selections


def _build_calibration_dataset(manifest_path: Path) -> StreamingManifestResponseDataset:
    return StreamingManifestResponseDataset(
        manifest_path,
        min_candidate_count=1,
        shuffle_shards=False,
        shuffle_rows=False,
        shuffle_buffer_size=0,
        include_legal_actions=True,
    )


def _normalize_pass_biases(pass_biases: list[float]) -> list[float]:
    if not pass_biases:
        raise ValueError("calibration requires at least one pass bias")
    normalized = [float(bias) for bias in pass_biases]
    for bias in normalized:
        if not math.isfinite(bias):
            raise ValueError(f"pass bias must be finite, got: {bias}")
    return normalized


def _normalize_false_call_rate_targets(targets: list[float]) -> list[float]:
    if not targets:
        raise ValueError("calibration requires at least one false call rate target")
    normalized = [float(target) for target in targets]
    for target in normalized:
        if not math.isfinite(target) or target < 0.0 or target > 1.0:
            raise ValueError(f"false call rate targets must be finite rates in [0, 1], got: {target}")
    return normalized


def _resolve_calibration_report_path(
    checkpoint_path: Path,
    split: str,
    report_path: str | Path | None,
) -> Path:
    if report_path is not None:
        return Path(report_path)
    return checkpoint_path.parent / f"response_calibration_{split}.json"


def _resolve_calibration_manifest_path(
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


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
