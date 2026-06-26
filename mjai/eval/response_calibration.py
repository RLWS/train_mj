from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from mjai.data.datasets.parquet_dataset import ResponseBatchCollator, StreamingManifestResponseDataset
from mjai.eval.distributed import (
    EvalDistributedContext,
    gather_eval_objects,
    initialize_eval_distributed_context,
    shutdown_eval_distributed_context,
)
from mjai.eval.offline import (
    _batch_size_for_task,
    _candidate_limit_for_task,
    _merge_dicts,
    _resolve_eval_num_workers,
    _resolve_manifest_path,
    _worker_loader_kwargs,
)
from mjai.features.action_features import RESPONSE_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM
from mjai.models.multitask_policy_net import MultitaskMahjongPolicyNet
from mjai.models.policy_net import MahjongPolicyNet
from mjai.training.metrics import RESPONSE_KIND_RANK
from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class ResponseCalibrationPoint:
    pass_bias: float
    no_yaku_call_penalty: float
    equivalent_call_margin: float
    no_yaku_equivalent_call_margin: float
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
    no_yaku_false_call_count: int
    no_yaku_false_call_candidate_count: int
    missed_call_count: int
    false_call_rate: float
    no_yaku_false_call_rate: float
    no_yaku_false_call_share: float
    missed_call_rate: float
    per_kind_samples: dict[str, int]
    per_kind_top1_accuracy: dict[str, float]


@dataclass(slots=True, frozen=True)
class ResponseCalibrationSelection:
    false_call_rate_target: float
    selected_pass_bias: float | None
    selected_no_yaku_call_penalty: float | None
    selected_equivalent_call_margin: float | None
    selected_no_yaku_equivalent_call_margin: float | None
    selected_non_pass_f1: float | None
    selected_non_pass_precision: float | None
    selected_non_pass_recall: float | None
    selected_false_call_rate: float | None
    selected_no_yaku_false_call_rate: float | None
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
    no_yaku_call_penalties: list[float]
    false_call_rate_targets: list[float]
    points: list[ResponseCalibrationPoint]
    selections: list[ResponseCalibrationSelection]


def calibrate_response_checkpoint(
    ckpt_path: str | Path,
    split: str,
    pass_biases: list[float],
    false_call_rate_targets: list[float],
    no_yaku_call_penalties: list[float] | None = None,
    config_override: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    progress_every_n_batches: int = 0,
) -> ResponseCalibrationSummary:
    """扫描响应模型跳过偏置并生成校准结果。
    
    参数:
        ckpt_path (str | Path): 模型检查点路径。
        split (str): 数据划分名称。
        pass_biases (list[float]): 跳过动作偏置列表。
        false_call_rate_targets (list[float]): 误鸣率目标列表。
        config_override (dict[str, Any] | None): 配置覆盖字典。
        report_path (str | Path | None): 报告输出路径。
        manifest_path (str | Path | None): 清单文件路径。
        progress_callback (Callable[[int, int, int], None] | None): 函数执行所需的输入参数。
        progress_every_n_batches (int): 相关数据列表。
    
    返回:
        ResponseCalibrationSummary: 函数执行结果。
    
    异常:
        TypeError: 当输入类型不符合要求时抛出。
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    pass_biases = _normalize_pass_biases(pass_biases)
    no_yaku_call_penalties = _normalize_no_yaku_call_penalties(no_yaku_call_penalties or [0.0])
    false_call_rate_targets = _normalize_false_call_rate_targets(false_call_rate_targets)

    checkpoint_path = Path(ckpt_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_config = checkpoint.get("config")
    if not isinstance(base_config, dict):
        raise TypeError(f"checkpoint config is missing or invalid: {checkpoint_path}")

    config = _merge_dicts(base_config, config_override or {})
    checkpoint_task_name = str(base_config.get("task", {}).get("task", ""))
    if checkpoint_task_name not in {"response", "multitask"}:
        raise ValueError(
            "response calibration requires task=response or task=multitask checkpoint, "
            f"got: {checkpoint_task_name}"
        )
    task_name = "response"

    resolved_manifest_path = _resolve_calibration_manifest_path(config, split, manifest_path)
    candidate_limit = _candidate_limit_for_task(config, "response")
    batch_size = _batch_size_for_task(config, "response")
    resolved_report_path = _resolve_calibration_report_path(checkpoint_path, split, report_path)
    distributed_context = initialize_eval_distributed_context(config)
    try:
        summary = _calibrate_response_checkpoint_inner(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            checkpoint_task_name=checkpoint_task_name,
            task_name=task_name,
            split=split,
            config=config,
            resolved_manifest_path=resolved_manifest_path,
            candidate_limit=candidate_limit,
            batch_size=batch_size,
            pass_biases=pass_biases,
            no_yaku_call_penalties=no_yaku_call_penalties,
            false_call_rate_targets=false_call_rate_targets,
            resolved_report_path=resolved_report_path,
            progress_callback=progress_callback,
            progress_every_n_batches=progress_every_n_batches,
            distributed_context=distributed_context,
        )
        if distributed_context.is_main:
            ensure_dir(resolved_report_path.parent)
            resolved_report_path.write_text(
                json.dumps(asdict(summary), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return summary
    finally:
        shutdown_eval_distributed_context(distributed_context)


def _calibrate_response_checkpoint_inner(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_task_name: str,
    task_name: str,
    split: str,
    config: dict[str, Any],
    resolved_manifest_path: Path,
    candidate_limit: int,
    batch_size: int,
    pass_biases: list[float],
    no_yaku_call_penalties: list[float],
    false_call_rate_targets: list[float],
    resolved_report_path: Path,
    progress_callback: Callable[[int, int, int], None] | None,
    progress_every_n_batches: int,
    distributed_context: EvalDistributedContext,
) -> ResponseCalibrationSummary:
    device = distributed_context.device
    num_workers = _resolve_eval_num_workers(config)
    pin_memory = bool(config.get("eval", {}).get("pin_memory", device.type == "cuda"))
    dataset = _build_calibration_dataset(
        resolved_manifest_path,
        distributed_rank=distributed_context.rank,
        distributed_world_size=distributed_context.world_size,
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

    if checkpoint_task_name == "multitask":
        model = MultitaskMahjongPolicyNet(
            hidden_size=int(config.get("model", {}).get("hidden_size", 512)),
            dropout=float(config.get("model", {}).get("dropout", 0.1)),
        ).to(device)
    else:
        model = MahjongPolicyNet(
            state_dim=RESPONSE_STATE_DIM,
            action_dim=RESPONSE_ACTION_DIM,
            hidden_size=int(config.get("model", {}).get("hidden_size", 128)),
            dropout=float(config.get("model", {}).get("dropout", 0.1)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scanner = _ResponseCalibrationScanner(
        pass_biases=pass_biases,
        no_yaku_call_penalties=no_yaku_call_penalties,
    )

    total_batches = len(data_loader)
    samples_processed = 0
    non_blocking = bool(pin_memory and device.type == "cuda")
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader, start=1):
            prepared_batch = {
                key: value.to(device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            logits = model.forward_response(prepared_batch)
            scanner.update(logits=logits, batch=prepared_batch)
            samples_processed += int(prepared_batch["labels"].shape[0])
            if (
                distributed_context.is_main
                and progress_callback is not None
                and _should_report_progress(
                    batch_index,
                    total_batches,
                    progress_every_n_batches,
                )
            ):
                progress_callback(batch_index, total_batches, samples_processed)

    scanners = gather_eval_objects(scanner, distributed_context)
    merged_scanner = _merge_calibration_scanners(scanners)
    points = merged_scanner.points()
    selections = _select_operating_points(points, false_call_rate_targets)
    return ResponseCalibrationSummary(
        task=task_name,
        split=split,
        checkpoint_path=str(checkpoint_path),
        manifest_path=str(resolved_manifest_path),
        report_path=str(resolved_report_path),
        used_device=str(device),
        pass_biases=[float(bias) for bias in pass_biases],
        no_yaku_call_penalties=[float(penalty) for penalty in no_yaku_call_penalties],
        false_call_rate_targets=[float(target) for target in false_call_rate_targets],
        points=points,
        selections=selections,
    )


class _ResponseCalibrationScanner:
    def __init__(self, *, pass_biases: list[float], no_yaku_call_penalties: list[float]) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
            pass_biases (list[float]): 跳过动作偏置列表。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.pass_biases = [float(bias) for bias in pass_biases]
        self.no_yaku_call_penalties = [float(penalty) for penalty in no_yaku_call_penalties]
        self._points = [
            (pass_bias, penalty)
            for penalty in self.no_yaku_call_penalties
            for pass_bias in self.pass_biases
        ]
        self._point_count = len(self._points)
        self._samples = 0
        self._multi_candidate_samples = 0
        self._actual_pass = 0
        self._actual_non_pass = 0
        self._top1_hits = [0] * self._point_count
        self._multi_top1_hits = [0] * self._point_count
        self._predicted_pass = [0] * self._point_count
        self._predicted_non_pass = [0] * self._point_count
        self._pass_tp = [0] * self._point_count
        self._non_pass_tp = [0] * self._point_count
        self._false_call = [0] * self._point_count
        self._no_yaku_false_call = [0] * self._point_count
        self._no_yaku_false_call_candidate = [0] * self._point_count
        self._missed_call = [0] * self._point_count
        self._per_kind_samples: dict[str, int] = {}
        self._per_kind_top1_hits: dict[str, list[int]] = {}

    def update(self, *, logits: torch.Tensor, batch: dict[str, Any]) -> None:
        """更新统计或状态累积结果。
        
        参数:
            self (Self): 当前对象实例。
            logits (torch.Tensor): 候选动作分数张量。
            batch (dict[str, Any]): 模型批次字典。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        device = logits.device
        labels = batch["labels"]
        candidate_mask = batch["candidate_mask"]
        candidate_feats = batch["candidate_feats"]
        candidate_counts = torch.as_tensor(batch["candidate_counts"], device=device)
        label_kind_ids = batch["label_kind_ids"]
        candidate_kind_ids = _candidate_kind_ids(batch["legal_actions"], logits.shape[1], device=device)
        no_yaku_call_mask = _no_yaku_call_candidate_mask(
            candidate_kind_ids=candidate_kind_ids,
            candidate_feats=candidate_feats,
            candidate_mask=candidate_mask,
        )

        pass_kind_id = RESPONSE_KIND_RANK["pass"]
        pass_mask = candidate_mask & (candidate_kind_ids == pass_kind_id)
        non_pass_mask = candidate_mask & ~pass_mask
        pass_exists = pass_mask.any(dim=1)
        non_pass_exists = non_pass_mask.any(dim=1)

        negative_inf = torch.finfo(logits.dtype).min
        pass_scores, pass_indices = logits.masked_fill(~pass_mask, negative_inf).max(dim=1)
        point_pass_biases = torch.tensor(
            [point[0] for point in self._points],
            dtype=logits.dtype,
            device=device,
        ).unsqueeze(1)
        point_penalties = torch.tensor(
            [point[1] for point in self._points],
            dtype=logits.dtype,
            device=device,
        ).view(-1, 1, 1)
        adjusted_logits = logits.unsqueeze(0) - (no_yaku_call_mask.unsqueeze(0).to(logits.dtype) * point_penalties)
        adjusted_non_pass_scores, non_pass_indices = adjusted_logits.masked_fill(
            ~non_pass_mask.unsqueeze(0),
            negative_inf,
        ).max(dim=2)
        score_delta = adjusted_non_pass_scores - pass_scores.unsqueeze(0)

        choose_non_pass = non_pass_exists.unsqueeze(0) & (
            (~pass_exists).unsqueeze(0) | (score_delta > point_pass_biases)
        )
        predicted_indices = torch.where(
            choose_non_pass,
            non_pass_indices,
            pass_indices.unsqueeze(0),
        )

        pass_pred_kind_ids = candidate_kind_ids.gather(1, pass_indices.unsqueeze(1)).squeeze(1)
        expanded_candidate_kind_ids = candidate_kind_ids.unsqueeze(0).expand(self._point_count, -1, -1)
        non_pass_pred_kind_ids = expanded_candidate_kind_ids.gather(
            2,
            non_pass_indices.unsqueeze(2),
        ).squeeze(2)
        predicted_kind_ids = torch.where(
            choose_non_pass,
            non_pass_pred_kind_ids,
            pass_pred_kind_ids.unsqueeze(0),
        )
        predicted_no_yaku_call = no_yaku_call_mask.unsqueeze(0).expand(
            self._point_count,
            -1,
            -1,
        ).gather(2, predicted_indices.unsqueeze(2)).squeeze(2)

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
            self._no_yaku_false_call,
            (label_is_pass.unsqueeze(0) & predicted_is_non_pass & predicted_no_yaku_call).sum(dim=1),
        )
        self._add_vector(
            self._no_yaku_false_call_candidate,
            (label_is_pass.unsqueeze(0) & predicted_no_yaku_call).sum(dim=1),
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
            self._per_kind_top1_hits.setdefault(kind, [0] * self._point_count)
            self._add_vector(
                self._per_kind_top1_hits[kind],
                (exact & kind_mask.unsqueeze(0)).sum(dim=1),
            )

    def points(self) -> list[ResponseCalibrationPoint]:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            list[ResponseCalibrationPoint]: 列表结果。
        
        异常:
            无。
        """
        points: list[ResponseCalibrationPoint] = []
        for index, (pass_bias, no_yaku_call_penalty) in enumerate(self._points):
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
                    no_yaku_call_penalty=no_yaku_call_penalty,
                    equivalent_call_margin=pass_bias,
                    no_yaku_equivalent_call_margin=pass_bias + no_yaku_call_penalty,
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
                    no_yaku_false_call_count=self._no_yaku_false_call[index],
                    no_yaku_false_call_candidate_count=self._no_yaku_false_call_candidate[index],
                    missed_call_count=self._missed_call[index],
                    false_call_rate=_safe_divide(self._false_call[index], self._actual_pass),
                    no_yaku_false_call_rate=_safe_divide(
                        self._no_yaku_false_call[index],
                        self._actual_pass,
                    ),
                    no_yaku_false_call_share=_safe_divide(
                        self._no_yaku_false_call[index],
                        self._false_call[index],
                    ),
                    missed_call_rate=_safe_divide(self._missed_call[index], self._actual_non_pass),
                    per_kind_samples=dict(sorted(self._per_kind_samples.items())),
                    per_kind_top1_accuracy={
                        kind: _safe_divide(hits[index], self._per_kind_samples[kind])
                        for kind, hits in sorted(self._per_kind_top1_hits.items())
                    },
                )
            )
        return points

    def merge(self, other: "_ResponseCalibrationScanner") -> None:
        if self._points != other._points:
            raise ValueError("cannot merge calibration scanners with different scan points")
        self._samples += other._samples
        self._multi_candidate_samples += other._multi_candidate_samples
        self._actual_pass += other._actual_pass
        self._actual_non_pass += other._actual_non_pass
        self._add_list(self._top1_hits, other._top1_hits)
        self._add_list(self._multi_top1_hits, other._multi_top1_hits)
        self._add_list(self._predicted_pass, other._predicted_pass)
        self._add_list(self._predicted_non_pass, other._predicted_non_pass)
        self._add_list(self._pass_tp, other._pass_tp)
        self._add_list(self._non_pass_tp, other._non_pass_tp)
        self._add_list(self._false_call, other._false_call)
        self._add_list(self._no_yaku_false_call, other._no_yaku_false_call)
        self._add_list(
            self._no_yaku_false_call_candidate,
            other._no_yaku_false_call_candidate,
        )
        self._add_list(self._missed_call, other._missed_call)
        for kind, samples in other._per_kind_samples.items():
            self._per_kind_samples[kind] = self._per_kind_samples.get(kind, 0) + int(samples)
        for kind, hits in other._per_kind_top1_hits.items():
            self._per_kind_top1_hits.setdefault(kind, [0] * self._point_count)
            self._add_list(self._per_kind_top1_hits[kind], hits)

    @staticmethod
    def _add_vector(target: list[int], values: torch.Tensor) -> None:
        """向当前状态添加输入数据。
        
        参数:
            target (list[int]): 函数执行所需的输入参数。
            values (torch.Tensor): 相关数据列表。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        for index, value in enumerate(values.detach().to("cpu").tolist()):
            target[index] += int(value)

    @staticmethod
    def _add_list(target: list[int], values: list[int]) -> None:
        for index, value in enumerate(values):
            target[index] += int(value)


def _candidate_kind_ids(
    legal_actions: list[list[dict[str, Any]]],
    candidate_width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """执行与函数名对应的业务逻辑。
    
    参数:
        legal_actions (list[list[dict[str, Any]]]): 合法动作列表。
        candidate_width (int): 函数执行所需的输入参数。
        device (torch.device): 计算设备。
    
    返回:
        torch.Tensor: 张量结果。
    
    异常:
        无。
    """
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


def _no_yaku_call_candidate_mask(
    *,
    candidate_kind_ids: torch.Tensor,
    candidate_feats: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    call_kind_ids = torch.tensor(
        [
            RESPONSE_KIND_RANK["chi"],
            RESPONSE_KIND_RANK["pon"],
            RESPONSE_KIND_RANK["daiminkan"],
        ],
        dtype=candidate_kind_ids.dtype,
        device=candidate_kind_ids.device,
    )
    call_mask = torch.isin(candidate_kind_ids, call_kind_ids)
    if candidate_feats.shape[2] <= 86:
        return torch.zeros_like(candidate_mask, dtype=torch.bool)
    return candidate_mask & call_mask & (candidate_feats[:, :, 48 + 38] > 0.5)


def _select_operating_points(
    points: list[ResponseCalibrationPoint],
    false_call_rate_targets: list[float],
) -> list[ResponseCalibrationSelection]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        points (list[ResponseCalibrationPoint]): 相关数据列表。
        false_call_rate_targets (list[float]): 误鸣率目标列表。
    
    返回:
        list[ResponseCalibrationSelection]: 列表结果。
    
    异常:
        无。
    """
    selections: list[ResponseCalibrationSelection] = []
    for target in false_call_rate_targets:
        eligible = [point for point in points if point.false_call_rate <= target]
        if not eligible:
            selections.append(
                ResponseCalibrationSelection(
                    false_call_rate_target=float(target),
                    selected_pass_bias=None,
                    selected_no_yaku_call_penalty=None,
                    selected_equivalent_call_margin=None,
                    selected_no_yaku_equivalent_call_margin=None,
                    selected_non_pass_f1=None,
                    selected_non_pass_precision=None,
                    selected_non_pass_recall=None,
                    selected_false_call_rate=None,
                    selected_no_yaku_false_call_rate=None,
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
                selected_no_yaku_call_penalty=selected.no_yaku_call_penalty,
                selected_equivalent_call_margin=selected.equivalent_call_margin,
                selected_no_yaku_equivalent_call_margin=selected.no_yaku_equivalent_call_margin,
                selected_non_pass_f1=selected.non_pass_f1,
                selected_non_pass_precision=selected.non_pass_precision,
                selected_non_pass_recall=selected.non_pass_recall,
                selected_false_call_rate=selected.false_call_rate,
                selected_no_yaku_false_call_rate=selected.no_yaku_false_call_rate,
                selected_missed_call_rate=selected.missed_call_rate,
            )
        )
    return selections


def _merge_calibration_scanners(
    scanners: list[_ResponseCalibrationScanner],
) -> _ResponseCalibrationScanner:
    if not scanners:
        raise ValueError("cannot merge empty calibration scanner list")
    merged = _ResponseCalibrationScanner(
        pass_biases=scanners[0].pass_biases,
        no_yaku_call_penalties=scanners[0].no_yaku_call_penalties,
    )
    for scanner in scanners:
        merged.merge(scanner)
    return merged


def _build_calibration_dataset(
    manifest_path: Path,
    *,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
) -> StreamingManifestResponseDataset:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        manifest_path (Path): 清单文件路径。
    
    返回:
        StreamingManifestResponseDataset: 函数执行结果。
    
    异常:
        无。
    """
    return StreamingManifestResponseDataset(
        manifest_path,
        min_candidate_count=1,
        shuffle_shards=False,
        shuffle_rows=False,
        shuffle_buffer_size=0,
        include_legal_actions=True,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )


def _normalize_pass_biases(pass_biases: list[float]) -> list[float]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        pass_biases (list[float]): 跳过动作偏置列表。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    if not pass_biases:
        raise ValueError("calibration requires at least one pass bias")
    normalized = [float(bias) for bias in pass_biases]
    for bias in normalized:
        if not math.isfinite(bias):
            raise ValueError(f"pass bias must be finite, got: {bias}")
    return normalized


def _normalize_no_yaku_call_penalties(penalties: list[float]) -> list[float]:
    if not penalties:
        raise ValueError("calibration requires at least one no-yaku call penalty")
    normalized = [float(penalty) for penalty in penalties]
    for penalty in normalized:
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError(f"no-yaku call penalty must be finite and non-negative, got: {penalty}")
    return normalized


def _normalize_false_call_rate_targets(targets: list[float]) -> list[float]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        targets (list[float]): 相关数据列表。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        ValueError: 当输入数据不满足函数约束时抛出。
    """
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
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        checkpoint_path (Path): 模型检查点路径。
        split (str): 数据划分名称。
        report_path (str | Path | None): 报告输出路径。
    
    返回:
        Path: 函数执行结果。
    
    异常:
        无。
    """
    if report_path is not None:
        return Path(report_path)
    return checkpoint_path.parent / f"response_calibration_{split}.json"


def _resolve_calibration_manifest_path(
    config: dict[str, Any],
    split: str,
    manifest_path: str | Path | None,
) -> Path:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        config (dict[str, Any]): 配置字典。
        split (str): 数据划分名称。
        manifest_path (str | Path | None): 清单文件路径。
    
    返回:
        Path: 函数执行结果。
    
    异常:
        FileNotFoundError: 当所需文件或目录不存在时抛出。
    """
    if manifest_path is None:
        return _resolve_manifest_path(config, split)
    resolved = Path(manifest_path)
    if not resolved.exists():
        raise FileNotFoundError(f"manifest file not found for split '{split}': {resolved}")
    return resolved


def _should_report_progress(batch_index: int, total_batches: int, every_n_batches: int) -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        batch_index (int): 相关索引值。
        total_batches (int): 相关数据列表。
        every_n_batches (int): 相关数据列表。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if every_n_batches <= 0:
        return False
    return batch_index == 1 or batch_index == total_batches or batch_index % every_n_batches == 0


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """执行与函数名对应的业务逻辑。
    
    参数:
        numerator (int | float): 函数执行所需的输入参数。
        denominator (int | float): 函数执行所需的输入参数。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _f1(precision: float, recall: float) -> float:
    """执行与函数名对应的业务逻辑。
    
    参数:
        precision (float): 函数执行所需的输入参数。
        recall (float): 函数执行所需的输入参数。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
