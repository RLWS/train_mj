from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mjai.data.datasets.parquet_dataset import (
    ResponseBatchCollator,
    StreamingManifestResponseDataset,
    StreamingManifestTurnDataset,
    TurnBatchCollator,
)
from mjai.eval.distributed import (
    EvalDistributedContext,
    gather_eval_objects,
    initialize_eval_distributed_context,
    shutdown_eval_distributed_context,
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
)
from mjai.utils.device import normalize_device_name
from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class EvalKindSummary:
    samples: int
    loss: float
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float


@dataclass(slots=True, frozen=True)
class EvalBucketSummary:
    samples: int
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float


@dataclass(slots=True, frozen=True)
class SampleLocator:
    sample_id: str
    game_id: str
    kyoku_index: int
    step_index: int
    actor: int
    decision_type: str


@dataclass(slots=True, frozen=True)
class CaseLocator:
    bronze_dir: str
    game_id: str
    kyoku_index: int
    step_index: int
    dump_case_command: str


@dataclass(slots=True, frozen=True)
class SampleInspectLocator:
    dataset_dir: str
    sample_id: str
    inspect_sample_command: str


@dataclass(slots=True, frozen=True)
class MisclassifiedExample:
    sample_id: str
    label_kind: str
    candidate_count: int
    label_index: int
    sample_locator: SampleLocator
    case_locator: CaseLocator
    inspect_locator: SampleInspectLocator
    label_action: dict[str, Any]
    top_predictions: list[dict[str, Any]]
    legal_actions: list[dict[str, Any]]


@dataclass(slots=True, frozen=True)
class EvalSummary:
    task: str
    split: str
    loss: float
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    macro_top1_accuracy: float
    macro_top3_accuracy: float
    macro_top5_accuracy: float
    samples: int
    batches: int
    manifest_path: str
    checkpoint_path: str
    report_path: str
    used_device: str
    per_kind: dict[str, EvalKindSummary]
    class_groups: dict[str, EvalKindSummary]
    candidate_buckets: dict[str, EvalBucketSummary]
    response: ResponseDiagnostics | None
    misclassified_examples: list[MisclassifiedExample]


@dataclass(slots=True, frozen=True)
class _EvalPartial:
    total_loss: float
    total_samples: int
    batches: int
    topk_hits: dict[int, int]
    kind_totals: dict[str, int]
    kind_loss_totals: dict[str, float]
    kind_hits: dict[str, dict[int, int]]
    group_totals: dict[str, int]
    group_loss_totals: dict[str, float]
    group_hits: dict[str, dict[int, int]]
    bucket_totals: dict[str, int]
    bucket_hits: dict[str, dict[int, int]]
    response: ResponseDiagnostics | None
    misclassified_examples: list[MisclassifiedExample]


def evaluate_checkpoint(
    ckpt_path: str | Path,
    split: str,
    config_override: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
    task_override: str | None = None,
) -> EvalSummary:
    """加载检查点并执行离线评估。
    
    参数:
        ckpt_path (str | Path): 模型检查点路径。
        split (str): 数据划分名称。
        config_override (dict[str, Any] | None): 配置覆盖字典。
        report_path (str | Path | None): 报告输出路径。
    
    返回:
        EvalSummary: 函数执行结果。
    
    异常:
        TypeError: 当输入类型不符合要求时抛出。
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    checkpoint_path = Path(ckpt_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    base_config = checkpoint.get("config")
    if not isinstance(base_config, dict):
        raise TypeError(f"checkpoint config is missing or invalid: {checkpoint_path}")

    config = _merge_dicts(base_config, config_override or {})
    checkpoint_task_name = str(base_config.get("task", {}).get("task", ""))
    task_name = _resolve_eval_task(
        checkpoint_task_name=checkpoint_task_name,
        config=config,
        task_override=task_override,
    )
    if task_name not in {"turn", "response"}:
        raise ValueError(f"offline eval currently supports only task=turn/response, got: {task_name}")

    manifest_path = _resolve_manifest_path(config, split)
    candidate_limit = _candidate_limit_for_task(config, task_name)
    batch_size = _batch_size_for_task(config, task_name)
    max_misclassified_examples = int(config.get("eval", {}).get("max_misclassified_examples", 10))
    resolved_report_path = _resolve_report_path(checkpoint_path, split, report_path)
    distributed_context = initialize_eval_distributed_context(config)
    try:
        summary = _evaluate_checkpoint_inner(
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
            checkpoint_task_name=checkpoint_task_name,
            task_name=task_name,
            split=split,
            config=config,
            manifest_path=manifest_path,
            candidate_limit=candidate_limit,
            batch_size=batch_size,
            max_misclassified_examples=max_misclassified_examples,
            resolved_report_path=resolved_report_path,
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


def _evaluate_checkpoint_inner(
    *,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_task_name: str,
    task_name: str,
    split: str,
    config: dict[str, Any],
    manifest_path: Path,
    candidate_limit: int,
    batch_size: int,
    max_misclassified_examples: int,
    resolved_report_path: Path,
    distributed_context: EvalDistributedContext,
) -> EvalSummary:
    device = distributed_context.device
    num_workers = _resolve_eval_num_workers(config)
    pin_memory = bool(config.get("eval", {}).get("pin_memory", device.type == "cuda"))
    dataset_dir = _resolve_dataset_dir(manifest_path, task_name)
    bronze_dir = _resolve_bronze_dir(config)

    dataset = _build_dataset(
        task_name,
        manifest_path,
        distributed_rank=distributed_context.rank,
        distributed_world_size=distributed_context.world_size,
    )
    if len(dataset) == 0:
        raise ValueError(f"manifest has 0 samples for split '{split}': {manifest_path}")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_build_collate_fn(task_name, candidate_limit),
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
            state_dim=TURN_STATE_DIM if task_name == "turn" else RESPONSE_STATE_DIM,
            action_dim=TURN_ACTION_DIM if task_name == "turn" else RESPONSE_ACTION_DIM,
            hidden_size=int(config.get("model", {}).get("hidden_size", 128)),
            dropout=float(config.get("model", {}).get("dropout", 0.1)),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    partial = _evaluate_loader(
        model=model,
        data_loader=data_loader,
        task_name=task_name,
        config=config,
        device=device,
        pin_memory=pin_memory,
        max_misclassified_examples=max_misclassified_examples,
        dataset_dir=dataset_dir,
        bronze_dir=bronze_dir,
    )
    partials = gather_eval_objects(partial, distributed_context)
    combined = _combine_eval_partials(
        partials,
        max_misclassified_examples=max_misclassified_examples,
    )
    return _summary_from_partial(
        partial=combined,
        task_name=task_name,
        split=split,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        resolved_report_path=resolved_report_path,
        device=device,
    )


def _evaluate_loader(
    *,
    model,
    data_loader: DataLoader,
    task_name: str,
    config: dict[str, Any],
    device: torch.device,
    pin_memory: bool,
    max_misclassified_examples: int,
    dataset_dir: Path,
    bronze_dir: Path,
) -> _EvalPartial:
    weight_map = class_weight_map(task_name, config.get("train", {}))
    total_loss = 0.0
    total_samples = 0
    batches = 0
    topk_hits = {1: 0, 3: 0, 5: 0}
    kind_totals: dict[str, int] = {}
    kind_loss_totals: dict[str, float] = {}
    kind_hits: dict[str, dict[int, int]] = {}
    group_totals: dict[str, int] = {}
    group_loss_totals: dict[str, float] = {}
    group_hits: dict[str, dict[int, int]] = {}
    bucket_totals: dict[str, int] = {}
    bucket_hits: dict[str, dict[int, int]] = {}
    response_metrics = ResponseMetricsAccumulator() if task_name == "response" else None
    misclassified_examples: list[MisclassifiedExample] = []
    non_blocking = bool(pin_memory and device.type == "cuda")

    with torch.no_grad():
        for batch in data_loader:
            batches += 1
            prepared_batch = {
                key: value.to(device, non_blocking=non_blocking)
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
            per_sample_losses = torch.nn.functional.cross_entropy(
                logits,
                prepared_batch["labels"],
                reduction="none",
            )
            batch_size_actual = int(prepared_batch["labels"].shape[0])
            total_loss += float(loss.item()) * batch_size_actual
            total_samples += batch_size_actual

            ranks = _compute_label_ranks(logits, prepared_batch["labels"])
            predictions = logits.argmax(dim=1)
            for batch_index, label_kind in enumerate(prepared_batch["label_kinds"]):
                rank = int(ranks[batch_index].item())
                candidate_count = int(prepared_batch["candidate_counts"][batch_index])
                bucket_name = _candidate_bucket_name(candidate_count)
                sample_loss = float(per_sample_losses[batch_index].item())
                group_name = class_group_name(task_name, label_kind)

                kind_totals[label_kind] = kind_totals.get(label_kind, 0) + 1
                kind_loss_totals[label_kind] = kind_loss_totals.get(label_kind, 0.0) + sample_loss
                kind_hits.setdefault(label_kind, {1: 0, 3: 0, 5: 0})
                group_totals[group_name] = group_totals.get(group_name, 0) + 1
                group_loss_totals[group_name] = group_loss_totals.get(group_name, 0.0) + sample_loss
                group_hits.setdefault(group_name, {1: 0, 3: 0, 5: 0})
                bucket_totals[bucket_name] = bucket_totals.get(bucket_name, 0) + 1
                bucket_hits.setdefault(bucket_name, {1: 0, 3: 0, 5: 0})

                for k in (1, 3, 5):
                    hit = int(rank <= k)
                    topk_hits[k] += hit
                    kind_hits[label_kind][k] += hit
                    group_hits[group_name][k] += hit
                    bucket_hits[bucket_name][k] += hit

                if response_metrics is not None:
                    legal_actions = prepared_batch["legal_actions"][batch_index]
                    predicted_index = int(predictions[batch_index].item())
                    predicted_action = legal_actions[predicted_index]
                    response_metrics.update(
                        label_kind=label_kind,
                        predicted_kind=str(predicted_action.get("kind", "unknown")),
                        candidate_kinds=[str(action.get("kind", "unknown")) for action in legal_actions],
                        candidate_count=candidate_count,
                        label_rank=rank,
                    )

                if rank > 1 and len(misclassified_examples) < max_misclassified_examples:
                    sample_id = str(prepared_batch["sample_ids"][batch_index])
                    game_id = str(prepared_batch["game_ids"][batch_index])
                    kyoku_index = int(prepared_batch["kyoku_indices"][batch_index])
                    step_index = int(prepared_batch["step_indices"][batch_index])
                    actor = int(prepared_batch["actors"][batch_index])
                    decision_type = str(prepared_batch["decision_types"][batch_index])
                    misclassified_examples.append(
                        MisclassifiedExample(
                            sample_id=sample_id,
                            label_kind=label_kind,
                            candidate_count=candidate_count,
                            label_index=int(prepared_batch["labels"][batch_index].item()),
                            sample_locator=SampleLocator(
                                sample_id=sample_id,
                                game_id=game_id,
                                kyoku_index=kyoku_index,
                                step_index=step_index,
                                actor=actor,
                                decision_type=decision_type,
                            ),
                            case_locator=CaseLocator(
                                bronze_dir=bronze_dir.as_posix(),
                                game_id=game_id,
                                kyoku_index=kyoku_index,
                                step_index=step_index,
                                dump_case_command=(
                                    f"python -m mjai.cli dump-case --bronze-dir {bronze_dir.as_posix()} "
                                    f"--game-id {game_id} --kyoku-id {kyoku_index} --step-id {step_index}"
                                ),
                            ),
                            inspect_locator=SampleInspectLocator(
                                dataset_dir=dataset_dir.as_posix(),
                                sample_id=sample_id,
                                inspect_sample_command=(
                                    f"python -m mjai.cli inspect-sample --dataset {dataset_dir.as_posix()} "
                                    f"--sample-id {sample_id}"
                                ),
                            ),
                            label_action=dict(prepared_batch["label_actions"][batch_index]),
                            top_predictions=_build_top_predictions(
                                logits[batch_index],
                                prepared_batch["legal_actions"][batch_index],
                            ),
                            legal_actions=list(prepared_batch["legal_actions"][batch_index]),
                        )
                    )

    return _EvalPartial(
        total_loss=total_loss,
        total_samples=total_samples,
        batches=batches,
        topk_hits=topk_hits,
        kind_totals=kind_totals,
        kind_loss_totals=kind_loss_totals,
        kind_hits=kind_hits,
        group_totals=group_totals,
        group_loss_totals=group_loss_totals,
        group_hits=group_hits,
        bucket_totals=bucket_totals,
        bucket_hits=bucket_hits,
        response=None if response_metrics is None else response_metrics.summary(),
        misclassified_examples=misclassified_examples,
    )


def _combine_eval_partials(
    partials: list[_EvalPartial],
    *,
    max_misclassified_examples: int,
) -> _EvalPartial:
    total_loss = sum(partial.total_loss for partial in partials)
    total_samples = sum(partial.total_samples for partial in partials)
    batches = sum(partial.batches for partial in partials)
    topk_hits = {k: sum(partial.topk_hits.get(k, 0) for partial in partials) for k in (1, 3, 5)}
    kind_totals = _sum_int_dicts(partial.kind_totals for partial in partials)
    kind_loss_totals = _sum_float_dicts(partial.kind_loss_totals for partial in partials)
    kind_hits = _sum_nested_hit_dicts(partial.kind_hits for partial in partials)
    group_totals = _sum_int_dicts(partial.group_totals for partial in partials)
    group_loss_totals = _sum_float_dicts(partial.group_loss_totals for partial in partials)
    group_hits = _sum_nested_hit_dicts(partial.group_hits for partial in partials)
    bucket_totals = _sum_int_dicts(partial.bucket_totals for partial in partials)
    bucket_hits = _sum_nested_hit_dicts(partial.bucket_hits for partial in partials)
    misclassified_examples: list[MisclassifiedExample] = []
    for partial in partials:
        remaining = max_misclassified_examples - len(misclassified_examples)
        if remaining <= 0:
            break
        misclassified_examples.extend(partial.misclassified_examples[:remaining])

    return _EvalPartial(
        total_loss=total_loss,
        total_samples=total_samples,
        batches=batches,
        topk_hits=topk_hits,
        kind_totals=kind_totals,
        kind_loss_totals=kind_loss_totals,
        kind_hits=kind_hits,
        group_totals=group_totals,
        group_loss_totals=group_loss_totals,
        group_hits=group_hits,
        bucket_totals=bucket_totals,
        bucket_hits=bucket_hits,
        response=combine_response_diagnostics(partial.response for partial in partials),
        misclassified_examples=misclassified_examples,
    )


def _summary_from_partial(
    *,
    partial: _EvalPartial,
    task_name: str,
    split: str,
    manifest_path: Path,
    checkpoint_path: Path,
    resolved_report_path: Path,
    device: torch.device,
) -> EvalSummary:
    if partial.total_samples <= 0:
        raise ValueError(f"eval produced 0 samples for split '{split}': {manifest_path}")
    per_kind = _build_eval_kind_summaries(
        partial.kind_totals,
        partial.kind_hits,
        partial.kind_loss_totals,
    )
    class_groups = _build_eval_kind_summaries(
        partial.group_totals,
        partial.group_hits,
        partial.group_loss_totals,
    )
    return EvalSummary(
        task=task_name,
        split=split,
        loss=partial.total_loss / partial.total_samples,
        top1_accuracy=partial.topk_hits[1] / partial.total_samples,
        top3_accuracy=partial.topk_hits[3] / partial.total_samples,
        top5_accuracy=partial.topk_hits[5] / partial.total_samples,
        macro_top1_accuracy=_macro_topk_accuracy(per_kind, 1),
        macro_top3_accuracy=_macro_topk_accuracy(per_kind, 3),
        macro_top5_accuracy=_macro_topk_accuracy(per_kind, 5),
        samples=partial.total_samples,
        batches=partial.batches,
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        report_path=str(resolved_report_path),
        used_device=str(device),
        per_kind=per_kind,
        class_groups=class_groups,
        candidate_buckets={
            bucket: EvalBucketSummary(
                samples=count,
                top1_accuracy=partial.bucket_hits[bucket][1] / count,
                top3_accuracy=partial.bucket_hits[bucket][3] / count,
                top5_accuracy=partial.bucket_hits[bucket][5] / count,
            )
            for bucket, count in sorted(partial.bucket_totals.items())
        },
        response=partial.response,
        misclassified_examples=partial.misclassified_examples,
    )


def _sum_int_dicts(dicts) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in dicts:
        for key, value in item.items():
            result[key] = result.get(key, 0) + int(value)
    return result


def _sum_float_dicts(dicts) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in dicts:
        for key, value in item.items():
            result[key] = result.get(key, 0.0) + float(value)
    return result


def _sum_nested_hit_dicts(dicts) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for item in dicts:
        for key, hits in item.items():
            target = result.setdefault(key, {1: 0, 3: 0, 5: 0})
            for top_k in (1, 3, 5):
                target[top_k] += int(hits.get(top_k, 0))
    return result


def _resolve_manifest_path(config: dict[str, Any], split: str) -> Path:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        config (dict[str, Any]): 配置字典。
        split (str): 数据划分名称。
    
    返回:
        Path: 函数执行结果。
    
    异常:
        FileNotFoundError: 当所需文件或目录不存在时抛出。
    """
    data_config = config.get("data", {})
    if split == "train":
        manifest = data_config.get("train_manifest")
    elif split == "valid":
        manifest = data_config.get("valid_manifest")
    else:
        manifest = data_config.get("test_manifest")
        if manifest is None:
            reference = data_config.get("valid_manifest") or data_config.get("train_manifest")
            if reference is not None:
                manifest = str(Path(reference).with_name("test.jsonl"))

    if manifest is None:
        raise FileNotFoundError(f"manifest path for split '{split}' is not configured")

    manifest_path = Path(manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest file not found for split '{split}': {manifest_path}")
    return manifest_path


def _resolve_eval_task(
    *,
    checkpoint_task_name: str,
    config: dict[str, Any],
    task_override: str | None,
) -> str:
    if task_override is not None:
        selected = str(task_override)
        if selected not in {"turn", "response"}:
            raise ValueError(f"eval --task must be turn or response, got: {selected}")
        if checkpoint_task_name not in {"multitask", selected}:
            raise ValueError(
                f"eval --task={selected} does not match checkpoint task={checkpoint_task_name}"
            )
        return selected

    configured_task = str(config.get("task", {}).get("task", ""))
    eval_task = config.get("eval", {}).get("task")
    if eval_task is not None:
        selected = str(eval_task)
        if selected not in {"turn", "response"}:
            raise ValueError(f"eval.task must be turn or response, got: {selected}")
        return selected

    if checkpoint_task_name == "multitask":
        raise ValueError("multitask checkpoints require eval --task turn or --task response")
    return configured_task


def _candidate_limit_for_task(config: dict[str, Any], task_name: str) -> int:
    data_config = config.get("data", {})
    task_key = f"{task_name}_candidate_limit"
    if task_key in data_config:
        return int(data_config[task_key])
    if "candidate_limit" in data_config:
        return int(data_config["candidate_limit"])
    return 16 if task_name == "response" else 64


def _batch_size_for_task(config: dict[str, Any], task_name: str) -> int:
    eval_config = config.get("eval", {})
    train_config = config.get("train", {})
    task_key = f"{task_name}_batch_size"
    if task_key in eval_config:
        return int(eval_config[task_key])
    if "batch_size" in eval_config:
        return int(eval_config["batch_size"])
    return int(train_config.get(task_key, train_config.get("batch_size", 32)))


def _resolve_report_path(
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
    return checkpoint_path.parent / f"eval_{split}.json"


def _resolve_dataset_dir(manifest_path: Path, task_name: str) -> Path:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        manifest_path (Path): 清单文件路径。
        task_name (str): 任务名称。
    
    返回:
        Path: 函数执行结果。
    
    异常:
        无。
    """
    first_record = _read_first_manifest_record(manifest_path)
    sample_path_key = "turn_path" if task_name == "turn" else "response_path"
    sample_path = Path(str(first_record[sample_path_key]))
    return sample_path.parent.parent


def _resolve_bronze_dir(config: dict[str, Any]) -> Path:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        config (dict[str, Any]): 配置字典。
    
    返回:
        Path: 函数执行结果。
    
    异常:
        无。
    """
    bronze_dir = config.get("data", {}).get("input_root", "data/bronze_events")
    return Path(str(bronze_dir))


def _read_first_manifest_record(manifest_path: Path) -> dict[str, Any]:
    """读取与函数名对应的数据。
    
    参数:
        manifest_path (Path): 清单文件路径。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        for line in manifest_file:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"manifest has no records: {manifest_path}")


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


def _resolve_device(config_device: str | None) -> torch.device:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        config_device (str | None): 函数执行所需的输入参数。
    
    返回:
        torch.device: 函数执行结果。
    
    异常:
        无。
    """
    normalized = normalize_device_name(config_device)
    if normalized in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _candidate_bucket_name(candidate_count: int) -> str:
    """执行与函数名对应的业务逻辑。
    
    参数:
        candidate_count (int): 相关数量。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    if candidate_count <= 4:
        return "1-4"
    if candidate_count <= 8:
        return "5-8"
    if candidate_count <= 16:
        return "9-16"
    return "17+"


def _build_top_predictions(
    logits: torch.Tensor,
    legal_actions: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        logits (torch.Tensor): 候选动作分数张量。
        legal_actions (list[dict[str, Any]]): 合法动作列表。
        top_k (int): 前若干候选数量。
    
    返回:
        list[dict[str, Any]]: 列表结果。
    
    异常:
        无。
    """
    k = min(top_k, len(legal_actions))
    scores, indices = torch.topk(logits, k=k)
    predictions: list[dict[str, Any]] = []
    for score, index in zip(scores.tolist(), indices.tolist()):
        predictions.append(
            {
                "index": int(index),
                "score": float(score),
                "action": dict(legal_actions[int(index)]),
            }
        )
    return predictions


def _build_dataset(
    task_name: str,
    manifest_path: Path,
    *,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
):
    # 大数据评估默认流式读取 parquet shard，不再把整个 valid/test split 载入内存。
    # 错例报告仍需要动作 JSON，因此只额外读取 legal_actions/label_action，不读取 state_json。
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        task_name (str): 任务名称。
        manifest_path (Path): 清单文件路径。
    
    返回:
        Any: 函数执行结果。
    
    异常:
        无。
    """
    if task_name == "turn":
        return StreamingManifestTurnDataset(
            manifest_path,
            shuffle_shards=False,
            shuffle_rows=False,
            shuffle_buffer_size=0,
            include_legal_actions=True,
            include_label_action=True,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
        )
    return StreamingManifestResponseDataset(
        manifest_path,
        min_candidate_count=1,
        shuffle_shards=False,
        shuffle_rows=False,
        shuffle_buffer_size=0,
        include_legal_actions=True,
        include_label_action=True,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
    )


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


def _resolve_eval_num_workers(config: dict[str, Any]) -> int:
    """解析输入配置或路径并返回规范化结果。
    
    参数:
        config (dict[str, Any]): 配置字典。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    eval_config = config.get("eval", {})
    train_config = config.get("train", {})
    return int(eval_config.get("num_workers", train_config.get("valid_num_workers", 0)))


def _worker_loader_kwargs(config: dict[str, Any], num_workers: int) -> dict[str, Any]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        config (dict[str, Any]): 配置字典。
        num_workers (int): 相关数据列表。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        无。
    """
    if num_workers <= 0:
        return {}
    eval_config = config.get("eval", {})
    train_config = config.get("train", {})
    kwargs: dict[str, Any] = {}
    prefetch_factor = eval_config.get("prefetch_factor", train_config.get("prefetch_factor"))
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    if bool(eval_config.get("persistent_workers", train_config.get("persistent_workers", False))):
        kwargs["persistent_workers"] = True
    return kwargs


def _build_eval_kind_summaries(
    totals: dict[str, int],
    hits: dict[str, dict[int, int]],
    loss_totals: dict[str, float],
) -> dict[str, EvalKindSummary]:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        totals (dict[str, int]): 相关数据列表。
        hits (dict[str, dict[int, int]]): 相关数据列表。
        loss_totals (dict[str, float]): 相关数据列表。
    
    返回:
        dict[str, EvalKindSummary]: 字典结果。
    
    异常:
        无。
    """
    return {
        kind: EvalKindSummary(
            samples=count,
            loss=0.0 if count == 0 else loss_totals.get(kind, 0.0) / count,
            top1_accuracy=hits[kind][1] / count,
            top3_accuracy=hits[kind][3] / count,
            top5_accuracy=hits[kind][5] / count,
        )
        for kind, count in sorted(totals.items())
    }


def _macro_topk_accuracy(per_kind: dict[str, EvalKindSummary], top_k: int) -> float:
    """执行与函数名对应的业务逻辑。
    
    参数:
        per_kind (dict[str, EvalKindSummary]): 函数执行所需的输入参数。
        top_k (int): 前若干候选数量。
    
    返回:
        float: 浮点数结果。
    
    异常:
        无。
    """
    if not per_kind:
        return 0.0
    metric_name = f"top{top_k}_accuracy"
    return sum(float(getattr(summary, metric_name)) for summary in per_kind.values()) / len(per_kind)


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """执行与函数名对应的业务逻辑。
    
    参数:
        base (dict[str, Any]): 函数执行所需的输入参数。
        update (dict[str, Any]): 函数执行所需的输入参数。
    
    返回:
        dict[str, Any]: 字典结果。
    
    异常:
        无。
    """
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
