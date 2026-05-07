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
from mjai.features.action_features import RESPONSE_ACTION_DIM, TURN_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM
from mjai.models.policy_net import MahjongPolicyNet
from mjai.training.class_balance import build_sample_weights, class_group_name, class_weight_map
from mjai.training.losses import policy_loss
from mjai.training.metrics import ResponseDiagnostics, ResponseMetricsAccumulator
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


def evaluate_checkpoint(
    ckpt_path: str | Path,
    split: str,
    config_override: dict[str, Any] | None = None,
    report_path: str | Path | None = None,
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
    task_name = str(config.get("task", {}).get("task", ""))
    if task_name not in {"turn", "response"}:
        raise ValueError(f"offline eval currently supports only task=turn/response, got: {task_name}")

    manifest_path = _resolve_manifest_path(config, split)
    candidate_limit = int(config.get("data", {}).get("candidate_limit", 64))
    batch_size = int(config.get("train", {}).get("batch_size", 32))
    max_misclassified_examples = int(config.get("eval", {}).get("max_misclassified_examples", 10))
    device = _resolve_device(config.get("train", {}).get("device"))
    num_workers = _resolve_eval_num_workers(config)
    pin_memory = bool(config.get("eval", {}).get("pin_memory", device.type == "cuda"))
    dataset_dir = _resolve_dataset_dir(manifest_path, task_name)
    bronze_dir = _resolve_bronze_dir(config)

    dataset = _build_dataset(task_name, manifest_path)
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

    model = MahjongPolicyNet(
        state_dim=TURN_STATE_DIM if task_name == "turn" else RESPONSE_STATE_DIM,
        action_dim=TURN_ACTION_DIM if task_name == "turn" else RESPONSE_ACTION_DIM,
        hidden_size=int(config.get("model", {}).get("hidden_size", 128)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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

    average_loss = total_loss / total_samples
    resolved_report_path = _resolve_report_path(checkpoint_path, split, report_path)
    ensure_dir(resolved_report_path.parent)
    per_kind = _build_eval_kind_summaries(kind_totals, kind_hits, kind_loss_totals)
    class_groups = _build_eval_kind_summaries(group_totals, group_hits, group_loss_totals)

    summary = EvalSummary(
        task=task_name,
        split=split,
        loss=average_loss,
        top1_accuracy=topk_hits[1] / total_samples,
        top3_accuracy=topk_hits[3] / total_samples,
        top5_accuracy=topk_hits[5] / total_samples,
        macro_top1_accuracy=_macro_topk_accuracy(per_kind, 1),
        macro_top3_accuracy=_macro_topk_accuracy(per_kind, 3),
        macro_top5_accuracy=_macro_topk_accuracy(per_kind, 5),
        samples=total_samples,
        batches=batches,
        manifest_path=str(manifest_path),
        checkpoint_path=str(checkpoint_path),
        report_path=str(resolved_report_path),
        used_device=str(device),
        per_kind=per_kind,
        class_groups=class_groups,
        candidate_buckets={
            bucket: EvalBucketSummary(
                samples=count,
                top1_accuracy=bucket_hits[bucket][1] / count,
                top3_accuracy=bucket_hits[bucket][3] / count,
                top5_accuracy=bucket_hits[bucket][5] / count,
            )
            for bucket, count in sorted(bucket_totals.items())
        },
        response=None if response_metrics is None else response_metrics.summary(),
        misclassified_examples=misclassified_examples,
    )
    resolved_report_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


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


def _build_dataset(task_name: str, manifest_path: Path):
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
        )
    return StreamingManifestResponseDataset(
        manifest_path,
        min_candidate_count=1,
        shuffle_shards=False,
        shuffle_rows=False,
        shuffle_buffer_size=0,
        include_legal_actions=True,
        include_label_action=True,
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
