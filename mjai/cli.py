from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from mjai.data.builders.event_builder import EventBuilder
from mjai.data.builders.event_validator import EventValidator
from mjai.data.builders.dataset_builder import DatasetBuilder
from mjai.data.builders.manifest_builder import ManifestBuilder
from mjai.data.datasets.parquet_dataset import inspect_dataset_sample, inspect_dataset_sample_by_id
from mjai.eval.case_dump import dump_case_to_json
from mjai.eval.offline import evaluate_checkpoint
from mjai.eval.response_calibration import calibrate_response_checkpoint
from mjai.training.engine import train_turn_model
from mjai.utils.config import load_resolved_yaml, load_yaml


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    """向当前状态添加输入数据。
    
    参数:
        parser (argparse.ArgumentParser): 命令行解析器对象。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")


def _load_optional_config(path: str | None) -> dict:
    """加载与函数名对应的数据。
    
    参数:
        path (str | None): 文件或目录路径。
    
    返回:
        dict: 字典结果。
    
    异常:
        无。
    """
    if path is None:
        return {}
    return load_yaml(path)


def _not_implemented(command: str, config_path: str | None = None) -> int:
    """执行与函数名对应的业务逻辑。
    
    参数:
        command (str): 函数执行所需的输入参数。
        config_path (str | None): 相关文件路径。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    config = _load_optional_config(config_path)
    print(f"mjai command '{command}' is registered.")
    if config_path:
        print(f"loaded config: {config_path} ({len(config)} top-level keys)")
    print("implementation pending in the next development tasks")
    return 0


def _run_preprocess(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    _load_optional_config(getattr(args, "config", None))
    builder = EventBuilder()
    summary = builder.build_from_xml_dir(args.input, args.output)
    print(
        f"preprocessed events: games={summary.games} events={summary.events} "
        f"event_types={summary.event_type_counts}"
    )
    _print_warnings(summary.warnings)
    return 0


def _run_validate_events(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    _load_optional_config(getattr(args, "config", None))
    validator = EventValidator()
    summary = validator.validate_dir(args.input, args.report)
    print(
        f"validated events: files={summary.files_seen} valid={summary.files_valid} "
        f"failed={summary.files_failed} total_events={summary.total_events} "
        f"total_kyokus={summary.total_kyokus} total_discards={summary.total_discards} "
        f"total_melds={summary.total_melds}"
    )
    print(f"event types: {summary.event_type_counts}")
    if summary.error_counts:
        print(f"errors: {summary.error_counts}")
    _print_warnings(summary.warnings)
    return 0


def _run_build_dataset(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    _load_optional_config(getattr(args, "config", None))
    builder = DatasetBuilder()
    summary = builder.build_from_event_dir(
        args.input,
        args.output,
        workers=max(1, int(args.workers)),
        validate=not bool(args.skip_validation),
    )
    print(
        f"built dataset: games={summary.games} turn_samples={summary.turn_samples} "
        f"response_samples={summary.response_samples} elapsed_s={summary.elapsed_seconds:.2f} "
        f"games_per_s={summary.games_per_second:.2f} "
        f"samples_per_s={summary.samples_per_second:.1f}"
    )
    print(f"turn labels: {summary.turn_label_counts}")
    print(f"response labels: {summary.response_label_counts}")
    _print_warnings(summary.warnings)
    return 0


def _run_make_manifest(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    builder = ManifestBuilder()
    summary = builder.build(args.input, args.output)
    print(
        f"built manifests: train_games={summary.train_games} "
        f"valid_games={summary.valid_games} test_games={summary.test_games}"
    )
    print(
        "manifest samples: "
        f"train_turn={summary.train_turn_samples} valid_turn={summary.valid_turn_samples} "
        f"test_turn={summary.test_turn_samples} train_response={summary.train_response_samples} "
        f"valid_response={summary.valid_response_samples} test_response={summary.test_response_samples}"
    )
    _print_warnings(summary.warnings)
    return 0


def _run_train(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    config = load_resolved_yaml(args.config)
    summary = train_turn_model(config)
    print(
        f"trained task={summary.task} epochs={summary.epochs} "
        f"train_loss={summary.train_loss:.4f} train_accuracy={summary.train_accuracy:.4f} "
        f"train_macro_accuracy={summary.train_macro_accuracy:.4f}"
    )
    if summary.valid_accuracy is not None:
        print(
            f"valid_loss={summary.valid_loss:.4f} "
            f"valid_accuracy={summary.valid_accuracy:.4f} "
            f"valid_macro_accuracy={summary.valid_macro_accuracy:.4f}"
        )
    print(f"device={summary.used_device}")
    if summary.filtered_train_samples > 0:
        print(
            "training_filter="
            f"original_train_samples={summary.original_train_samples} "
            f"filtered_train_samples={summary.filtered_train_samples} "
            f"kept_train_samples={summary.train_samples}"
        )
    print(f"best_checkpoint={summary.best_checkpoint}")
    if summary.best_response_checkpoint is not None:
        print(f"best_response_checkpoint={summary.best_response_checkpoint}")
    print(f"best_loss_checkpoint={summary.best_loss_checkpoint}")
    print(f"best_macro_checkpoint={summary.best_macro_checkpoint}")
    print(f"final_checkpoint={summary.final_checkpoint}")
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    config_override = _load_optional_config(getattr(args, "config", None))
    summary = evaluate_checkpoint(
        ckpt_path=args.ckpt,
        split=args.split,
        config_override=config_override,
        report_path=getattr(args, "report", None),
    )
    print(
        f"evaluated task={summary.task} split={summary.split} "
        f"loss={summary.loss:.4f} top1={summary.top1_accuracy:.4f} "
        f"top3={summary.top3_accuracy:.4f} top5={summary.top5_accuracy:.4f} "
        f"macro_top1={summary.macro_top1_accuracy:.4f}"
    )
    print(f"samples={summary.samples} batches={summary.batches} device={summary.used_device}")
    print(f"manifest={summary.manifest_path}")
    print(f"report={summary.report_path}")
    if summary.per_kind:
        per_kind = ", ".join(
            f"{kind}:{kind_summary.samples}/{kind_summary.top1_accuracy:.4f}"
            for kind, kind_summary in sorted(summary.per_kind.items())
        )
        print(f"per_kind_top1={per_kind}")
    if summary.class_groups:
        class_groups = ", ".join(
            f"{group}:{group_summary.samples}/{group_summary.top1_accuracy:.4f}"
            for group, group_summary in sorted(summary.class_groups.items())
        )
        print(f"class_group_top1={class_groups}")
    if summary.response is not None:
        response = summary.response
        print(
            "response_multi_candidate="
            f"samples={response.multi_candidate.samples} "
            f"top1={response.multi_candidate.top1_accuracy:.4f} "
            f"macro_top1={response.multi_candidate.macro_top1_accuracy:.4f}"
        )
        print(
            "response_binary="
            f"non_pass_precision={response.binary.non_pass_precision:.4f} "
            f"non_pass_recall={response.binary.non_pass_recall:.4f} "
            f"non_pass_f1={response.binary.non_pass_f1:.4f} "
            f"false_call_rate={response.binary.false_call_rate:.4f} "
            f"missed_call_rate={response.binary.missed_call_rate:.4f}"
        )
    return 0


def _run_calibrate_response(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    config_override = _load_optional_config(getattr(args, "config", None))
    pass_biases = _parse_float_list(args.pass_biases, option_name="--pass-biases")
    false_call_targets = _parse_rate_list(
        args.false_call_targets,
        option_name="--false-call-targets",
    )
    summary = calibrate_response_checkpoint(
        ckpt_path=args.ckpt,
        split=args.split,
        pass_biases=pass_biases,
        false_call_rate_targets=false_call_targets,
        config_override=config_override,
        report_path=getattr(args, "report", None),
        manifest_path=getattr(args, "manifest", None),
        progress_callback=_print_calibration_progress,
        progress_every_n_batches=max(0, int(args.log_every_n_batches)),
    )
    print(
        f"calibrated task={summary.task} split={summary.split} "
        f"points={len(summary.points)} samples={summary.points[0].samples if summary.points else 0} "
        f"device={summary.used_device}"
    )
    print(f"manifest={summary.manifest_path}")
    print(f"report={summary.report_path}")
    for selection in summary.selections:
        if selection.selected_pass_bias is None:
            print(
                "calibration_selection="
                f"target_false_call_rate={selection.false_call_rate_target:.4f} "
                "selected_pass_bias=None"
            )
            continue
        print(
            "calibration_selection="
            f"target_false_call_rate={selection.false_call_rate_target:.4f} "
            f"selected_pass_bias={selection.selected_pass_bias:.4f} "
            f"false_call_rate={selection.selected_false_call_rate:.4f} "
            f"missed_call_rate={selection.selected_missed_call_rate:.4f} "
            f"non_pass_precision={selection.selected_non_pass_precision:.4f} "
            f"non_pass_recall={selection.selected_non_pass_recall:.4f} "
            f"non_pass_f1={selection.selected_non_pass_f1:.4f}"
        )
    return 0


def _run_inspect_sample(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    _load_optional_config(getattr(args, "config", None))
    if args.sample_id is None and args.index is None:
        raise ValueError("inspect-sample requires either --index or --sample-id")
    if args.sample_id is not None:
        sample = inspect_dataset_sample_by_id(args.dataset, args.sample_id)
    else:
        sample = inspect_dataset_sample(args.dataset, args.index)
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    return 0


def _run_dump_case(args: argparse.Namespace) -> int:
    """运行与函数名对应的处理流程。
    
    参数:
        args (argparse.Namespace): 命令行参数命名空间。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    _load_optional_config(getattr(args, "config", None))
    print(
        dump_case_to_json(
            bronze_dir=args.bronze_dir,
            game_id=args.game_id,
            kyoku_index=int(args.kyoku_id),
            step_index=int(args.step_id),
            context_radius=int(args.context_radius),
        )
    )
    return 0


def _print_warnings(warnings: tuple[str, ...]) -> None:
    """打印与函数名对应的进度或诊断信息。
    
    参数:
        warnings (tuple[str, ...]): 相关数据列表。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    for warning in warnings:
        print(f"warning: {warning}")


def _print_calibration_progress(batch_index: int, total_batches: int, samples_processed: int) -> None:
    """打印与函数名对应的进度或诊断信息。
    
    参数:
        batch_index (int): 相关索引值。
        total_batches (int): 相关数据列表。
        samples_processed (int): 函数执行所需的输入参数。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    print(
        "calibration_progress="
        f"batch={batch_index}/{total_batches} "
        f"samples={samples_processed}",
        flush=True,
    )


def _parse_float_list(raw_value: str, *, option_name: str) -> list[float]:
    """解析输入数据并返回结构化结果。
    
    参数:
        raw_value (str): 函数执行所需的输入参数。
        option_name (str): 函数执行所需的输入参数。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    values: list[float] = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError as exc:
            raise ValueError(f"{option_name} contains a non-float value: {item}") from exc
    if not values:
        raise ValueError(f"{option_name} requires at least one value")
    return values


def _parse_rate_list(raw_value: str, *, option_name: str) -> list[float]:
    """解析输入数据并返回结构化结果。
    
    参数:
        raw_value (str): 函数执行所需的输入参数。
        option_name (str): 函数执行所需的输入参数。
    
    返回:
        list[float]: 列表结果。
    
    异常:
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    values: list[float] = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            if item.endswith("%"):
                value = float(item[:-1].strip()) / 100.0
            else:
                value = float(item)
        except ValueError as exc:
            raise ValueError(f"{option_name} contains a non-rate value: {item}") from exc
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{option_name} values must be in [0, 1] after percent parsing: {item}")
        values.append(value)
    if not values:
        raise ValueError(f"{option_name} requires at least one value")
    return values


def build_parser() -> argparse.ArgumentParser:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        无。
    
    返回:
        argparse.ArgumentParser: 函数执行结果。
    
    异常:
        无。
    """
    parser = argparse.ArgumentParser(prog="mjai", description="Mahjong AI training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Convert Tenhou XML into event shards.")
    preprocess.add_argument("--input", required=True, help="Input raw XML directory.")
    preprocess.add_argument("--output", required=True, help="Output event directory.")
    _add_config_arg(preprocess)

    validate_events = subparsers.add_parser(
        "validate-events", help="Replay and validate event shards."
    )
    validate_events.add_argument("--input", required=True, help="Input event directory.")
    validate_events.add_argument("--report", required=True, help="Output validation report path.")
    _add_config_arg(validate_events)

    build_dataset = subparsers.add_parser(
        "build-dataset", help="Build supervised samples from validated events."
    )
    build_dataset.add_argument("--input", required=True, help="Input event directory.")
    build_dataset.add_argument("--output", required=True, help="Output sample directory.")
    build_dataset.add_argument("--workers", type=int, default=1, help="Parallel shard workers for dataset building.")
    build_dataset.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip replay validation during sample building after validate-events has already passed.",
    )
    _add_config_arg(build_dataset)

    make_manifest = subparsers.add_parser(
        "make-manifest", help="Create train/valid/test manifests."
    )
    make_manifest.add_argument("--input", required=True, help="Input sample directory.")
    make_manifest.add_argument("--output", required=True, help="Output manifest directory.")
    _add_config_arg(make_manifest)

    train = subparsers.add_parser("train", help="Train a policy or value task.")
    train.add_argument("--config", required=True, help="Path to the resolved training config.")

    eval_parser = subparsers.add_parser("eval", help="Run offline evaluation.")
    eval_parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    eval_parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    eval_parser.add_argument("--report", default=None, help="Optional output report path.")
    _add_config_arg(eval_parser)

    calibrate_response = subparsers.add_parser(
        "calibrate-response",
        help="Scan response pass/call score margins for false-call-rate targets.",
    )
    calibrate_response.add_argument("--ckpt", required=True, help="Response checkpoint path.")
    calibrate_response.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    calibrate_response.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest override for the selected split.",
    )
    calibrate_response.add_argument("--report", default=None, help="Optional output report path.")
    calibrate_response.add_argument(
        "--pass-biases",
        default="0,0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.30,0.40,0.60,0.80,1.00",
        help="Comma-separated pass bias / call margin values in logit units.",
    )
    calibrate_response.add_argument(
        "--false-call-targets",
        default="0.010,0.008,0.005",
        help="Comma-separated false-call-rate targets; percent values like 1%% are accepted.",
    )
    calibrate_response.add_argument(
        "--log-every-n-batches",
        type=int,
        default=100,
        help="Print calibration progress every N batches; use 0 to disable.",
    )
    _add_config_arg(calibrate_response)

    replay_eval = subparsers.add_parser("replay-eval", help="Run fixed-game replay evaluation.")
    replay_eval.add_argument("--ckpt", required=True, help="Checkpoint path.")
    replay_eval.add_argument("--games", required=True, help="Game manifest or case list.")
    _add_config_arg(replay_eval)

    inspect_sample = subparsers.add_parser("inspect-sample", help="Inspect one dataset sample.")
    inspect_sample.add_argument("--dataset", required=True, help="Sample dataset path.")
    inspect_sample.add_argument("--index", type=int, default=None, help="Sample index.")
    inspect_sample.add_argument("--sample-id", default=None, help="Sample id.")
    _add_config_arg(inspect_sample)

    infer_on_state = subparsers.add_parser("infer-on-state", help="Run inference on one state.")
    infer_on_state.add_argument("--ckpt", required=True, help="Checkpoint path.")
    infer_on_state.add_argument("--state-json", required=True, help="State JSON path.")
    _add_config_arg(infer_on_state)

    dump_case = subparsers.add_parser("dump-case", help="Dump a parsed decision case.")
    dump_case.add_argument("--game-id", required=True)
    dump_case.add_argument("--kyoku-id", required=True)
    dump_case.add_argument("--step-id", required=True)
    dump_case.add_argument(
        "--bronze-dir",
        default="data/bronze_events",
        help="Directory containing bronze event parquet shards.",
    )
    dump_case.add_argument(
        "--context-radius",
        type=int,
        default=5,
        help="Number of same-kyoku events to include before and after the target event.",
    )
    _add_config_arg(dump_case)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行命令行入口并返回进程退出码。
    
    参数:
        argv (Sequence[str] | None): 命令行参数序列。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "preprocess":
        return _run_preprocess(args)
    if args.command == "validate-events":
        return _run_validate_events(args)
    if args.command == "build-dataset":
        return _run_build_dataset(args)
    if args.command == "make-manifest":
        return _run_make_manifest(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "eval":
        return _run_eval(args)
    if args.command == "calibrate-response":
        return _run_calibrate_response(args)
    if args.command == "inspect-sample":
        return _run_inspect_sample(args)
    if args.command == "dump-case":
        return _run_dump_case(args)
    return _not_implemented(args.command, getattr(args, "config", None))


if __name__ == "__main__":
    raise SystemExit(main())
