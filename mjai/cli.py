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
from mjai.training.engine import train_turn_model
from mjai.utils.config import load_resolved_yaml, load_yaml


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config file.")


def _load_optional_config(path: str | None) -> dict:
    if path is None:
        return {}
    return load_yaml(path)


def _not_implemented(command: str, config_path: str | None = None) -> int:
    config = _load_optional_config(config_path)
    print(f"mjai command '{command}' is registered.")
    if config_path:
        print(f"loaded config: {config_path} ({len(config)} top-level keys)")
    print("implementation pending in the next development tasks")
    return 0


def _run_preprocess(args: argparse.Namespace) -> int:
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
    _load_optional_config(getattr(args, "config", None))
    builder = DatasetBuilder()
    summary = builder.build_from_event_dir(args.input, args.output)
    print(
        f"built dataset: games={summary.games} turn_samples={summary.turn_samples} "
        f"response_samples={summary.response_samples}"
    )
    print(f"turn labels: {summary.turn_label_counts}")
    print(f"response labels: {summary.response_label_counts}")
    _print_warnings(summary.warnings)
    return 0


def _run_make_manifest(args: argparse.Namespace) -> int:
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
    config = load_resolved_yaml(args.config)
    summary = train_turn_model(config)
    print(
        f"trained task={summary.task} epochs={summary.epochs} "
        f"train_loss={summary.train_loss:.4f} train_accuracy={summary.train_accuracy:.4f}"
    )
    if summary.valid_accuracy is not None:
        print(
            f"valid_loss={summary.valid_loss:.4f} "
            f"valid_accuracy={summary.valid_accuracy:.4f}"
        )
    print(f"device={summary.used_device}")
    print(f"best_checkpoint={summary.best_checkpoint}")
    print(f"final_checkpoint={summary.final_checkpoint}")
    return 0


def _run_eval(args: argparse.Namespace) -> int:
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
        f"top3={summary.top3_accuracy:.4f} top5={summary.top5_accuracy:.4f}"
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
    return 0


def _run_inspect_sample(args: argparse.Namespace) -> int:
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
    for warning in warnings:
        print(f"warning: {warning}")


def build_parser() -> argparse.ArgumentParser:
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
    if args.command == "inspect-sample":
        return _run_inspect_sample(args)
    if args.command == "dump-case":
        return _run_dump_case(args)
    return _not_implemented(args.command, getattr(args, "config", None))


if __name__ == "__main__":
    raise SystemExit(main())
