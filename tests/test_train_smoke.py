from __future__ import annotations

import json
from pathlib import Path

from mjai.data.datasets.parquet_dataset import (
    StreamingManifestResponseDataset,
    StreamingManifestTurnDataset,
)
from mjai.cli import main
from mjai.eval.offline import _build_dataset as _build_eval_dataset
from mjai.eval.response_calibration import _build_calibration_dataset


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_train_turn_smoke_from_generated_dataset(tmp_path: Path):
    """验证基于生成数据集的自摸巡训练冒烟流程。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"
    manifest_dir = tmp_path / "manifests"

    assert main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]) == 0
    assert main(["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)]) == 0
    assert main(["make-manifest", "--input", str(silver_dir), "--output", str(manifest_dir)]) == 0

    train_manifest = manifest_dir / "train_small.jsonl"
    valid_manifest = manifest_dir / "valid_small.jsonl"
    train_line = (manifest_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    valid_line = (manifest_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines()[0]
    train_manifest.write_text(train_line + "\n", encoding="utf-8")
    valid_manifest.write_text(valid_line + "\n", encoding="utf-8")

    output_dir = tmp_path / "outputs" / "checkpoints" / "turn_smoke_test"
    log_dir = tmp_path / "outputs" / "logs" / "turn_smoke_test"
    config_path = tmp_path / "turn_smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 20260423",
                "exp_name: turn_smoke_test",
                f"output_dir: {output_dir.as_posix()}",
                f"log_dir: {log_dir.as_posix()}",
                "data:",
                f"  train_manifest: {train_manifest.as_posix()}",
                f"  valid_manifest: {valid_manifest.as_posix()}",
                "  candidate_limit: 64",
                "task:",
                "  task: turn",
                "model:",
                "  hidden_size: 32",
                "  dropout: 0.0",
                "train:",
                "  device: cpu",
                "  batch_size: 64",
                "  num_workers: 0",
                "  epochs: 1",
                "  lr: 0.005",
                "  weight_decay: 0.0",
                "  turn_class_weights:",
                "    discard: 1.0",
                "    riichi_discard: 2.0",
                "    tsumo: 3.0",
                "    ankan: 4.0",
                "    kakan: 4.0",
                "    kyuushu_kyuuhai: 4.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config_path)]) == 0

    metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert (output_dir / "config.resolved.yaml").exists()
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "best_accuracy.pt").exists()
    assert (output_dir / "best_loss.pt").exists()
    assert (output_dir / "best_macro.pt").exists()
    assert "best_response_selection_score" in metrics
    assert (output_dir / "final.pt").exists()
    assert (log_dir / "epoch_metrics.jsonl").exists()
    assert metrics["train_samples"] > 0
    assert metrics["valid_samples"] > 0
    assert metrics["filtered_train_samples"] == 0
    assert metrics["epoch_summaries"]
    first_epoch = metrics["epoch_summaries"][0]
    assert "macro_accuracy" in first_epoch
    assert first_epoch["per_kind"]
    assert first_epoch["class_groups"]


def test_train_prints_batch_progress_from_generated_dataset(tmp_path: Path, capsys):
    """验证基于生成数据集的训练过程会打印批次进度。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
        capsys (Any): pytest 提供的标准输出捕获夹具。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"
    manifest_dir = tmp_path / "manifests"

    assert main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]) == 0
    assert main(["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)]) == 0
    assert main(["make-manifest", "--input", str(silver_dir), "--output", str(manifest_dir)]) == 0

    train_manifest = manifest_dir / "train_small.jsonl"
    valid_manifest = manifest_dir / "valid_small.jsonl"
    train_line = (manifest_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    valid_line = (manifest_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines()[0]
    train_manifest.write_text(train_line + "\n", encoding="utf-8")
    valid_manifest.write_text(valid_line + "\n", encoding="utf-8")

    output_dir = tmp_path / "outputs" / "checkpoints" / "turn_progress_test"
    log_dir = tmp_path / "outputs" / "logs" / "turn_progress_test"
    config_path = tmp_path / "turn_progress.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 20260423",
                "exp_name: turn_progress_test",
                f"output_dir: {output_dir.as_posix()}",
                f"log_dir: {log_dir.as_posix()}",
                "data:",
                f"  train_manifest: {train_manifest.as_posix()}",
                f"  valid_manifest: {valid_manifest.as_posix()}",
                "  candidate_limit: 64",
                "task:",
                "  task: turn",
                "model:",
                "  hidden_size: 32",
                "  dropout: 0.0",
                "train:",
                "  device: cpu",
                "  precision: bf16",
                "  batch_size: 64",
                "  num_workers: 0",
                "  epochs: 1",
                "  lr: 0.005",
                "  weight_decay: 0.0",
                "  log_every_n_steps: 1",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config_path)]) == 0

    captured = capsys.readouterr()
    assert "[train] starting task=turn" in captured.out
    assert "device=cpu" in captured.out
    assert "train_samples=" in captured.out
    assert "valid_samples=" in captured.out
    assert "train_batches=" in captured.out
    assert "valid_batches=" in captured.out
    assert "batch_size=64" in captured.out
    assert "num_workers=0" in captured.out
    assert "precision=bf16" in captured.out
    assert "autocast=False" in captured.out
    assert "output_dir=" in captured.out
    assert "[train] epoch=1 split=train batch=" in captured.out
    assert "batch_time=" in captured.out
    assert "data_wait=" in captured.out
    assert "compute_time=" in captured.out
    assert "avg_batch_time=" in captured.out
    assert "avg_data_wait=" in captured.out
    assert "avg_compute_time=" in captured.out
    assert "samples_per_s=" in captured.out
    assert "eta=" in captured.out


def test_train_response_smoke_from_generated_dataset(tmp_path: Path):
    """验证基于生成数据集的响应训练冒烟流程。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"
    manifest_dir = tmp_path / "manifests"

    assert main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]) == 0
    assert main(["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)]) == 0
    assert main(["make-manifest", "--input", str(silver_dir), "--output", str(manifest_dir)]) == 0

    train_manifest = manifest_dir / "train_small.jsonl"
    valid_manifest = manifest_dir / "valid_small.jsonl"
    train_line = (manifest_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    valid_line = (manifest_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines()[0]
    train_manifest.write_text(train_line + "\n", encoding="utf-8")
    valid_manifest.write_text(valid_line + "\n", encoding="utf-8")

    output_dir = tmp_path / "outputs" / "checkpoints" / "response_smoke_test"
    log_dir = tmp_path / "outputs" / "logs" / "response_smoke_test"
    config_path = tmp_path / "response_smoke.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 20260423",
                "exp_name: response_smoke_test",
                f"output_dir: {output_dir.as_posix()}",
                f"log_dir: {log_dir.as_posix()}",
                "data:",
                f"  train_manifest: {train_manifest.as_posix()}",
                f"  valid_manifest: {valid_manifest.as_posix()}",
                "  candidate_limit: 16",
                "task:",
                "  task: response",
                "model:",
                "  hidden_size: 32",
                "  dropout: 0.0",
                "train:",
                "  device: cpu",
                "  batch_size: 128",
                "  num_workers: 0",
                "  epochs: 1",
                "  lr: 0.005",
                "  weight_decay: 0.0",
                "  response_min_train_candidate_count: 2",
                "  response_class_weights:",
                "    pass: 1.0",
                "    chi: 6.0",
                "    pon: 6.0",
                "    daiminkan: 6.0",
                "    ron: 10.0",
                "    chankan: 10.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config_path)]) == 0

    metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert (output_dir / "config.resolved.yaml").exists()
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "best_accuracy.pt").exists()
    assert (output_dir / "best_loss.pt").exists()
    assert (output_dir / "best_macro.pt").exists()
    assert (output_dir / "best_response.pt").exists()
    assert (output_dir / "final.pt").exists()
    assert (log_dir / "epoch_metrics.jsonl").exists()
    assert metrics["train_samples"] > 0
    assert metrics["valid_samples"] > 0
    assert metrics["original_train_samples"] >= metrics["train_samples"]
    assert metrics["filtered_train_samples"] == (
        metrics["original_train_samples"] - metrics["train_samples"]
    )
    assert metrics["epoch_summaries"]
    first_epoch = metrics["epoch_summaries"][0]
    assert "macro_accuracy" in first_epoch
    assert first_epoch["response"] is None
    assert first_epoch["per_kind"]
    assert first_epoch["class_groups"]
    valid_epoch = metrics["epoch_summaries"][1]
    assert valid_epoch["response"]


def test_eval_turn_smoke_from_generated_checkpoint(tmp_path: Path):
    """验证基于生成检查点的自摸巡评估冒烟流程。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"
    manifest_dir = tmp_path / "manifests"

    assert main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]) == 0
    assert main(["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)]) == 0
    assert main(["make-manifest", "--input", str(silver_dir), "--output", str(manifest_dir)]) == 0

    train_manifest = manifest_dir / "train_small.jsonl"
    valid_manifest = manifest_dir / "valid_small.jsonl"
    train_line = (manifest_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    valid_line = (manifest_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines()[0]
    train_manifest.write_text(train_line + "\n", encoding="utf-8")
    valid_manifest.write_text(valid_line + "\n", encoding="utf-8")
    eval_dataset = _build_eval_dataset("turn", valid_manifest)
    assert isinstance(eval_dataset, StreamingManifestTurnDataset)
    assert eval_dataset.include_legal_actions is True
    assert eval_dataset.include_label_action is True

    output_dir = tmp_path / "outputs" / "checkpoints" / "turn_eval_test"
    log_dir = tmp_path / "outputs" / "logs" / "turn_eval_test"
    config_path = tmp_path / "turn_eval.yaml"
    report_path = tmp_path / "reports" / "turn_valid_eval.json"
    config_path.write_text(
        "\n".join(
            [
                "seed: 20260423",
                "exp_name: turn_eval_test",
                f"output_dir: {output_dir.as_posix()}",
                f"log_dir: {log_dir.as_posix()}",
                "data:",
                f"  train_manifest: {train_manifest.as_posix()}",
                f"  valid_manifest: {valid_manifest.as_posix()}",
                "  candidate_limit: 64",
                "task:",
                "  task: turn",
                "model:",
                "  hidden_size: 32",
                "  dropout: 0.0",
                "train:",
                "  device: cpu",
                "  batch_size: 64",
                "  num_workers: 0",
                "  epochs: 1",
                "  lr: 0.005",
                "  weight_decay: 0.0",
                "  turn_class_weights:",
                "    discard: 1.0",
                "    riichi_discard: 2.0",
                "    tsumo: 3.0",
                "    ankan: 4.0",
                "    kakan: 4.0",
                "    kyuushu_kyuuhai: 4.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config_path)]) == 0
    assert main(
        [
            "eval",
            "--ckpt",
            str(output_dir / "best.pt"),
            "--split",
            "valid",
            "--report",
            str(report_path),
        ]
    ) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task"] == "turn"
    assert report["split"] == "valid"
    assert report["samples"] > 0
    assert 0.0 <= report["top1_accuracy"] <= 1.0
    assert 0.0 <= report["top3_accuracy"] <= 1.0
    assert 0.0 <= report["top5_accuracy"] <= 1.0
    assert 0.0 <= report["macro_top1_accuracy"] <= 1.0
    assert report["per_kind"]
    assert report["class_groups"]
    assert report["candidate_buckets"]
    assert report["response"] is None
    assert "misclassified_examples" in report
    if report["misclassified_examples"]:
        example = report["misclassified_examples"][0]
        assert example["sample_locator"]["sample_id"] == example["sample_id"]
        assert "inspect-sample --dataset" in example["inspect_locator"]["inspect_sample_command"]
        assert "dump-case --bronze-dir" in example["case_locator"]["dump_case_command"]


def test_eval_response_smoke_from_generated_checkpoint(tmp_path: Path, capsys):
    """验证基于生成检查点的响应评估与校准冒烟流程。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
        capsys (Any): pytest 提供的标准输出捕获夹具。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"
    manifest_dir = tmp_path / "manifests"

    assert main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]) == 0
    assert main(["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)]) == 0
    assert main(["make-manifest", "--input", str(silver_dir), "--output", str(manifest_dir)]) == 0

    train_manifest = manifest_dir / "train_small.jsonl"
    valid_manifest = manifest_dir / "valid_small.jsonl"
    calibration_manifest = manifest_dir / "valid_calibration.jsonl"
    train_line = (manifest_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    valid_line = (manifest_dir / "valid.jsonl").read_text(encoding="utf-8").splitlines()[0]
    train_manifest.write_text(train_line + "\n", encoding="utf-8")
    valid_manifest.write_text(valid_line + "\n", encoding="utf-8")
    calibration_manifest.write_text(valid_line + "\n", encoding="utf-8")
    eval_dataset = _build_eval_dataset("response", valid_manifest)
    assert isinstance(eval_dataset, StreamingManifestResponseDataset)
    assert eval_dataset.include_legal_actions is True
    assert eval_dataset.include_label_action is True

    output_dir = tmp_path / "outputs" / "checkpoints" / "response_eval_test"
    log_dir = tmp_path / "outputs" / "logs" / "response_eval_test"
    config_path = tmp_path / "response_eval.yaml"
    report_path = tmp_path / "reports" / "response_valid_eval.json"
    calibration_report_path = tmp_path / "reports" / "response_valid_calibration.json"
    config_path.write_text(
        "\n".join(
            [
                "seed: 20260423",
                "exp_name: response_eval_test",
                f"output_dir: {output_dir.as_posix()}",
                f"log_dir: {log_dir.as_posix()}",
                "data:",
                f"  train_manifest: {train_manifest.as_posix()}",
                f"  valid_manifest: {valid_manifest.as_posix()}",
                "  candidate_limit: 16",
                "task:",
                "  task: response",
                "model:",
                "  hidden_size: 32",
                "  dropout: 0.0",
                "train:",
                "  device: cpu",
                "  batch_size: 128",
                "  num_workers: 0",
                "  epochs: 1",
                "  lr: 0.005",
                "  weight_decay: 0.0",
                "  response_class_weights:",
                "    pass: 1.0",
                "    chi: 6.0",
                "    pon: 6.0",
                "    daiminkan: 6.0",
                "    ron: 10.0",
                "    chankan: 10.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config_path)]) == 0
    assert main(
        [
            "eval",
            "--ckpt",
            str(output_dir / "best.pt"),
            "--split",
            "valid",
            "--report",
            str(report_path),
        ]
    ) == 0
    assert main(
        [
            "calibrate-response",
            "--ckpt",
            str(output_dir / "best.pt"),
            "--split",
            "valid",
            "--manifest",
            str(calibration_manifest),
            "--report",
            str(calibration_report_path),
            "--pass-biases",
            "0,0.1,0.2",
            "--false-call-targets",
            "100%,0.5%",
        ]
    ) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task"] == "response"
    assert report["split"] == "valid"
    assert report["samples"] > 0
    assert 0.0 <= report["top1_accuracy"] <= 1.0
    assert 0.0 <= report["top3_accuracy"] <= 1.0
    assert 0.0 <= report["top5_accuracy"] <= 1.0
    assert 0.0 <= report["macro_top1_accuracy"] <= 1.0
    assert report["per_kind"]
    assert report["class_groups"]
    assert report["candidate_buckets"]
    assert report["response"]
    assert "multi_candidate" in report["response"]
    assert "binary" in report["response"]
    assert "candidate_sets" in report["response"]
    assert "confusion" in report["response"]
    assert "misclassified_examples" in report
    if report["misclassified_examples"]:
        example = report["misclassified_examples"][0]
        assert example["sample_locator"]["sample_id"] == example["sample_id"]
        assert "inspect-sample --dataset" in example["inspect_locator"]["inspect_sample_command"]
        assert "dump-case --bronze-dir" in example["case_locator"]["dump_case_command"]

    calibration_report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
    calibration_dataset = _build_calibration_dataset(calibration_manifest)
    assert calibration_dataset.include_legal_actions is True
    assert calibration_dataset.shuffle_rows is False
    assert calibration_dataset.shuffle_shards is False
    assert calibration_report["task"] == "response"
    assert calibration_report["split"] == "valid"
    assert calibration_report["manifest_path"] == str(calibration_manifest)
    assert calibration_report["pass_biases"] == [0.0, 0.1, 0.2]
    assert calibration_report["false_call_rate_targets"] == [1.0, 0.005]
    assert len(calibration_report["points"]) == 3
    assert len(calibration_report["selections"]) == 2
    assert calibration_report["selections"][0]["selected_pass_bias"] is not None
    first_point = calibration_report["points"][0]
    assert first_point["samples"] > 0
    assert 0.0 <= first_point["top1_accuracy"] <= 1.0
    assert 0.0 <= first_point["false_call_rate"] <= 1.0
    assert 0.0 <= first_point["missed_call_rate"] <= 1.0
    assert "pass" in first_point["per_kind_samples"]

    captured = capsys.readouterr()
    assert "calibrated task=response split=valid" in captured.out
    assert "calibration_selection=target_false_call_rate=1.0000" in captured.out
