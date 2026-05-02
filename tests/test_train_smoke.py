from __future__ import annotations

import json
from pathlib import Path

from mjai.cli import main


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_train_turn_smoke_from_generated_dataset(tmp_path: Path):
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
    assert (output_dir / "final.pt").exists()
    assert (log_dir / "epoch_metrics.jsonl").exists()
    assert metrics["train_samples"] > 0
    assert metrics["valid_samples"] > 0
    assert metrics["epoch_summaries"]
    first_epoch = metrics["epoch_summaries"][0]
    assert "macro_accuracy" in first_epoch
    assert first_epoch["per_kind"]
    assert first_epoch["class_groups"]


def test_train_prints_batch_progress_from_generated_dataset(tmp_path: Path, capsys):
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
    assert (output_dir / "final.pt").exists()
    assert (log_dir / "epoch_metrics.jsonl").exists()
    assert metrics["train_samples"] > 0
    assert metrics["valid_samples"] > 0
    assert metrics["epoch_summaries"]
    first_epoch = metrics["epoch_summaries"][0]
    assert "macro_accuracy" in first_epoch
    assert first_epoch["per_kind"]
    assert first_epoch["class_groups"]


def test_eval_turn_smoke_from_generated_checkpoint(tmp_path: Path):
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
    assert "misclassified_examples" in report
    if report["misclassified_examples"]:
        example = report["misclassified_examples"][0]
        assert example["sample_locator"]["sample_id"] == example["sample_id"]
        assert "inspect-sample --dataset" in example["inspect_locator"]["inspect_sample_command"]
        assert "dump-case --bronze-dir" in example["case_locator"]["dump_case_command"]


def test_eval_response_smoke_from_generated_checkpoint(tmp_path: Path):
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

    output_dir = tmp_path / "outputs" / "checkpoints" / "response_eval_test"
    log_dir = tmp_path / "outputs" / "logs" / "response_eval_test"
    config_path = tmp_path / "response_eval.yaml"
    report_path = tmp_path / "reports" / "response_valid_eval.json"
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
    assert "misclassified_examples" in report
    if report["misclassified_examples"]:
        example = report["misclassified_examples"][0]
        assert example["sample_locator"]["sample_id"] == example["sample_id"]
        assert "inspect-sample --dataset" in example["inspect_locator"]["inspect_sample_command"]
        assert "dump-case --bronze-dir" in example["case_locator"]["dump_case_command"]
