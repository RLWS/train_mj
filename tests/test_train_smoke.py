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
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config_path)]) == 0

    metrics = json.loads((output_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert (output_dir / "config.resolved.yaml").exists()
    assert (output_dir / "best.pt").exists()
    assert (output_dir / "final.pt").exists()
    assert (log_dir / "epoch_metrics.jsonl").exists()
    assert metrics["train_samples"] > 0
    assert metrics["valid_samples"] > 0
    assert metrics["epoch_summaries"]
