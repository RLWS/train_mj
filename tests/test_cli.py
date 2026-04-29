from __future__ import annotations

import json
from pathlib import Path

from mjai.cli import main


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_cli_preprocess_and_validate_events(tmp_path: Path, capsys):
    bronze_dir = tmp_path / "bronze_events"
    report_path = tmp_path / "reports" / "event_validation.json"

    preprocess_exit_code = main(
        ["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]
    )
    preprocess_output = capsys.readouterr().out

    validate_exit_code = main(
        ["validate-events", "--input", str(bronze_dir), "--report", str(report_path)]
    )
    captured = capsys.readouterr()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert preprocess_exit_code == 0
    assert validate_exit_code == 0
    assert "preprocessed events:" in preprocess_output
    assert "event_types=" in preprocess_output
    assert "validated events:" in captured.out
    assert "event types:" in captured.out
    assert report["summary"]["files_seen"] == len(list(FIXTURE_DIR.glob("*.xml")))
    assert report["summary"]["files_failed"] == 0
    assert report["summary"]["event_type_counts"]["init"] > 0


def test_cli_build_dataset_from_bronze_events(tmp_path: Path, capsys):
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"

    preprocess_exit_code = main(
        ["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]
    )
    preprocess_output = capsys.readouterr().out

    build_exit_code = main(
        ["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir), "--workers", "1"]
    )
    build_output = capsys.readouterr().out

    assert preprocess_exit_code == 0
    assert build_exit_code == 0
    assert "preprocessed events:" in preprocess_output
    assert "built dataset:" in build_output
    assert "turn labels:" in build_output
    assert "response labels:" in build_output
    assert (silver_dir / "turn").exists()
    assert (silver_dir / "response").exists()


def test_cli_make_manifest_prints_sample_counts_and_warnings(tmp_path: Path, capsys):
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"
    manifest_dir = tmp_path / "manifests"

    main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)])
    capsys.readouterr()
    main(["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)])
    capsys.readouterr()

    exit_code = main(["make-manifest", "--input", str(silver_dir), "--output", str(manifest_dir)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "built manifests:" in captured.out
    assert "manifest samples:" in captured.out
    assert "train_games=" in captured.out
    assert "valid_games=" in captured.out
    assert "test_games=" in captured.out


def test_cli_train_requires_real_inputs(tmp_path: Path):
    config_path = tmp_path / "missing_data.yaml"
    missing_manifest = tmp_path / "missing_train.jsonl"
    config_path.write_text(
        "\n".join(
            [
                "seed: 20260423",
                "exp_name: missing_data_test",
                "output_dir: outputs/checkpoints/missing_data_test",
                "log_dir: outputs/logs/missing_data_test",
                "data:",
                f"  train_manifest: {missing_manifest.as_posix()}",
                "task:",
                "  task: turn",
                "model:",
                "  hidden_size: 32",
                "  dropout: 0.0",
                "train:",
                "  device: cpu",
                "  batch_size: 8",
                "  num_workers: 0",
                "  epochs: 1",
                "  lr: 0.001",
                "  weight_decay: 0.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    try:
        main(["train", "--config", str(config_path)])
    except FileNotFoundError as exc:
        assert "manifest file not found" in str(exc)
    else:
        raise AssertionError("expected train command to require an existing manifest")


def test_cli_inspect_sample_reads_from_sample_directory(tmp_path: Path, capsys):
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"

    assert main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]) == 0
    capsys.readouterr()
    assert main(["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)]) == 0
    capsys.readouterr()

    exit_code = main(["inspect-sample", "--dataset", str(silver_dir), "--index", "0"])
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["dataset_path"] == silver_dir.as_posix()
    assert payload["global_index"] == 0
    assert payload["sample_id"]
    assert payload["decision_type"] in {"turn", "response"}
    assert payload["candidate_count"] > 0
    assert isinstance(payload["legal_actions"], list)
    assert isinstance(payload["state"], dict)
    assert "kyoku_index" in payload


def test_cli_inspect_sample_reads_by_sample_id(tmp_path: Path, capsys):
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"

    assert main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]) == 0
    capsys.readouterr()
    assert main(["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "inspect-sample",
            "--dataset",
            str(silver_dir),
            "--sample-id",
            "2023010100gm-00e1-0000-8a80056a:0:2:0:turn",
        ]
    )
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["sample_id"] == "2023010100gm-00e1-0000-8a80056a:0:2:0:turn"
    assert payload["game_id"] == "2023010100gm-00e1-0000-8a80056a"
    assert "temporary_furiten" in payload["state"]
    assert "riichi_furiten" in payload["state"]


def test_cli_dump_case_reads_bronze_event_context(tmp_path: Path, capsys):
    bronze_dir = tmp_path / "bronze_events"

    assert main(["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]) == 0
    capsys.readouterr()

    exit_code = main(
        [
            "dump-case",
            "--bronze-dir",
            str(bronze_dir),
            "--game-id",
            "2023010100gm-00e1-0000-8a80056a",
            "--kyoku-id",
            "0",
            "--step-id",
            "1",
            "--context-radius",
            "2",
        ]
    )
    captured = capsys.readouterr()

    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["game_id"] == "2023010100gm-00e1-0000-8a80056a"
    assert payload["kyoku_index"] == 0
    assert payload["step_index"] == 1
    assert payload["target_event"]["event_type"]
    assert isinstance(payload["kyoku_events"], list)
    assert isinstance(payload["context_before"], list)
    assert isinstance(payload["context_after"], list)
