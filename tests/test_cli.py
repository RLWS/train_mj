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
    assert "validated events:" in captured.out
    assert report["summary"]["files_seen"] == len(list(FIXTURE_DIR.glob("*.xml")))
    assert report["summary"]["files_failed"] == 0


def test_cli_build_dataset_from_bronze_events(tmp_path: Path, capsys):
    bronze_dir = tmp_path / "bronze_events"
    silver_dir = tmp_path / "silver_samples"

    preprocess_exit_code = main(
        ["preprocess", "--input", str(FIXTURE_DIR), "--output", str(bronze_dir)]
    )
    preprocess_output = capsys.readouterr().out

    build_exit_code = main(
        ["build-dataset", "--input", str(bronze_dir), "--output", str(silver_dir)]
    )
    build_output = capsys.readouterr().out

    assert preprocess_exit_code == 0
    assert build_exit_code == 0
    assert "preprocessed events:" in preprocess_output
    assert "built dataset:" in build_output
    assert (silver_dir / "turn").exists()
    assert (silver_dir / "response").exists()


def test_cli_train_command_is_still_registered(capsys):
    exit_code = main(["train", "--config", "configs/base.yaml"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "mjai command 'train' is registered" in captured.out
    assert "loaded config: configs/base.yaml" in captured.out
