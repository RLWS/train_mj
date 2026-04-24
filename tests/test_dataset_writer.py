from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from mjai.data.builders.dataset_builder import DatasetBuilder
from mjai.data.builders.manifest_builder import ManifestBuilder


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_build_dataset_writes_turn_and_response_parquet(tmp_path: Path):
    builder = DatasetBuilder()
    summary = builder.build_from_xml_dir(FIXTURE_DIR, tmp_path / "silver_samples")

    turn_files = sorted((tmp_path / "silver_samples" / "turn").glob("*.parquet"))
    response_files = sorted((tmp_path / "silver_samples" / "response").glob("*.parquet"))

    assert summary.games == len(list(FIXTURE_DIR.glob("*.xml")))
    assert summary.turn_samples > 0
    assert summary.response_samples > 0
    assert len(turn_files) == summary.games
    assert len(response_files) == summary.games

    turn_rows = pq.read_table(turn_files[0]).to_pylist()
    assert turn_rows
    assert "sample_id" in turn_rows[0]
    assert "label_action_json" in turn_rows[0]


def test_make_manifest_writes_jsonl_splits(tmp_path: Path):
    dataset_builder = DatasetBuilder()
    dataset_builder.build_from_xml_dir(FIXTURE_DIR, tmp_path / "silver_samples")

    manifest_builder = ManifestBuilder()
    summary = manifest_builder.build(tmp_path / "silver_samples", tmp_path / "manifests")

    train_path = tmp_path / "manifests" / "train.jsonl"
    valid_path = tmp_path / "manifests" / "valid.jsonl"
    test_path = tmp_path / "manifests" / "test.jsonl"

    assert train_path.exists()
    assert valid_path.exists()
    assert test_path.exists()
    assert summary.train_games > 0

    first_line = train_path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    assert "game_id" in record
    assert "turn_path" in record
    assert "response_path" in record
