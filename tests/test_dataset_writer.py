from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from mjai.data.builders.event_builder import EventBuilder
from mjai.data.builders.dataset_builder import DatasetBuilder
from mjai.data.datasets.event_store import read_events_from_parquet
from mjai.data.builders.manifest_builder import ManifestBuilder


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_preprocess_writes_bronze_event_parquet(tmp_path: Path):
    builder = EventBuilder()
    summary = builder.build_from_xml_dir(FIXTURE_DIR, tmp_path / "bronze_events")

    event_files = sorted((tmp_path / "bronze_events").glob("*.parquet"))
    preprocess_summary = json.loads(
        (tmp_path / "bronze_events" / "preprocess_summary.json").read_text(encoding="utf-8")
    )

    assert summary.games == len(list(FIXTURE_DIR.glob("*.xml")))
    assert summary.events > 0
    assert len(event_files) == summary.games
    assert preprocess_summary["event_type_counts"]["init"] > 0

    events = read_events_from_parquet(event_files[0])
    assert events
    assert events[0].game_id == event_files[0].stem


def test_build_dataset_writes_turn_and_response_parquet(tmp_path: Path):
    event_builder = EventBuilder()
    event_builder.build_from_xml_dir(FIXTURE_DIR, tmp_path / "bronze_events")

    builder = DatasetBuilder()
    summary = builder.build_from_event_dir(tmp_path / "bronze_events", tmp_path / "silver_samples")

    turn_files = sorted((tmp_path / "silver_samples" / "turn").glob("*.parquet"))
    response_files = sorted((tmp_path / "silver_samples" / "response").glob("*.parquet"))
    build_summary = json.loads(
        (tmp_path / "silver_samples" / "build_summary.json").read_text(encoding="utf-8")
    )

    assert summary.games == len(list(FIXTURE_DIR.glob("*.xml")))
    assert summary.turn_samples > 0
    assert summary.response_samples > 0
    assert len(turn_files) == summary.games
    assert len(response_files) == summary.games
    assert build_summary["turn_label_counts"]["discard"] > 0
    assert build_summary["turn_label_counts"]["tsumo"] > 0
    assert build_summary["turn_label_counts"]["ankan"] > 0
    assert build_summary["turn_label_counts"]["kyuushu_kyuuhai"] > 0
    assert build_summary["response_label_counts"]["pass"] > 0

    turn_rows = pq.read_table(turn_files[0]).to_pylist()
    assert turn_rows
    assert "sample_id" in turn_rows[0]
    assert "label_action_json" in turn_rows[0]


def test_make_manifest_writes_jsonl_splits(tmp_path: Path):
    event_builder = EventBuilder()
    event_builder.build_from_xml_dir(FIXTURE_DIR, tmp_path / "bronze_events")

    dataset_builder = DatasetBuilder()
    dataset_builder.build_from_event_dir(tmp_path / "bronze_events", tmp_path / "silver_samples")

    manifest_builder = ManifestBuilder()
    summary = manifest_builder.build(tmp_path / "silver_samples", tmp_path / "manifests")

    train_path = tmp_path / "manifests" / "train.jsonl"
    valid_path = tmp_path / "manifests" / "valid.jsonl"
    test_path = tmp_path / "manifests" / "test.jsonl"
    summary_path = tmp_path / "manifests" / "manifest_summary.json"

    assert train_path.exists()
    assert valid_path.exists()
    assert test_path.exists()
    assert summary_path.exists()
    assert summary.train_games > 0

    summary_json = json.loads(summary_path.read_text(encoding="utf-8"))
    first_line = train_path.read_text(encoding="utf-8").splitlines()[0]
    record = json.loads(first_line)
    assert "game_id" in record
    assert "turn_path" in record
    assert "response_path" in record
    assert summary_json["train_turn_samples"] > 0
    assert summary_json["train_games"] > 0
    assert summary_json["valid_games"] > 0
    assert summary_json["test_games"] > 0
