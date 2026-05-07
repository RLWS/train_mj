from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from mjai.data.builders.event_builder import EventBuilder
from mjai.data.builders.dataset_builder import DatasetBuilder
from mjai.data.datasets.parquet_dataset import (
    ManifestTurnDataset,
    StreamingManifestResponseDataset,
    StreamingManifestTurnDataset,
    ResponseBatchCollator,
    TurnBatchCollator,
)
from mjai.data.datasets.event_store import read_events_from_parquet
from mjai.data.builders.manifest_builder import ManifestBuilder
from mjai.features.action_features import RESPONSE_ACTION_DIM, TURN_ACTION_DIM
from mjai.features.state_features import RESPONSE_STATE_DIM, TURN_STATE_DIM


FIXTURE_DIR = Path("tests/fixtures/tenhou")


def test_preprocess_writes_bronze_event_parquet(tmp_path: Path):
    """验证预处理流程写出青铜层事件 Parquet。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
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
    """验证数据集构建流程写出自摸巡和响应 Parquet。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
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
    assert build_summary["elapsed_seconds"] >= 0
    assert build_summary["samples_per_second"] > 0

    turn_file = next(path for path in turn_files if pq.read_metadata(path).num_rows > 0)
    turn_rows = pq.read_table(turn_file).to_pylist()
    assert turn_rows
    assert "sample_id" in turn_rows[0]
    assert "label_action_json" in turn_rows[0]
    assert "state_features_bin" in turn_rows[0]
    assert "candidate_features_bin" in turn_rows[0]
    assert "label_index" in turn_rows[0]
    assert "label_kind" in turn_rows[0]

    first_legal_actions = json.loads(turn_rows[0]["legal_actions_json"])
    assert len(bytes(turn_rows[0]["state_features_bin"])) == TURN_STATE_DIM * 4
    assert len(bytes(turn_rows[0]["candidate_features_bin"])) == len(first_legal_actions) * TURN_ACTION_DIM * 4

    response_file = next(path for path in response_files if pq.read_metadata(path).num_rows > 0)
    response_rows = pq.read_table(response_file).to_pylist()
    response_legal_actions = json.loads(response_rows[0]["legal_actions_json"])
    assert len(bytes(response_rows[0]["state_features_bin"])) == RESPONSE_STATE_DIM * 4
    assert len(bytes(response_rows[0]["candidate_features_bin"])) == (
        len(response_legal_actions) * RESPONSE_ACTION_DIM * 4
    )

    manifest_path = tmp_path / "turn_only.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "game_id": turn_file.stem,
                "turn_path": turn_file.as_posix(),
                "turn_rows": pq.read_metadata(turn_file).num_rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ManifestTurnDataset(manifest_path)
    batch = TurnBatchCollator(candidate_limit=64)(dataset.rows[:4])
    assert batch["state_feats"].shape[1] == TURN_STATE_DIM
    assert batch["candidate_feats"].shape[2] == TURN_ACTION_DIM

    streaming_dataset = StreamingManifestTurnDataset(manifest_path, shuffle_rows=False)
    streaming_rows = [row for _, row in zip(range(4), streaming_dataset)]
    streaming_batch = TurnBatchCollator(candidate_limit=64)(streaming_rows)
    assert len(streaming_dataset) == len(dataset)
    assert streaming_batch["state_feats"].shape[1] == TURN_STATE_DIM
    assert streaming_batch["candidate_feats"].shape[2] == TURN_ACTION_DIM

    streaming_eval_dataset = StreamingManifestTurnDataset(
        manifest_path,
        shuffle_rows=False,
        include_legal_actions=True,
        include_label_action=True,
    )
    streaming_eval_row = next(iter(streaming_eval_dataset))
    assert streaming_eval_row["legal_actions"]
    assert streaming_eval_row["label_action"]
    assert "state" not in streaming_eval_row

    response_manifest = tmp_path / "response_only.jsonl"
    response_file = next(path for path in response_files if pq.read_metadata(path).num_rows > 0)
    response_manifest.write_text(
        json.dumps(
            {
                "game_id": response_file.stem,
                "response_path": response_file.as_posix(),
                "response_rows": pq.read_metadata(response_file).num_rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    response_dataset = StreamingManifestResponseDataset(
        response_manifest,
        min_candidate_count=2,
        shuffle_rows=False,
    )
    response_rows = [row for _, row in zip(range(4), response_dataset)]
    response_batch = ResponseBatchCollator(candidate_limit=16)(response_rows)
    assert response_rows
    assert all(row["candidate_count"] >= 2 for row in response_rows)
    assert response_batch["state_feats"].shape[1] == RESPONSE_STATE_DIM
    assert response_batch["candidate_feats"].shape[2] == RESPONSE_ACTION_DIM

    response_eval_dataset = StreamingManifestResponseDataset(
        response_manifest,
        min_candidate_count=1,
        shuffle_rows=False,
        include_legal_actions=True,
        include_label_action=True,
    )
    response_eval_row = next(iter(response_eval_dataset))
    assert response_eval_row["legal_actions"]
    assert response_eval_row["label_action"]
    assert "state" not in response_eval_row


def test_build_dataset_supports_parallel_workers(tmp_path: Path):
    """验证数据集构建支持并行工作单元。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
    event_builder = EventBuilder()
    event_builder.build_from_xml_dir(FIXTURE_DIR, tmp_path / "bronze_events")

    builder = DatasetBuilder()
    summary = builder.build_from_event_dir(
        tmp_path / "bronze_events",
        tmp_path / "silver_samples",
        workers=2,
    )

    turn_files = sorted((tmp_path / "silver_samples" / "turn").glob("*.parquet"))
    response_files = sorted((tmp_path / "silver_samples" / "response").glob("*.parquet"))

    assert summary.games == len(list(FIXTURE_DIR.glob("*.xml")))
    assert summary.turn_samples > 0
    assert summary.response_samples > 0
    assert len(turn_files) == summary.games
    assert len(response_files) == summary.games


def test_make_manifest_writes_jsonl_splits(tmp_path: Path):
    """验证清单构建流程写出 JSONL 数据划分。
    
    参数:
        tmp_path (Path): pytest 提供的临时目录路径。
    
    返回:
        None: 无返回值。
    
    异常:
        无。
    """
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
