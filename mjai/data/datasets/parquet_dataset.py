from __future__ import annotations

import json
from array import array
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from mjai.data.schema.samples import TurnSample
from mjai.features.action_features import (
    RESPONSE_ACTION_DIM,
    TURN_ACTION_DIM,
    encode_response_action,
    encode_turn_candidate_actions,
)
from mjai.features.state_features import (
    RESPONSE_STATE_DIM,
    TURN_STATE_DIM,
    encode_response_state,
    encode_turn_state,
)
from mjai.utils.paths import ensure_dir


SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("game_id", pa.string()),
        ("kyoku_index", pa.int32()),
        ("step_index", pa.int32()),
        ("actor", pa.int8()),
        ("decision_type", pa.string()),
        ("state_json", pa.string()),
        ("legal_actions_json", pa.string()),
        ("label_action_json", pa.string()),
        ("state_features_bin", pa.binary()),
        ("candidate_features_bin", pa.binary()),
        ("label_index", pa.int16()),
        ("label_kind", pa.string()),
    ]
)


def serialize_sample(sample: TurnSample) -> dict[str, object]:
    state_features, candidate_features = _encode_training_features(sample)
    label_index, label_kind = _label_metadata(sample)
    return {
        "sample_id": sample.sample_id,
        "game_id": sample.game_id,
        "kyoku_index": sample.kyoku_index,
        "step_index": sample.step_index,
        "actor": sample.actor,
        "decision_type": sample.decision_type,
        "state_json": json.dumps(sample.state, ensure_ascii=False, sort_keys=True),
        "legal_actions_json": json.dumps(sample.legal_actions, ensure_ascii=False, sort_keys=True),
        "label_action_json": json.dumps(sample.label_action, ensure_ascii=False, sort_keys=True),
        "state_features_bin": _float32_bytes(state_features),
        "candidate_features_bin": _float32_bytes(
            value
            for candidate in candidate_features
            for value in candidate
        ),
        "label_index": label_index,
        "label_kind": label_kind,
    }


def write_samples_to_parquet(
    samples: Iterable[TurnSample],
    output_path: str | Path,
) -> int:
    rows = [serialize_sample(sample) for sample in samples]
    table = pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA)
    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    pq.write_table(table, output_file, compression="zstd")
    return table.num_rows


def _encode_training_features(sample: TurnSample) -> tuple[list[float], list[list[float]]]:
    if sample.decision_type == "response":
        return (
            encode_response_state(sample.state),
            [encode_response_action(action) for action in sample.legal_actions],
        )
    if sample.decision_type == "turn":
        return (
            encode_turn_state(sample.state),
            encode_turn_candidate_actions(sample.legal_actions, sample.state),
        )
    raise ValueError(f"unsupported decision_type for policy sample: {sample.decision_type}")


def _label_metadata(sample: TurnSample) -> tuple[int, str]:
    label_key = _action_key(sample.label_action)
    legal_action_keys = [_action_key(action) for action in sample.legal_actions]
    try:
        label_index = legal_action_keys.index(label_key)
    except ValueError as exc:
        raise ValueError(f"label action missing from legal actions: {sample.sample_id}") from exc
    return label_index, str(sample.label_action["kind"])


def _float32_bytes(values: Iterable[float]) -> bytes:
    encoded = array("f")
    encoded.extend(float(value) for value in values)
    if encoded.itemsize != 4:
        raise RuntimeError("array('f') is not 32-bit on this platform")
    return encoded.tobytes()


class ManifestTurnDataset(Dataset):
    """In-memory turn dataset loaded from manifest-listed parquet shards."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = _load_turn_rows(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class ManifestResponseDataset(Dataset):
    """In-memory response dataset loaded from manifest-listed parquet shards."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = _load_response_rows(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class TurnBatchCollator:
    def __init__(self, candidate_limit: int = 64) -> None:
        self.candidate_limit = candidate_limit

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        return build_turn_batch(rows, candidate_limit=self.candidate_limit)


class ResponseBatchCollator:
    def __init__(self, candidate_limit: int = 64) -> None:
        self.candidate_limit = candidate_limit

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor | list[str]]:
        return build_response_batch(rows, candidate_limit=self.candidate_limit)


def inspect_dataset_sample(dataset_path: str | Path, index: int) -> dict[str, object]:
    if index < 0:
        raise IndexError(f"sample index must be non-negative, got: {index}")

    source_path = Path(dataset_path)
    parquet_files = _resolve_sample_parquet_files(source_path)
    remaining = index

    for parquet_path in parquet_files:
        rows_in_file = pq.read_metadata(parquet_path).num_rows
        if remaining < rows_in_file:
            table = pq.read_table(parquet_path).slice(remaining, 1)
            rows = table.to_pylist()
            if not rows:
                break
            return _build_inspected_sample(
                row=rows[0],
                dataset_path=source_path,
                source_path=parquet_path,
                global_index=index,
                source_row_index=remaining,
            )
        remaining -= rows_in_file

    raise IndexError(f"sample index out of range: {index}")


def inspect_dataset_sample_by_id(dataset_path: str | Path, sample_id: str) -> dict[str, object]:
    source_path = Path(dataset_path)
    parquet_files = _resolve_sample_parquet_files(source_path)
    global_index = 0

    for parquet_path in parquet_files:
        rows = pq.read_table(parquet_path).to_pylist()
        for source_row_index, row in enumerate(rows):
            if str(row["sample_id"]) != sample_id:
                global_index += 1
                continue
            return _build_inspected_sample(
                row=row,
                dataset_path=source_path,
                source_path=parquet_path,
                global_index=global_index,
                source_row_index=source_row_index,
            )

    raise ValueError(f"sample_id not found in dataset: {sample_id}")


def build_turn_batch(rows: list[dict], candidate_limit: int = 64) -> dict[str, torch.Tensor | list[str]]:
    max_candidates = max(len(row["legal_actions"]) for row in rows)
    if max_candidates > candidate_limit:
        raise ValueError(f"candidate count exceeds configured limit: {max_candidates} > {candidate_limit}")

    batch_size = len(rows)
    state_feats = torch.zeros((batch_size, TURN_STATE_DIM), dtype=torch.float32)
    candidate_feats = torch.zeros((batch_size, max_candidates, TURN_ACTION_DIM), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    label_kinds: list[str] = []
    sample_ids: list[str] = []
    game_ids: list[str] = []
    kyoku_indices: list[int] = []
    step_indices: list[int] = []
    actors: list[int] = []
    decision_types: list[str] = []

    for batch_index, row in enumerate(rows):
        _copy_float32_feature_bytes(
            state_feats[batch_index],
            row["state_features_bin"],
            expected_values=TURN_STATE_DIM,
        )
        labels[batch_index] = _row_label_index(row)
        candidate_count = len(row["legal_actions"])

        if candidate_count > 0:
            _copy_float32_feature_bytes(
                candidate_feats[batch_index, :candidate_count],
                row["candidate_features_bin"],
                expected_values=candidate_count * TURN_ACTION_DIM,
            )
            candidate_mask[batch_index, :candidate_count] = True
        label_kinds.append(str(row["label_kind"]))
        sample_ids.append(str(row["sample_id"]))
        game_ids.append(str(row["game_id"]))
        kyoku_indices.append(int(row["kyoku_index"]))
        step_indices.append(int(row["step_index"]))
        actors.append(int(row["actor"]))
        decision_types.append(str(row["decision_type"]))

    return {
        "state_feats": state_feats,
        "candidate_feats": candidate_feats,
        "candidate_mask": candidate_mask,
        "labels": labels,
        "label_kinds": label_kinds,
        "sample_ids": sample_ids,
        "label_actions": [row["label_action"] for row in rows],
        "legal_actions": [row["legal_actions"] for row in rows],
        "candidate_counts": [len(row["legal_actions"]) for row in rows],
        "game_ids": game_ids,
        "kyoku_indices": kyoku_indices,
        "step_indices": step_indices,
        "actors": actors,
        "decision_types": decision_types,
    }


def build_response_batch(rows: list[dict], candidate_limit: int = 64) -> dict[str, torch.Tensor | list[str]]:
    max_candidates = max(len(row["legal_actions"]) for row in rows)
    if max_candidates > candidate_limit:
        raise ValueError(f"candidate count exceeds configured limit: {max_candidates} > {candidate_limit}")

    batch_size = len(rows)
    state_feats = torch.zeros((batch_size, RESPONSE_STATE_DIM), dtype=torch.float32)
    candidate_feats = torch.zeros((batch_size, max_candidates, RESPONSE_ACTION_DIM), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    label_kinds: list[str] = []
    sample_ids: list[str] = []
    game_ids: list[str] = []
    kyoku_indices: list[int] = []
    step_indices: list[int] = []
    actors: list[int] = []
    decision_types: list[str] = []

    for batch_index, row in enumerate(rows):
        _copy_float32_feature_bytes(
            state_feats[batch_index],
            row["state_features_bin"],
            expected_values=RESPONSE_STATE_DIM,
        )
        labels[batch_index] = _row_label_index(row)
        candidate_count = len(row["legal_actions"])

        if candidate_count > 0:
            _copy_float32_feature_bytes(
                candidate_feats[batch_index, :candidate_count],
                row["candidate_features_bin"],
                expected_values=candidate_count * RESPONSE_ACTION_DIM,
            )
            candidate_mask[batch_index, :candidate_count] = True
        label_kinds.append(str(row["label_kind"]))
        sample_ids.append(str(row["sample_id"]))
        game_ids.append(str(row["game_id"]))
        kyoku_indices.append(int(row["kyoku_index"]))
        step_indices.append(int(row["step_index"]))
        actors.append(int(row["actor"]))
        decision_types.append(str(row["decision_type"]))

    return {
        "state_feats": state_feats,
        "candidate_feats": candidate_feats,
        "candidate_mask": candidate_mask,
        "labels": labels,
        "label_kinds": label_kinds,
        "sample_ids": sample_ids,
        "label_actions": [row["label_action"] for row in rows],
        "legal_actions": [row["legal_actions"] for row in rows],
        "candidate_counts": [len(row["legal_actions"]) for row in rows],
        "game_ids": game_ids,
        "kyoku_indices": kyoku_indices,
        "step_indices": step_indices,
        "actors": actors,
        "decision_types": decision_types,
    }


def _copy_float32_feature_bytes(
    target: torch.Tensor,
    data: bytes,
    expected_values: int,
) -> None:
    source = torch.frombuffer(bytearray(data), dtype=torch.float32)
    if int(source.numel()) != expected_values:
        raise ValueError(f"feature byte length mismatch: {source.numel()} != {expected_values}")
    target.copy_(source.reshape(target.shape))


def _load_turn_rows(manifest_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        turn_path = Path(record["turn_path"])
        rows.extend(_deserialize_dataset_row(row) for row in pq.read_table(turn_path).to_pylist())
    return rows


def _load_response_rows(manifest_path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest file not found: {manifest_path}")

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        response_path = Path(record["response_path"])
        rows.extend(_deserialize_dataset_row(row) for row in pq.read_table(response_path).to_pylist())
    return rows


def _action_key(action: dict) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def _row_label_index(row: dict) -> int:
    label_index = row.get("label_index")
    if label_index is None:
        raise ValueError(f"label action missing from legal actions: {row['sample_id']}")
    return int(label_index)


def _resolve_sample_parquet_files(dataset_path: Path) -> list[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix != ".parquet":
            raise ValueError(f"inspect-sample expects a sample directory or parquet file: {dataset_path}")
        return [dataset_path]

    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset path not found: {dataset_path}")

    parquet_files: list[Path] = []
    turn_dir = dataset_path / "turn"
    response_dir = dataset_path / "response"

    if turn_dir.is_dir():
        parquet_files.extend(sorted(turn_dir.glob("*.parquet")))
    if response_dir.is_dir():
        parquet_files.extend(sorted(response_dir.glob("*.parquet")))
    if parquet_files:
        return parquet_files

    parquet_files = sorted(dataset_path.glob("*.parquet"))
    if parquet_files:
        return parquet_files

    raise FileNotFoundError(f"no parquet sample files found under: {dataset_path}")


def _build_inspected_sample(
    row: dict[str, object],
    dataset_path: Path,
    source_path: Path,
    global_index: int,
    source_row_index: int,
) -> dict[str, object]:
    legal_actions = json.loads(str(row["legal_actions_json"]))
    label_action = json.loads(str(row["label_action_json"]))
    state = json.loads(str(row["state_json"]))
    label_key = _action_key(label_action)
    legal_action_keys = [_action_key(action) for action in legal_actions]
    label_index = int(_required_sample_field(row, "label_index"))
    label_in_legal = (
        0 <= label_index < len(legal_action_keys)
        and legal_action_keys[label_index] == label_key
    )
    state_features_bin = bytes(_required_sample_field(row, "state_features_bin"))
    candidate_features_bin = bytes(_required_sample_field(row, "candidate_features_bin"))

    return {
        "dataset_path": str(dataset_path.as_posix()),
        "global_index": global_index,
        "source_path": str(source_path.as_posix()),
        "source_row_index": source_row_index,
        "sample_id": row["sample_id"],
        "game_id": row["game_id"],
        "kyoku_index": int(row["kyoku_index"]),
        "step_index": int(row["step_index"]),
        "actor": int(row["actor"]),
        "decision_type": row["decision_type"],
        "candidate_count": len(legal_actions),
        "label_in_legal": label_in_legal,
        "label_index": label_index,
        "label_kind": _required_sample_field(row, "label_kind"),
        "state_feature_bytes": len(state_features_bin),
        "candidate_feature_bytes": len(candidate_features_bin),
        "label_action": label_action,
        "legal_actions": legal_actions,
        "state": state,
    }


def _deserialize_dataset_row(row: dict[str, object]) -> dict[str, object]:
    legal_actions = json.loads(row["legal_actions_json"])
    label_action = json.loads(row["label_action_json"])

    return {
        "sample_id": row["sample_id"],
        "game_id": row["game_id"],
        "kyoku_index": int(row["kyoku_index"]),
        "step_index": int(row["step_index"]),
        "actor": int(row["actor"]),
        "decision_type": row["decision_type"],
        "state": json.loads(row["state_json"]),
        "legal_actions": legal_actions,
        "label_action": label_action,
        "state_features_bin": bytes(_required_sample_field(row, "state_features_bin")),
        "candidate_features_bin": bytes(_required_sample_field(row, "candidate_features_bin")),
        "label_index": int(_required_sample_field(row, "label_index")),
        "label_kind": str(_required_sample_field(row, "label_kind")),
    }


def _required_sample_field(row: dict[str, object], field_name: str) -> object:
    try:
        return row[field_name]
    except KeyError as exc:
        raise ValueError(
            f"sample parquet is missing '{field_name}'; rebuild data/silver_samples "
            "with the current build-dataset command and regenerate manifests"
        ) from exc
