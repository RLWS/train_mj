from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from mjai.data.schema.samples import TurnSample
from mjai.features.action_features import (
    RESPONSE_ACTION_DIM,
    RESPONSE_ACTION_KINDS,
    TURN_ACTION_DIM,
    TURN_ACTION_KINDS,
)
from mjai.features.state_features import (
    ACTOR_DIM,
    EXACT_TILE_INDEX,
    EXACT_TILE_VOCAB,
    OPTIONAL_ACTOR_DIM,
    OPTIONAL_TILE_DIM,
    RESPONSE_STATE_DIM,
    TURN_STATE_DIM,
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
    ]
)


def serialize_sample(sample: TurnSample) -> dict[str, object]:
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
        _fill_turn_state_features(state_feats[batch_index], row["state"])
        labels[batch_index] = _row_label_index(row)

        for candidate_index, action in enumerate(row["legal_actions"]):
            _fill_turn_action_features(candidate_feats[batch_index, candidate_index], action)
            candidate_mask[batch_index, candidate_index] = True
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
        _fill_response_state_features(state_feats[batch_index], row["state"])
        labels[batch_index] = _row_label_index(row)

        for candidate_index, action in enumerate(row["legal_actions"]):
            _fill_response_action_features(candidate_feats[batch_index, candidate_index], action)
            candidate_mask[batch_index, candidate_index] = True
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


def _fill_turn_state_features(target: torch.Tensor, state: dict) -> None:
    offset = 0
    offset = _fill_exact_tile_counter(target, offset, state.get("hand", {}))

    all_discards = state.get("all_discards", [])
    for player_index in range(4):
        player_discards = all_discards[player_index] if player_index < len(all_discards) else []
        discard_counts: dict[str, int] = {}
        for tile in player_discards:
            discard_counts[tile] = discard_counts.get(tile, 0) + 1
        offset = _fill_exact_tile_counter(target, offset, discard_counts)

    offset = _fill_optional_tile(target, offset, state.get("last_discard_tile"))
    offset = _fill_optional_actor(target, offset, state.get("last_discard_actor"))
    offset = _fill_actor_onehot(target, offset, int(state.get("actor", 0)))
    target[offset] = 1.0 if state.get("riichi_declared", False) else 0.0
    offset += 1
    target[offset] = 1.0 if state.get("temporary_furiten", False) else 0.0
    offset += 1
    target[offset] = 1.0 if state.get("riichi_furiten", False) else 0.0
    offset += 1
    _fill_own_meld_counts(target, offset, state.get("own_melds", []))


def _fill_response_state_features(target: torch.Tensor, state: dict) -> None:
    _fill_turn_state_features(target[:TURN_STATE_DIM], state)
    offset = TURN_STATE_DIM
    response_target = state.get("response_target", {})
    offset = _fill_optional_actor(target, offset, response_target.get("source"))
    _fill_optional_tile(target, offset, response_target.get("tile"))


def _fill_turn_action_features(target: torch.Tensor, action: dict) -> None:
    kind = str(action.get("kind", "unknown"))
    try:
        target[TURN_ACTION_KINDS.index(kind)] = 1.0
    except ValueError:
        target[TURN_ACTION_KINDS.index("unknown")] = 1.0

    tile = action.get("tile")
    if tile is not None:
        target[len(TURN_ACTION_KINDS) + EXACT_TILE_INDEX[str(tile)]] = 1.0


def _fill_response_action_features(target: torch.Tensor, action: dict) -> None:
    kind = str(action.get("kind", "unknown"))
    try:
        target[RESPONSE_ACTION_KINDS.index(kind)] = 1.0
    except ValueError:
        target[RESPONSE_ACTION_KINDS.index("unknown")] = 1.0

    offset = len(RESPONSE_ACTION_KINDS)
    tile = action.get("tile")
    if tile is not None:
        target[offset + EXACT_TILE_INDEX[str(tile)]] = 1.0

    source = action.get("source")
    if source is not None:
        target[offset + len(EXACT_TILE_VOCAB) + int(source)] = 1.0


def _fill_exact_tile_counter(target: torch.Tensor, offset: int, counter: dict[str, int]) -> int:
    for tile, count in counter.items():
        index = EXACT_TILE_INDEX.get(tile)
        if index is not None:
            target[offset + index] = float(count)
    return offset + len(EXACT_TILE_VOCAB)


def _fill_optional_tile(target: torch.Tensor, offset: int, tile: str | None) -> int:
    if tile is None:
        target[offset + OPTIONAL_TILE_DIM - 1] = 1.0
    else:
        target[offset + EXACT_TILE_INDEX[tile]] = 1.0
    return offset + OPTIONAL_TILE_DIM


def _fill_optional_actor(target: torch.Tensor, offset: int, actor: int | None) -> int:
    if actor is None:
        target[offset + OPTIONAL_ACTOR_DIM - 1] = 1.0
    else:
        target[offset + int(actor)] = 1.0
    return offset + OPTIONAL_ACTOR_DIM


def _fill_actor_onehot(target: torch.Tensor, offset: int, actor: int) -> int:
    target[offset + actor] = 1.0
    return offset + ACTOR_DIM


def _fill_own_meld_counts(target: torch.Tensor, offset: int, melds: list[dict]) -> None:
    kinds = ["chi", "pon", "daiminkan", "ankan", "kakan"]
    counts = {kind: 0.0 for kind in kinds}
    for meld in melds:
        kind = meld.get("kind")
        if kind in counts:
            counts[kind] += 1.0
    for index, kind in enumerate(kinds):
        target[offset + index] = counts[kind]


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
    label_index = legal_action_keys.index(label_key) if label_key in legal_action_keys else None

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
        "label_in_legal": label_index is not None,
        "label_index": label_index,
        "label_action": label_action,
        "legal_actions": legal_actions,
        "state": state,
    }


def _deserialize_dataset_row(row: dict[str, object]) -> dict[str, object]:
    legal_actions = json.loads(row["legal_actions_json"])
    label_action = json.loads(row["label_action_json"])
    label_key = _action_key(label_action)
    legal_action_keys = [_action_key(action) for action in legal_actions]
    try:
        label_index = legal_action_keys.index(label_key)
    except ValueError as exc:
        raise ValueError(f"label action missing from legal actions: {row['sample_id']}") from exc

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
        "label_index": label_index,
        "label_kind": str(label_action["kind"]),
    }
