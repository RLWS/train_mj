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
from mjai.features.action_features import TURN_ACTION_DIM, encode_turn_action
from mjai.features.state_features import TURN_STATE_DIM, encode_turn_state
from mjai.utils.paths import ensure_dir


SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("game_id", pa.string()),
        ("hand_index", pa.int32()),
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
        "hand_index": sample.hand_index,
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
        row = self.rows[index]
        return {
            "sample_id": row["sample_id"],
            "state": json.loads(row["state_json"]),
            "legal_actions": json.loads(row["legal_actions_json"]),
            "label_action": json.loads(row["label_action_json"]),
        }


def build_turn_batch(rows: list[dict], candidate_limit: int = 64) -> dict[str, torch.Tensor | list[str]]:
    max_candidates = max(len(row["legal_actions"]) for row in rows)
    if max_candidates > candidate_limit:
        raise ValueError(f"candidate count exceeds configured limit: {max_candidates} > {candidate_limit}")

    batch_size = len(rows)
    state_feats = torch.zeros((batch_size, TURN_STATE_DIM), dtype=torch.float32)
    candidate_feats = torch.zeros((batch_size, max_candidates, TURN_ACTION_DIM), dtype=torch.float32)
    candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    sample_ids: list[str] = []

    for batch_index, row in enumerate(rows):
        state_feats[batch_index] = torch.tensor(encode_turn_state(row["state"]), dtype=torch.float32)
        label_key = _action_key(row["label_action"])
        legal_action_keys = [_action_key(action) for action in row["legal_actions"]]
        try:
            labels[batch_index] = legal_action_keys.index(label_key)
        except ValueError as exc:
            raise ValueError(f"label action missing from legal actions: {row['sample_id']}") from exc

        for candidate_index, action in enumerate(row["legal_actions"]):
            candidate_feats[batch_index, candidate_index] = torch.tensor(
                encode_turn_action(action),
                dtype=torch.float32,
            )
            candidate_mask[batch_index, candidate_index] = True
        sample_ids.append(str(row["sample_id"]))

    return {
        "state_feats": state_feats,
        "candidate_feats": candidate_feats,
        "candidate_mask": candidate_mask,
        "labels": labels,
        "sample_ids": sample_ids,
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
        rows.extend(pq.read_table(turn_path).to_pylist())
    return rows


def _action_key(action: dict) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True)
