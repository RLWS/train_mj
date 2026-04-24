from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from mjai.data.schema.samples import TurnSample
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
