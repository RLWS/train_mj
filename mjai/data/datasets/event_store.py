from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from mjai.data.schema.events import Event
from mjai.utils.paths import ensure_dir


EVENT_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("kyoku_index", pa.int32()),
        ("step_index", pa.int32()),
        ("event_type", pa.string()),
        ("actor", pa.int8()),
        ("raw_tag", pa.string()),
        ("payload_json", pa.string()),
    ]
)


def serialize_event(event: Event) -> dict[str, object]:
    return {
        "game_id": event.game_id,
        "kyoku_index": event.kyoku_index,
        "step_index": event.step_index,
        "event_type": event.event_type,
        "actor": event.actor,
        "raw_tag": event.raw_tag,
        "payload_json": json.dumps(event.payload, ensure_ascii=False, sort_keys=True),
    }


def deserialize_event(row: dict[str, object]) -> Event:
    return Event(
        game_id=str(row["game_id"]),
        kyoku_index=int(row["kyoku_index"]),
        step_index=int(row["step_index"]),
        event_type=str(row["event_type"]),
        actor=None if row["actor"] is None else int(row["actor"]),
        raw_tag=None if row["raw_tag"] is None else str(row["raw_tag"]),
        payload=json.loads(str(row["payload_json"])),
    )


def write_events_to_parquet(events: Iterable[Event], output_path: str | Path) -> int:
    rows = [serialize_event(event) for event in events]
    table = pa.Table.from_pylist(rows, schema=EVENT_SCHEMA)
    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    pq.write_table(table, output_file, compression="zstd")
    return table.num_rows


def read_events_from_parquet(path: str | Path) -> list[Event]:
    table = pq.read_table(path)
    return [deserialize_event(row) for row in table.to_pylist()]
