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
    """将事件对象序列化为列式存储行。
    
    参数:
        event (Event): 单个规范化事件。
    
    返回:
        dict[str, object]: 字典结果。
    
    异常:
        无。
    """
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
    """将列式存储行反序列化为事件对象。
    
    参数:
        row (dict[str, object]): 单条数据行。
    
    返回:
        Event: 函数执行结果。
    
    异常:
        无。
    """
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
    """将事件序列写入列式文件。
    
    参数:
        events (Iterable[Event]): 规范化事件列表。
        output_path (str | Path): 输出文件路径。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    rows = [serialize_event(event) for event in events]
    table = pa.Table.from_pylist(rows, schema=EVENT_SCHEMA)
    output_file = Path(output_path)
    ensure_dir(output_file.parent)
    pq.write_table(table, output_file, compression="zstd")
    return table.num_rows


def read_events_from_parquet(path: str | Path) -> list[Event]:
    """从列式文件读取事件序列。
    
    参数:
        path (str | Path): 文件或目录路径。
    
    返回:
        list[Event]: 列表结果。
    
    异常:
        无。
    """
    table = pq.read_table(path)
    return [deserialize_event(row) for row in table.to_pylist()]
