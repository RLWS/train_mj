from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mjai.data.datasets.event_store import read_events_from_parquet
from mjai.data.schema.events import Event


@dataclass(slots=True, frozen=True)
class DumpedCase:
    game_id: str
    kyoku_index: int
    step_index: int
    source_path: str
    target_event_index: int
    target_event: dict[str, Any]
    kyoku_events: list[dict[str, Any]]
    context_before: list[dict[str, Any]]
    context_after: list[dict[str, Any]]


def dump_decision_case(
    bronze_dir: str | Path,
    game_id: str,
    kyoku_index: int,
    step_index: int,
    context_radius: int = 5,
) -> DumpedCase:
    """导出指定决策点及其同局上下文。
    
    参数:
        bronze_dir (str | Path): 青铜层事件目录路径。
        game_id (str): 函数执行所需的输入参数。
        kyoku_index (int): 相关索引值。
        step_index (int): 相关索引值。
        context_radius (int): 相关数据列表。
    
    返回:
        DumpedCase: 函数执行结果。
    
    异常:
        FileNotFoundError: 当所需文件或目录不存在时抛出。
        ValueError: 当输入数据不满足函数约束时抛出。
    """
    bronze_root = Path(bronze_dir)
    shard_path = bronze_root / f"{game_id}.parquet"
    if not shard_path.exists():
        raise FileNotFoundError(f"bronze event shard not found for game_id '{game_id}': {shard_path}")

    events = read_events_from_parquet(shard_path)
    kyoku_events = [event for event in events if event.kyoku_index == kyoku_index]
    if not kyoku_events:
        raise ValueError(f"kyoku_index not found in event shard: game_id={game_id} kyoku_index={kyoku_index}")

    target_event_index = next(
        (index for index, event in enumerate(kyoku_events) if event.step_index == step_index),
        None,
    )
    if target_event_index is None:
        raise ValueError(
            f"step_index not found in event shard: game_id={game_id} kyoku_index={kyoku_index} step_index={step_index}"
        )

    start = max(0, target_event_index - context_radius)
    end = min(len(kyoku_events), target_event_index + context_radius + 1)

    return DumpedCase(
        game_id=game_id,
        kyoku_index=kyoku_index,
        step_index=step_index,
        source_path=str(shard_path.as_posix()),
        target_event_index=target_event_index,
        target_event=_serialize_event(kyoku_events[target_event_index]),
        kyoku_events=[_serialize_event(event) for event in kyoku_events],
        context_before=[_serialize_event(event) for event in kyoku_events[start:target_event_index]],
        context_after=[_serialize_event(event) for event in kyoku_events[target_event_index + 1 : end]],
    )


def dump_case_to_json(
    bronze_dir: str | Path,
    game_id: str,
    kyoku_index: int,
    step_index: int,
    context_radius: int = 5,
) -> str:
    """导出指定决策案例并编码为 JSON 字符串。
    
    参数:
        bronze_dir (str | Path): 青铜层事件目录路径。
        game_id (str): 函数执行所需的输入参数。
        kyoku_index (int): 相关索引值。
        step_index (int): 相关索引值。
        context_radius (int): 相关数据列表。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    dumped_case = dump_decision_case(
        bronze_dir=bronze_dir,
        game_id=game_id,
        kyoku_index=kyoku_index,
        step_index=step_index,
        context_radius=context_radius,
    )
    return json.dumps(asdict(dumped_case), ensure_ascii=False, indent=2)


def _serialize_event(event: Event) -> dict[str, Any]:
    """序列化输入对象。
    
    参数:
        event (Event): 单个规范化事件。
    
    返回:
        dict[str, Any]: 字典结果。
    
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
        "payload": event.payload,
    }
