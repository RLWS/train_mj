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
    hand_index: int
    step_index: int
    source_path: str
    target_event_index: int
    target_event: dict[str, Any]
    hand_events: list[dict[str, Any]]
    context_before: list[dict[str, Any]]
    context_after: list[dict[str, Any]]


def dump_decision_case(
    bronze_dir: str | Path,
    game_id: str,
    hand_index: int,
    step_index: int,
    context_radius: int = 5,
) -> DumpedCase:
    bronze_root = Path(bronze_dir)
    shard_path = bronze_root / f"{game_id}.parquet"
    if not shard_path.exists():
        raise FileNotFoundError(f"bronze event shard not found for game_id '{game_id}': {shard_path}")

    events = read_events_from_parquet(shard_path)
    hand_events = [event for event in events if event.hand_index == hand_index]
    if not hand_events:
        raise ValueError(f"hand_index not found in event shard: game_id={game_id} hand_index={hand_index}")

    target_event_index = next(
        (index for index, event in enumerate(hand_events) if event.step_index == step_index),
        None,
    )
    if target_event_index is None:
        raise ValueError(
            f"step_index not found in event shard: game_id={game_id} hand_index={hand_index} step_index={step_index}"
        )

    start = max(0, target_event_index - context_radius)
    end = min(len(hand_events), target_event_index + context_radius + 1)

    return DumpedCase(
        game_id=game_id,
        hand_index=hand_index,
        step_index=step_index,
        source_path=str(shard_path.as_posix()),
        target_event_index=target_event_index,
        target_event=_serialize_event(hand_events[target_event_index]),
        hand_events=[_serialize_event(event) for event in hand_events],
        context_before=[_serialize_event(event) for event in hand_events[start:target_event_index]],
        context_after=[_serialize_event(event) for event in hand_events[target_event_index + 1 : end]],
    )


def dump_case_to_json(
    bronze_dir: str | Path,
    game_id: str,
    hand_index: int,
    step_index: int,
    context_radius: int = 5,
) -> str:
    dumped_case = dump_decision_case(
        bronze_dir=bronze_dir,
        game_id=game_id,
        hand_index=hand_index,
        step_index=step_index,
        context_radius=context_radius,
    )
    return json.dumps(asdict(dumped_case), ensure_ascii=False, indent=2)


def _serialize_event(event: Event) -> dict[str, Any]:
    return {
        "game_id": event.game_id,
        "hand_index": event.hand_index,
        "step_index": event.step_index,
        "event_type": event.event_type,
        "actor": event.actor,
        "raw_tag": event.raw_tag,
        "payload": event.payload,
    }
