from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mjai.data.datasets.event_store import write_events_to_parquet
from mjai.data.tenhou.parser import TenhouParser
from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class PreprocessSummary:
    games: int
    events: int


class EventBuilder:
    """Build persisted bronze event shards from Tenhou XML files."""

    def __init__(self) -> None:
        self.parser = TenhouParser()

    def build_from_xml_dir(self, input_dir: str | Path, output_dir: str | Path) -> PreprocessSummary:
        input_root = Path(input_dir)
        output_root = ensure_dir(output_dir)

        games = 0
        events_written = 0

        for xml_path in sorted(input_root.glob("*.xml")):
            events = self.parser.parse_file(xml_path)
            event_count = write_events_to_parquet(events, output_root / f"{xml_path.stem}.parquet")
            games += 1
            events_written += event_count

        summary = PreprocessSummary(games=games, events=events_written)
        (output_root / "preprocess_summary.json").write_text(
            json.dumps(
                {
                    "games": summary.games,
                    "events": summary.events,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return summary
