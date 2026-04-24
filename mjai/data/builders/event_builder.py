from __future__ import annotations

from collections import Counter
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mjai.data.datasets.event_store import write_events_to_parquet
from mjai.data.tenhou.parser import TenhouParser
from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class PreprocessSummary:
    games: int
    events: int
    event_type_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


class EventBuilder:
    """Build persisted bronze event shards from Tenhou XML files."""

    def __init__(self) -> None:
        self.parser = TenhouParser()

    def build_from_xml_dir(self, input_dir: str | Path, output_dir: str | Path) -> PreprocessSummary:
        input_root = Path(input_dir)
        output_root = ensure_dir(output_dir)

        games = 0
        events_written = 0
        event_type_counts: Counter[str] = Counter()

        for xml_path in sorted(input_root.glob("*.xml")):
            events = self.parser.parse_file(xml_path)
            event_count = write_events_to_parquet(events, output_root / f"{xml_path.stem}.parquet")
            games += 1
            events_written += event_count
            event_type_counts.update(event.event_type for event in events)

        warnings: list[str] = []
        if games == 0:
            warnings.append("no xml files found in input directory")

        summary = PreprocessSummary(
            games=games,
            events=events_written,
            event_type_counts=dict(sorted(event_type_counts.items())),
            warnings=tuple(warnings),
        )
        (output_root / "preprocess_summary.json").write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
