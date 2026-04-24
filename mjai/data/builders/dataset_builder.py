from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mjai.data.builders.sample_builder import ResponseSampleBuilder, TurnSampleBuilder
from mjai.data.datasets.parquet_dataset import write_samples_to_parquet
from mjai.data.tenhou.parser import TenhouParser
from mjai.rules.validators import ReplayValidator
from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class BuildSummary:
    games: int
    turn_samples: int
    response_samples: int


class DatasetBuilder:
    """Build parquet-backed supervised datasets directly from Tenhou XML fixtures."""

    def __init__(self) -> None:
        self.parser = TenhouParser()
        self.validator = ReplayValidator()
        self.turn_builder = TurnSampleBuilder()
        self.response_builder = ResponseSampleBuilder()

    def build_from_xml_dir(self, input_dir: str | Path, output_dir: str | Path) -> BuildSummary:
        input_root = Path(input_dir)
        output_root = Path(output_dir)
        ensure_dir(output_root / "turn")
        ensure_dir(output_root / "response")

        turn_samples = 0
        response_samples = 0
        games = 0

        for xml_path in sorted(input_root.glob("*.xml")):
            events = self.parser.parse_file(xml_path)
            self.validator.validate_events(events)

            turn_rows = self.turn_builder.build_samples(events)
            response_rows = self.response_builder.build_samples(events)

            turn_count = write_samples_to_parquet(
                turn_rows,
                output_root / "turn" / f"{xml_path.stem}.parquet",
            )
            response_count = write_samples_to_parquet(
                response_rows,
                output_root / "response" / f"{xml_path.stem}.parquet",
            )

            turn_samples += turn_count
            response_samples += response_count
            games += 1

        summary = BuildSummary(games=games, turn_samples=turn_samples, response_samples=response_samples)
        summary_path = output_root / "build_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "games": summary.games,
                    "turn_samples": summary.turn_samples,
                    "response_samples": summary.response_samples,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return summary
