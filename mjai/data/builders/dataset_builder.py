from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mjai.data.datasets.event_store import read_events_from_parquet
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
    turn_label_counts: dict[str, int]
    response_label_counts: dict[str, int]
    games_with_zero_turn_samples: int
    games_with_zero_response_samples: int
    warnings: tuple[str, ...] = ()


class DatasetBuilder:
    """Build parquet-backed supervised datasets directly from Tenhou XML fixtures."""

    def __init__(self) -> None:
        self.parser = TenhouParser()
        self.validator = ReplayValidator()
        self.turn_builder = TurnSampleBuilder()
        self.response_builder = ResponseSampleBuilder()

    def build_from_xml_dir(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        workers: int = 1,
    ) -> BuildSummary:
        input_root = Path(input_dir)
        return self._build_from_sources(
            output_dir=output_dir,
            source_paths=sorted(input_root.glob("*.xml")),
            build_func=_build_one_xml_file,
            workers=workers,
            empty_warning="no xml files found in input directory",
        )

    def build_from_event_dir(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        workers: int = 1,
    ) -> BuildSummary:
        input_root = Path(input_dir)
        return self._build_from_sources(
            output_dir=output_dir,
            source_paths=sorted(input_root.glob("*.parquet")),
            build_func=_build_one_event_shard,
            workers=workers,
            empty_warning="no event shards found in input directory",
        )

    def _build_from_sources(
        self,
        output_dir: str | Path,
        source_paths: list[Path],
        build_func,
        workers: int,
        empty_warning: str,
    ) -> BuildSummary:
        output_root = Path(output_dir)
        ensure_dir(output_root / "turn")
        ensure_dir(output_root / "response")

        turn_samples = 0
        response_samples = 0
        games = 0
        turn_label_counts: Counter[str] = Counter()
        response_label_counts: Counter[str] = Counter()
        games_with_zero_turn_samples = 0
        games_with_zero_response_samples = 0

        results = _run_build_workers(
            source_paths=source_paths,
            output_root=output_root,
            build_func=build_func,
            workers=workers,
        )

        for result in results:
            turn_label_counts.update(result.turn_label_counts)
            response_label_counts.update(result.response_label_counts)
            turn_samples += result.turn_samples
            response_samples += result.response_samples
            games += 1
            if result.turn_samples == 0:
                games_with_zero_turn_samples += 1
            if result.response_samples == 0:
                games_with_zero_response_samples += 1

        warnings: list[str] = []
        if games == 0:
            warnings.append(empty_warning)
        if games_with_zero_turn_samples > 0:
            warnings.append(f"{games_with_zero_turn_samples} games produced 0 turn samples")
        if games_with_zero_response_samples > 0:
            warnings.append(f"{games_with_zero_response_samples} games produced 0 response samples")

        summary = BuildSummary(
            games=games,
            turn_samples=turn_samples,
            response_samples=response_samples,
            turn_label_counts=dict(sorted(turn_label_counts.items())),
            response_label_counts=dict(sorted(response_label_counts.items())),
            games_with_zero_turn_samples=games_with_zero_turn_samples,
            games_with_zero_response_samples=games_with_zero_response_samples,
            warnings=tuple(warnings),
        )
        summary_path = output_root / "build_summary.json"
        summary_path.write_text(
            json.dumps(asdict(summary), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary


@dataclass(slots=True, frozen=True)
class _ShardBuildResult:
    game_id: str
    turn_samples: int
    response_samples: int
    turn_label_counts: dict[str, int]
    response_label_counts: dict[str, int]


def _run_build_workers(
    source_paths: list[Path],
    output_root: Path,
    build_func,
    workers: int,
) -> list[_ShardBuildResult]:
    if workers <= 1:
        return [build_func(str(path), str(output_root)) for path in source_paths]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            executor.map(
                build_func,
                [str(path) for path in source_paths],
                [str(output_root)] * len(source_paths),
            )
        )


def _build_one_xml_file(xml_path: str, output_root: str) -> _ShardBuildResult:
    parser = TenhouParser()
    path = Path(xml_path)
    events = parser.parse_file(path)
    return _build_one_game(path.stem, events, Path(output_root))


def _build_one_event_shard(shard_path: str, output_root: str) -> _ShardBuildResult:
    path = Path(shard_path)
    events = read_events_from_parquet(path)
    return _build_one_game(path.stem, events, Path(output_root))


def _build_one_game(game_id: str, events: list, output_root: Path) -> _ShardBuildResult:
    ReplayValidator().validate_events(events)

    turn_rows = TurnSampleBuilder().build_samples(events)
    response_rows = ResponseSampleBuilder().build_samples(events)
    turn_label_counts = Counter(str(sample.label_action["kind"]) for sample in turn_rows)
    response_label_counts = Counter(str(sample.label_action["kind"]) for sample in response_rows)

    turn_count = write_samples_to_parquet(
        turn_rows,
        output_root / "turn" / f"{game_id}.parquet",
    )
    response_count = write_samples_to_parquet(
        response_rows,
        output_root / "response" / f"{game_id}.parquet",
    )

    return _ShardBuildResult(
        game_id=game_id,
        turn_samples=turn_count,
        response_samples=response_count,
        turn_label_counts=dict(turn_label_counts),
        response_label_counts=dict(response_label_counts),
    )
