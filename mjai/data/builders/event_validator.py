from __future__ import annotations

from collections import Counter
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mjai.data.datasets.event_store import read_events_from_parquet
from mjai.rules.validators import ReplayValidator


@dataclass(slots=True, frozen=True)
class ValidationSummary:
    files_seen: int
    files_valid: int
    files_failed: int
    total_events: int
    total_hands: int
    total_discards: int
    total_melds: int
    event_type_counts: dict[str, int]
    error_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


class EventValidator:
    """Validate persisted bronze event shards via replay."""

    def __init__(self) -> None:
        self.validator = ReplayValidator()

    def validate_dir(self, input_dir: str | Path, report_path: str | Path) -> ValidationSummary:
        input_root = Path(input_dir)
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        files_seen = 0
        files_valid = 0
        files_failed = 0
        total_events = 0
        total_hands = 0
        total_discards = 0
        total_melds = 0
        event_type_counts: Counter[str] = Counter()
        error_counts: Counter[str] = Counter()
        file_reports: list[dict[str, object]] = []

        for shard_path in sorted(input_root.glob("*.parquet")):
            files_seen += 1
            events = read_events_from_parquet(shard_path)
            total_events += len(events)
            event_type_counts.update(event.event_type for event in events)

            try:
                summary = self.validator.validate_events(events)
            except Exception as exc:
                files_failed += 1
                error_key = _classify_error_message(str(exc))
                error_counts[error_key] += 1
                file_reports.append(
                    {
                        "game_id": shard_path.stem,
                        "path": str(shard_path.as_posix()),
                        "ok": False,
                        "error": str(exc),
                        "error_kind": error_key,
                    }
                )
                continue

            files_valid += 1
            total_hands += summary.hands_seen
            total_discards += summary.total_discards
            total_melds += summary.total_melds
            file_reports.append(
                {
                    "game_id": shard_path.stem,
                    "path": str(shard_path.as_posix()),
                    "ok": True,
                    "summary": asdict(summary),
                }
            )

        warnings: list[str] = []
        if files_seen == 0:
            warnings.append("no event shards found in input directory")
        if files_failed > 0:
            warnings.append(f"{files_failed} event shards failed validation")

        overall = ValidationSummary(
            files_seen=files_seen,
            files_valid=files_valid,
            files_failed=files_failed,
            total_events=total_events,
            total_hands=total_hands,
            total_discards=total_discards,
            total_melds=total_melds,
            event_type_counts=dict(sorted(event_type_counts.items())),
            error_counts=dict(sorted(error_counts.items())),
            warnings=tuple(warnings),
        )
        report_file.write_text(
            json.dumps(
                {
                    "summary": asdict(overall),
                    "files": file_reports,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return overall


def _classify_error_message(message: str) -> str:
    prefix, _, _ = message.partition(":")
    return prefix.strip() or "unknown"
