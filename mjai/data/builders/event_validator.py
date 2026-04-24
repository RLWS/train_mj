from __future__ import annotations

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
        file_reports: list[dict[str, object]] = []

        for shard_path in sorted(input_root.glob("*.parquet")):
            files_seen += 1
            events = read_events_from_parquet(shard_path)
            total_events += len(events)

            try:
                summary = self.validator.validate_events(events)
            except Exception as exc:
                files_failed += 1
                file_reports.append(
                    {
                        "game_id": shard_path.stem,
                        "path": str(shard_path.as_posix()),
                        "ok": False,
                        "error": str(exc),
                    }
                )
                continue

            files_valid += 1
            file_reports.append(
                {
                    "game_id": shard_path.stem,
                    "path": str(shard_path.as_posix()),
                    "ok": True,
                    "summary": asdict(summary),
                }
            )

        overall = ValidationSummary(
            files_seen=files_seen,
            files_valid=files_valid,
            files_failed=files_failed,
            total_events=total_events,
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
