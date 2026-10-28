from __future__ import annotations

from collections import Counter
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mjai.data.datasets.event_store import read_events_from_parquet
from mjai.rules.validators import ReplayValidator
from mjai.utils.process_workers import process_map


@dataclass(slots=True, frozen=True)
class ValidationSummary:
    files_seen: int
    files_valid: int
    files_failed: int
    total_events: int
    total_kyokus: int
    total_discards: int
    total_melds: int
    event_type_counts: dict[str, int]
    error_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


class EventValidator:
    """Validate persisted bronze event shards via replay."""

    def __init__(self) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.validator = ReplayValidator()

    def validate_dir(
        self, input_dir: str | Path, report_path: str | Path, workers: int = 1,
    ) -> ValidationSummary:
        """校验输入数据的一致性和合法性。
        
        参数:
            self (Self): 当前对象实例。
            input_dir (str | Path): 输入目录路径。
            report_path (str | Path): 报告输出路径。
        
        返回:
            ValidationSummary: 函数执行结果。
        
        异常:
            无。
        """
        input_root = Path(input_dir)
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        files_seen = 0
        files_valid = 0
        files_failed = 0
        total_events = 0
        total_kyokus = 0
        total_discards = 0
        total_melds = 0
        event_type_counts: Counter[str] = Counter()
        error_counts: Counter[str] = Counter()
        file_reports: list[dict[str, object]] = []

        paths = sorted(input_root.glob("*.parquet"))
        for count, counts, report in process_map(
            _validate_one_shard, paths, min(workers, len(paths)),
        ):
            files_seen += 1
            total_events += count
            event_type_counts.update(counts)
            file_reports.append(report)
            if not report["ok"]:
                files_failed += 1
                error_counts[report["error_kind"]] += 1
                continue
            files_valid += 1
            summary = report["summary"]
            total_kyokus += summary["kyokus_seen"]
            total_discards += summary["total_discards"]
            total_melds += summary["total_melds"]

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
            total_kyokus=total_kyokus,
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
    """执行与函数名对应的业务逻辑。
    
    参数:
        message (str): 错误消息文本。
    
    返回:
        str: 字符串结果。
    
    异常:
        无。
    """
    prefix, _, _ = message.partition(":")
    return prefix.strip() or "unknown"


def _validate_one_shard(shard_path: Path) -> tuple[int, dict[str, int], dict]:
    # Preserve read-error behavior: unreadable shards abort the command.
    events = read_events_from_parquet(shard_path)
    counts = dict(Counter(event.event_type for event in events))
    report = {"game_id": shard_path.stem, "path": shard_path.as_posix()}
    try:
        summary = ReplayValidator().validate_events(events)
    except Exception as exc:
        report.update(ok=False, error=str(exc), error_kind=_classify_error_message(str(exc)))
    else:
        report.update(ok=True, summary=asdict(summary))
    return len(events), counts, report
