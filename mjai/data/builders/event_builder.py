from __future__ import annotations

from collections import Counter
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mjai.data.datasets.event_store import write_events_to_parquet
from mjai.data.tenhou.parser import TenhouParser
from mjai.utils.paths import ensure_dir
from mjai.utils.process_workers import process_map


@dataclass(slots=True, frozen=True)
class PreprocessSummary:
    games: int
    events: int
    event_type_counts: dict[str, int]
    warnings: tuple[str, ...] = ()


class EventBuilder:
    """Build persisted bronze event shards from Tenhou XML files."""

    def __init__(self) -> None:
        """执行与函数名对应的业务逻辑。
        
        参数:
            self (Self): 当前对象实例。
        
        返回:
            None: 无返回值。
        
        异常:
            无。
        """
        self.parser = TenhouParser()

    def build_from_xml_dir(
        self, input_dir: str | Path, output_dir: str | Path, workers: int = 1,
    ) -> PreprocessSummary:
        """构建与函数名对应的数据结构或处理结果。
        
        参数:
            self (Self): 当前对象实例。
            input_dir (str | Path): 输入目录路径。
            output_dir (str | Path): 输出目录路径。
        
        返回:
            PreprocessSummary: 函数执行结果。
        
        异常:
            无。
        """
        input_root = Path(input_dir)
        output_root = ensure_dir(output_dir)

        games = 0
        events_written = 0
        event_type_counts: Counter[str] = Counter()

        paths = sorted(input_root.glob("*.xml"))
        jobs = ((path, output_root) for path in paths)
        for event_count, counts in process_map(
            _preprocess_one_file, jobs, min(workers, len(paths)),
        ):
            games += 1
            events_written += event_count
            event_type_counts.update(counts)

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


def _preprocess_one_file(job: tuple[Path, Path]) -> tuple[int, dict[str, int]]:
    xml_path, output_root = job
    events = TenhouParser().parse_file(xml_path)
    count = write_events_to_parquet(events, output_root / f"{xml_path.stem}.parquet")
    return count, dict(Counter(event.event_type for event in events))
