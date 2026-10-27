from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from mjai.data.datasets.event_store import read_events_from_parquet
from mjai.data.builders.sample_builder import ResponseSampleBuilder, TurnSampleBuilder
from mjai.data.datasets.sample_store import write_samples_to_parquet
from mjai.data.tenhou.parser import TenhouParser
from mjai.rules.validators import ReplayValidator
from mjai.utils.paths import ensure_dir


@dataclass(slots=True, frozen=True)
class BuildSummary:
    games: int
    turn_samples: int
    response_samples: int
    elapsed_seconds: float
    games_per_second: float
    samples_per_second: float
    turn_label_counts: dict[str, int]
    response_label_counts: dict[str, int]
    games_with_zero_turn_samples: int
    games_with_zero_response_samples: int
    warnings: tuple[str, ...] = ()


class DatasetBuilder:
    """Build parquet-backed supervised datasets directly from Tenhou XML fixtures."""

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
        self.validator = ReplayValidator()
        self.turn_builder = TurnSampleBuilder()
        self.response_builder = ResponseSampleBuilder()

    def build_from_xml_dir(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        workers: int = 1,
        validate: bool = True,
    ) -> BuildSummary:
        """构建与函数名对应的数据结构或处理结果。
        
        参数:
            self (Self): 当前对象实例。
            input_dir (str | Path): 输入目录路径。
            output_dir (str | Path): 输出目录路径。
            workers (int): 并行工作单元数量。
            validate (bool): 是否执行回放校验。
        
        返回:
            BuildSummary: 函数执行结果。
        
        异常:
            无。
        """
        input_root = Path(input_dir)
        return self._build_from_sources(
            output_dir=output_dir,
            source_paths=sorted(input_root.glob("*.xml")),
            build_func=_build_one_xml_file,
            workers=workers,
            validate=validate,
            empty_warning="no xml files found in input directory",
        )

    def build_from_event_dir(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        workers: int = 1,
        validate: bool = True,
    ) -> BuildSummary:
        """构建与函数名对应的数据结构或处理结果。
        
        参数:
            self (Self): 当前对象实例。
            input_dir (str | Path): 输入目录路径。
            output_dir (str | Path): 输出目录路径。
            workers (int): 并行工作单元数量。
            validate (bool): 是否执行回放校验。
        
        返回:
            BuildSummary: 函数执行结果。
        
        异常:
            无。
        """
        input_root = Path(input_dir)
        return self._build_from_sources(
            output_dir=output_dir,
            source_paths=sorted(input_root.glob("*.parquet")),
            build_func=_build_one_event_shard,
            workers=workers,
            validate=validate,
            empty_warning="no event shards found in input directory",
        )

    def _build_from_sources(
        self,
        output_dir: str | Path,
        source_paths: list[Path],
        build_func,
        workers: int,
        validate: bool,
        empty_warning: str,
    ) -> BuildSummary:
        """构建与函数名对应的数据结构或处理结果。
        
        参数:
            self (Self): 当前对象实例。
            output_dir (str | Path): 输出目录路径。
            source_paths (list[Path]): 相关数据列表。
            build_func (Any): 函数执行所需的输入参数。
            workers (int): 并行工作单元数量。
            validate (bool): 是否执行回放校验。
            empty_warning (str): 函数执行所需的输入参数。
        
        返回:
            BuildSummary: 函数执行结果。
        
        异常:
            无。
        """
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

        started_at = perf_counter()
        results = _run_build_workers(
            source_paths=source_paths,
            output_root=output_root,
            build_func=build_func,
            workers=workers,
            validate=validate,
        )
        elapsed_seconds = perf_counter() - started_at

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
        if not validate:
            warnings.append("replay validation skipped during dataset build")
        if games_with_zero_turn_samples > 0:
            warnings.append(f"{games_with_zero_turn_samples} games produced 0 turn samples")
        if games_with_zero_response_samples > 0:
            warnings.append(f"{games_with_zero_response_samples} games produced 0 response samples")

        summary = BuildSummary(
            games=games,
            turn_samples=turn_samples,
            response_samples=response_samples,
            elapsed_seconds=elapsed_seconds,
            games_per_second=games / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            samples_per_second=(
                (turn_samples + response_samples) / elapsed_seconds if elapsed_seconds > 0 else 0.0
            ),
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
    validate: bool,
) -> list[_ShardBuildResult]:
    """运行与函数名对应的处理流程。
    
    参数:
        source_paths (list[Path]): 相关数据列表。
        output_root (Path): 相关目录路径。
        build_func (Any): 函数执行所需的输入参数。
        workers (int): 并行工作单元数量。
        validate (bool): 是否执行回放校验。
    
    返回:
        list[_ShardBuildResult]: 列表结果。
    
    异常:
        Exception: 当函数执行过程中检测到未分类错误时抛出。
    """
    if not source_paths:
        return []

    worker_count = min(max(1, workers), len(source_paths))
    output_root_text = str(output_root)
    if worker_count <= 1:
        return [build_func(str(path), output_root_text, validate) for path in source_paths]

    chunksize = _build_worker_chunksize(source_count=len(source_paths), workers=worker_count)
    source_path_texts = [str(path) for path in source_paths]
    output_root_texts = [output_root_text] * len(source_paths)
    validate_flags = [validate] * len(source_paths)
    try:
        return _run_process_build_workers(
            build_func=build_func,
            source_paths=source_path_texts,
            output_roots=output_root_texts,
            validate_flags=validate_flags,
            workers=worker_count,
            chunksize=chunksize,
        )
    except (BrokenProcessPool, PermissionError) as exc:
        if not _should_fallback_to_threads(exc):
            raise
        return _run_thread_build_workers(
            build_func=build_func,
            source_paths=source_path_texts,
            output_roots=output_root_texts,
            validate_flags=validate_flags,
            workers=worker_count,
        )


def _run_process_build_workers(
    build_func,
    source_paths: list[str],
    output_roots: list[str],
    validate_flags: list[bool],
    workers: int,
    chunksize: int,
) -> list[_ShardBuildResult]:
    """运行与函数名对应的处理流程。
    
    参数:
        build_func (Any): 函数执行所需的输入参数。
        source_paths (list[str]): 相关数据列表。
        output_roots (list[str]): 相关数据列表。
        validate_flags (list[bool]): 相关数据列表。
        workers (int): 并行工作单元数量。
        chunksize (int): 函数执行所需的输入参数。
    
    返回:
        list[_ShardBuildResult]: 列表结果。
    
    异常:
        无。
    """
    max_pending = _build_max_pending_chunks(workers)
    context = _build_multiprocessing_context()
    pending: set[Future[list[_ShardBuildResult]]] = set()
    results: list[_ShardBuildResult] = []

    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for start in range(0, len(source_paths), chunksize):
            end = min(start + chunksize, len(source_paths))
            pending.add(
                executor.submit(
                    _build_source_chunk,
                    build_func,
                    source_paths[start:end],
                    output_roots[start:end],
                    validate_flags[start:end],
                )
            )
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                results.extend(_collect_chunk_results(done))

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            results.extend(_collect_chunk_results(done))

    return results


def _build_source_chunk(
    build_func,
    source_paths: list[str],
    output_roots: list[str],
    validate_flags: list[bool],
) -> list[_ShardBuildResult]:
    return [
        build_func(source_path, output_root, validate)
        for source_path, output_root, validate in zip(source_paths, output_roots, validate_flags)
    ]


def _collect_chunk_results(
    futures: set[Future[list[_ShardBuildResult]]],
) -> list[_ShardBuildResult]:
    results: list[_ShardBuildResult] = []
    for future in futures:
        results.extend(future.result())
    return results


def _build_multiprocessing_context() -> mp.context.BaseContext:
    requested = os.environ.get("MJAI_BUILD_MP_START_METHOD", "spawn").strip()
    available = set(mp.get_all_start_methods())
    if requested not in available:
        raise ValueError(
            f"unsupported MJAI_BUILD_MP_START_METHOD={requested!r}; "
            f"available methods: {sorted(available)}"
        )
    return mp.get_context(requested)


def _build_max_pending_chunks(workers: int) -> int:
    configured = os.environ.get("MJAI_BUILD_MAX_PENDING_CHUNKS")
    if configured:
        try:
            return max(1, int(configured))
        except ValueError as exc:
            raise ValueError("MJAI_BUILD_MAX_PENDING_CHUNKS must be an integer") from exc
    return max(1, workers * 2)


def _run_thread_build_workers(
    build_func,
    source_paths: list[str],
    output_roots: list[str],
    validate_flags: list[bool],
    workers: int,
) -> list[_ShardBuildResult]:
    """运行与函数名对应的处理流程。
    
    参数:
        build_func (Any): 函数执行所需的输入参数。
        source_paths (list[str]): 相关数据列表。
        output_roots (list[str]): 相关数据列表。
        validate_flags (list[bool]): 相关数据列表。
        workers (int): 并行工作单元数量。
    
    返回:
        list[_ShardBuildResult]: 列表结果。
    
    异常:
        无。
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(build_func, source_paths, output_roots, validate_flags))


def _should_fallback_to_threads(exc: BaseException) -> bool:
    """执行与函数名对应的业务逻辑。
    
    参数:
        exc (BaseException): 函数执行所需的输入参数。
    
    返回:
        bool: 判断结果。
    
    异常:
        无。
    """
    if isinstance(exc, BrokenProcessPool):
        return True
    return isinstance(exc, PermissionError) and getattr(exc, "winerror", None) == 5


def _build_worker_chunksize(source_count: int, workers: int) -> int:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        source_count (int): 相关数量。
        workers (int): 并行工作单元数量。
    
    返回:
        int: 整数结果。
    
    异常:
        无。
    """
    if source_count <= workers * 16:
        return 1
    return max(1, min(8, source_count // (workers * 16)))


def _build_one_xml_file(xml_path: str, output_root: str, validate: bool) -> _ShardBuildResult:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        xml_path (str): 相关文件路径。
        output_root (str): 相关目录路径。
        validate (bool): 是否执行回放校验。
    
    返回:
        _ShardBuildResult: 函数执行结果。
    
    异常:
        无。
    """
    parser = TenhouParser()
    path = Path(xml_path)
    events = parser.parse_file(path)
    return _build_one_game(path.stem, events, Path(output_root), validate=validate)


def _build_one_event_shard(shard_path: str, output_root: str, validate: bool) -> _ShardBuildResult:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        shard_path (str): 相关文件路径。
        output_root (str): 相关目录路径。
        validate (bool): 是否执行回放校验。
    
    返回:
        _ShardBuildResult: 函数执行结果。
    
    异常:
        无。
    """
    path = Path(shard_path)
    events = read_events_from_parquet(path)
    return _build_one_game(path.stem, events, Path(output_root), validate=validate)


def _build_one_game(game_id: str, events: list, output_root: Path, validate: bool) -> _ShardBuildResult:
    """构建与函数名对应的数据结构或处理结果。
    
    参数:
        game_id (str): 函数执行所需的输入参数。
        events (list): 规范化事件列表。
        output_root (Path): 相关目录路径。
        validate (bool): 是否执行回放校验。
    
    返回:
        _ShardBuildResult: 函数执行结果。
    
    异常:
        无。
    """
    for event in events:
        if event.event_type == "init" and event.payload.get("feature_schema") != "full_state_v2":
            raise ValueError("old bronze schema: rerun preprocess for full_state_v2")
    if validate:
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
