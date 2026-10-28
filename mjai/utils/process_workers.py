from __future__ import annotations

import multiprocessing as mp
import sys
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

import pyarrow as pa

T = TypeVar("T")
R = TypeVar("R")


def initialize_data_worker() -> None:
    """Parallelize across files, without a large Arrow pool per process."""
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def process_map(func: Callable[[T], R], items: Iterable[T], workers: int) -> Iterator[R]:
    """Ordered spawn map with bounded submission and no silent thread fallback."""
    if workers <= 1:
        yield from map(func, items)
        return
    # Windows ProcessPoolExecutor supports at most 61 workers.
    worker_count = min(workers, 61) if sys.platform == "win32" else workers
    iterator = iter(items)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp.get_context("spawn"),
        initializer=initialize_data_worker,
    ) as executor:
        pending = deque()
        for _ in range(worker_count * 2):
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.append(executor.submit(func, item))
        while pending:
            yield pending.popleft().result()
            try:
                item = next(iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(func, item))
