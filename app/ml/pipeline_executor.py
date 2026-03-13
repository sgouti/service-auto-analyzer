from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Callable, Optional, Sequence, TypeVar

from app.commons import logging
from app.commons.model.db import Hit
from app.commons.model.log_item_index import LogItemIndexData

LOGGER = logging.getLogger("analyzerApp.pipelineExecutor")
T = TypeVar("T")


def execute_with_timeout(
    items: Sequence[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]],
    worker: Callable[[LogItemIndexData, list[Hit[LogItemIndexData]]], list[Hit[LogItemIndexData]]],
    *,
    timeout_seconds: int,
    enabled: bool,
) -> list[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]]:
    if not enabled or len(items) <= 1:
        return [(query_log, worker(query_log, hits)) for query_log, hits in items]

    results: list[Optional[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]]] = [None] * len(items)
    future_index: dict[object, int] = {}

    with ThreadPoolExecutor(max_workers=min(4, len(items))) as executor:
        for index, (query_log, hits) in enumerate(items):
            future = executor.submit(worker, query_log, hits)
            future_index[future] = index

        try:
            for future in as_completed(future_index, timeout=timeout_seconds):
                index = future_index[future]
                query_log, original_hits = items[index]
                try:
                    results[index] = (query_log, future.result())
                except Exception as exc:
                    LOGGER.warning("ML pipeline worker failed, falling back to original ranking: %s", exc)
                    results[index] = (query_log, original_hits)
        except TimeoutError:
            LOGGER.warning(
                "ML pipeline timed out after %s seconds, falling back for unfinished tasks",
                timeout_seconds,
            )

    for index, result in enumerate(results):
        if result is None:
            results[index] = items[index]

    return [result for result in results if result is not None]
