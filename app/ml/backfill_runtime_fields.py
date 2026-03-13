from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import opensearchpy.helpers

from app.commons.model.test_item_index import TestItemIndexData
from app.commons.os_client import OsClient, get_test_item_index_name
from app.ml.flaky_detector import FlakyTestDetector
from app.ml.runtime_settings import get_runtime_settings
from app.ml.semantic_runtime import get_semantic_runtime


def iter_project_ids() -> Iterable[str]:
    raw_value = os.getenv("AA_BACKFILL_PROJECTS", "").strip()
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def apply_semantic_fields(test_item: TestItemIndexData) -> None:
    settings = get_runtime_settings()
    runtime = get_semantic_runtime(settings)
    log_vectors = []
    for log_item in test_item.logs or []:
        text = log_item.whole_message or ""
        if not text.strip():
            continue
        vector = runtime.embed_texts([text])[0].tolist()
        log_item.semantic_vector = vector
        log_item.semantic_vector_model = settings.semantic_embedder_model_id
        log_vectors.append(vector)
    if log_vectors:
        test_item.semantic_vector = np.asarray(log_vectors, dtype=np.float32).mean(axis=0).tolist()
        test_item.semantic_vector_model = settings.semantic_embedder_model_id


def main() -> None:
    settings = get_runtime_settings()
    app_config = None
    from app.main import APP_CONFIG

    app_config = APP_CONFIG
    os_client = OsClient(app_config)
    detector = FlakyTestDetector(settings.flaky_quarantine_threshold) if settings.enable_flaky_detection else None

    for project_id in iter_project_ids():
        index_name = get_test_item_index_name(project_id, app_config.esProjectIndexPrefix)
        hits = opensearchpy.helpers.scan(
            os_client._os_client,
            index=index_name,
            query={"query": {"match_all": {}}},
            preserve_order=False,
        )

        batch: list[TestItemIndexData] = []
        for hit in hits:
            batch.append(TestItemIndexData.from_dict(hit.get("_source", {})))
            if len(batch) >= app_config.esChunkNumber:
                if detector:
                    for item in batch:
                        apply_semantic_fields(item)
                    flaky_scores = detector.score_items(batch)
                    for item in batch:
                        score = flaky_scores.get(item.test_item_id)
                        if score is not None:
                            item.flaky_score = score.score
                            item.is_quarantined = score.is_quarantined
                else:
                    for item in batch:
                        apply_semantic_fields(item)
                os_client.bulk_index(project_id, batch, refresh=False)
                batch = []

        if batch:
            if detector:
                for item in batch:
                    apply_semantic_fields(item)
                flaky_scores = detector.score_items(batch)
                for item in batch:
                    score = flaky_scores.get(item.test_item_id)
                    if score is not None:
                        item.flaky_score = score.score
                        item.is_quarantined = score.is_quarantined
            else:
                for item in batch:
                    apply_semantic_fields(item)
            os_client.bulk_index(project_id, batch, refresh=True)


if __name__ == "__main__":
    main()
