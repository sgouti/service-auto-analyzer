from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from app.commons import logging
from app.commons.model.test_item_index import TestItemIndexData

LOGGER = logging.getLogger("analyzerApp.flakyDetector")


@dataclass(frozen=True)
class FlakyScore:
    score: int
    is_quarantined: bool


def _parse_timestamp(timestamp: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(timestamp, fmt)
        except ValueError:
            continue
    return None


def _extract_features(test_item: TestItemIndexData) -> list[float]:
    issue_history = list(test_item.issue_history or [])
    unique_issue_types = {entry.issue_type for entry in issue_history if entry.issue_type}
    issue_switches = 0
    previous_issue_type = ""
    manual_updates = 0
    timestamps: list[datetime] = []

    for entry in issue_history:
        if entry.issue_type and previous_issue_type and previous_issue_type != entry.issue_type:
            issue_switches += 1
        if not entry.is_auto_analyzed:
            manual_updates += 1
        if entry.issue_type:
            previous_issue_type = entry.issue_type
        parsed = _parse_timestamp(entry.timestamp)
        if parsed is not None:
            timestamps.append(parsed)

    gaps: list[float] = []
    for previous, current in zip(timestamps, timestamps[1:], strict=False):
        gaps.append(max(0.0, (current - previous).total_seconds()))

    message_lengths = [len((log.message or "").split()) for log in test_item.logs or []]
    return [
        float(len(issue_history)),
        float(len(unique_issue_types)),
        float(issue_switches),
        float(manual_updates),
        float(np.mean(gaps) if gaps else 0.0),
        float(np.std(gaps) if gaps else 0.0),
        float(test_item.log_count or len(test_item.logs or [])),
        float(np.mean(message_lengths) if message_lengths else 0.0),
    ]


class FlakyTestDetector:
    def __init__(self, quarantine_threshold: int) -> None:
        self.quarantine_threshold = quarantine_threshold

    def _heuristic_scores(self, test_items: Sequence[TestItemIndexData]) -> dict[str, FlakyScore]:
        results: dict[str, FlakyScore] = {}
        for test_item in test_items:
            features = _extract_features(test_item)
            score = min(
                100,
                int(
                    features[2] * 20
                    + features[1] * 10
                    + (15 if features[3] > 0 else 0)
                    + min(25, features[5] / 3600)
                ),
            )
            results[test_item.test_item_id] = FlakyScore(
                score=score,
                is_quarantined=score >= self.quarantine_threshold,
            )
        return results

    def score_items(self, test_items: Sequence[TestItemIndexData]) -> dict[str, FlakyScore]:
        if not test_items:
            return {}

        matrix = np.asarray([_extract_features(test_item) for test_item in test_items], dtype=np.float32)
        if len(test_items) < 10:
            return self._heuristic_scores(test_items)

        try:
            from pyod.models.iforest import IForest

            detector = IForest(contamination=0.1, random_state=42)
            detector.fit(matrix)
            raw_scores = np.asarray(detector.decision_scores_, dtype=np.float32)
        except Exception as exc:
            LOGGER.warning("Falling back to heuristic flaky scoring: %s", exc)
            return self._heuristic_scores(test_items)

        score_min = float(np.min(raw_scores))
        score_max = float(np.max(raw_scores))
        denominator = score_max - score_min if score_max > score_min else 1.0
        scores: dict[str, FlakyScore] = {}
        for test_item, raw_score in zip(test_items, raw_scores, strict=True):
            normalized = int(round(((float(raw_score) - score_min) / denominator) * 100.0))
            scores[test_item.test_item_id] = FlakyScore(
                score=normalized,
                is_quarantined=normalized >= self.quarantine_threshold,
            )
        return scores
