from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import psutil
from rank_bm25 import BM25Okapi

from app.commons.model.db import Hit
from app.commons.model.log_item_index import LogItemIndexData
from app.ml.runtime_settings import MlRuntimeSettings, get_runtime_settings
from app.ml.semantic_runtime import get_semantic_runtime, semantic_tokenize

LOGGER = logging.getLogger("analyzerApp")

CPU_HIGH_THRESHOLD = 90
CPU_MEDIUM_THRESHOLD = 70
REDUCED_RERANK_DEPTH = 3


def build_ranking_text(log_item: LogItemIndexData) -> str:
    parts = [
        log_item.whole_message,
        log_item.detected_message_without_params_extended,
        log_item.message_without_params_extended,
        log_item.stacktrace_extended,
        log_item.test_item_name,
    ]
    return " ".join([part.strip() for part in parts if part and part.strip()])


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], *, k: int) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, item_index in enumerate(ranking):
            fused[item_index] = fused.get(item_index, 0.0) + (1.0 / (k + position + 1))
    return fused


@dataclass
class RankedHit:
    hit: Hit[LogItemIndexData]
    bm25_score: float
    dense_score: float
    hybrid_score: float
    rerank_score: float


class HybridRankingPipeline:
    def __init__(self, settings: MlRuntimeSettings | None = None) -> None:
        self.settings = settings or get_runtime_settings()
        self.runtime = get_semantic_runtime(self.settings)

    def _score_bm25(self, query_text: str, documents: Sequence[str]) -> list[float]:
        tokenized_documents = [semantic_tokenize(document) for document in documents]
        query_tokens = semantic_tokenize(query_text)
        if not query_tokens or not tokenized_documents:
            return [0.0 for _ in documents]
        bm25 = BM25Okapi(tokenized_documents)
        scores = bm25.get_scores(query_tokens)
        return [float(score) for score in scores.tolist()]

    def rank_hits(
        self, query_log: LogItemIndexData, hits: Sequence[Hit[LogItemIndexData]],
    ) -> list[Hit[LogItemIndexData]]:
        if not hits:
            return []

        query_text = build_ranking_text(query_log)
        documents = [build_ranking_text(hit.source) for hit in hits]
        bm25_scores = self._score_bm25(query_text, documents)
        dense_scores = self.runtime.dense_scores(query_text, documents)

        lexical_ranking = sorted(range(len(hits)), key=lambda index: bm25_scores[index], reverse=True)
        dense_ranking = sorted(range(len(hits)), key=lambda index: dense_scores[index], reverse=True)
        fused = reciprocal_rank_fusion((lexical_ranking, dense_ranking), k=self.settings.hybrid_rrf_k)

        hybrid_ranking = sorted(range(len(hits)), key=lambda index: fused.get(index, 0.0), reverse=True)

        cpu_pct = psutil.cpu_percent(interval=0.1)
        if cpu_pct > CPU_HIGH_THRESHOLD:
            LOGGER.warning("CPU at %.0f%% — bypassing reranker entirely", cpu_pct)
            candidate_window = hybrid_ranking[: self.settings.reranker_result_window]
            rerank_by_index = {idx: fused.get(idx, 0.0) for idx in candidate_window}
        else:
            depth = REDUCED_RERANK_DEPTH if cpu_pct > CPU_MEDIUM_THRESHOLD else self.settings.reranker_candidate_window
            if cpu_pct > CPU_MEDIUM_THRESHOLD:
                LOGGER.info("CPU at %.0f%% — reducing rerank depth to %d", cpu_pct, depth)
            candidate_window = hybrid_ranking[: depth]
            candidate_docs = [documents[index] for index in candidate_window]
            rerank_scores = self.runtime.rerank_scores(query_text, candidate_docs)
            rerank_by_index = {
                candidate_index: rerank_scores[position]
                for position, candidate_index in enumerate(candidate_window)
            }

        ranked_hits: list[RankedHit] = []
        for index, hit in enumerate(hits):
            rerank_score = rerank_by_index.get(index, 0.0)
            ranked_hit = hit.model_copy(deep=True)
            ranked_hit.fields = {
                **(ranked_hit.fields or {}),
                "bm25_score": bm25_scores[index],
                "dense_score": dense_scores[index],
                "hybrid_rrf_score": fused.get(index, 0.0),
                "rerank_score": rerank_score,
                "original_score": ranked_hit.score or 0.0,
            }
            ranked_hit.score = rerank_score if rerank_score > 0.0 else fused.get(index, 0.0)
            ranked_hits.append(
                RankedHit(
                    hit=ranked_hit,
                    bm25_score=bm25_scores[index],
                    dense_score=dense_scores[index],
                    hybrid_score=fused.get(index, 0.0),
                    rerank_score=rerank_score,
                )
            )

        ranked_hits.sort(
            key=lambda item: (
                item.rerank_score,
                item.hybrid_score,
                item.dense_score,
                item.bm25_score,
            ),
            reverse=True,
        )
        return [item.hit for item in ranked_hits[: self.settings.reranker_result_window]]


def rank_search_results(
    search_results: Sequence[tuple[LogItemIndexData, Sequence[Hit[LogItemIndexData]]]],
    settings: MlRuntimeSettings | None = None,
) -> list[tuple[LogItemIndexData, list[Hit[LogItemIndexData]]]]:
    pipeline = HybridRankingPipeline(settings)
    return [
        (query_log, pipeline.rank_hits(query_log, hits))
        for query_log, hits in search_results
    ]
