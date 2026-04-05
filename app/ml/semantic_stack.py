from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.commons import logging

LOGGER = logging.getLogger("analyzerApp.semanticStack")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


def get_rrf_k() -> int:
    return int(os.getenv("RP_SEMANTIC_RRF_K", "60"))


def get_min_cluster_size() -> int:
    return int(os.getenv("RP_SEMANTIC_MIN_CLUSTER_SIZE", "2"))


def get_embedding_model_name() -> str:
    return os.getenv("RP_SEMANTIC_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def get_cache_dir() -> str | None:
    cache_dir = os.getenv("RP_SEMANTIC_CACHE_DIR", "").strip()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    default_cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "semantic-models"
    default_cache_dir.mkdir(parents=True, exist_ok=True)
    return str(default_cache_dir)

try:
    from fastembed import TextEmbedding  # type: ignore
    import faiss  # type: ignore
    import hdbscan  # type: ignore
    from rank_bm25 import BM25Okapi  # type: ignore

    SEMANTIC_DEPS_AVAILABLE = True
except ImportError:
    TextEmbedding = None
    faiss = None
    hdbscan = None
    BM25Okapi = None
    SEMANTIC_DEPS_AVAILABLE = False


@dataclass(frozen=True)
class HybridRankedResult:
    index: int
    score: float


def _tokenize(text: str) -> list[str]:
    return [token for token in (text or "").lower().split() if token]


class SemanticFailureModels:
    def __init__(self, model_name: str | None = None):
        if not SEMANTIC_DEPS_AVAILABLE:
            raise RuntimeError("Semantic dependencies are not installed")

        resolved_model_name = model_name or get_embedding_model_name()
        cache_dir = get_cache_dir()

        try:
            self.model = TextEmbedding(
                resolved_model_name,
                cache_dir=cache_dir,
                providers=["CPUExecutionProvider"],
            )
        except Exception:
            if _env_flag("RP_SEMANTIC_ALLOW_MODEL_DOWNLOAD", "false"):
                self.model = TextEmbedding(
                    resolved_model_name,
                    cache_dir=cache_dir,
                    providers=["CPUExecutionProvider"],
                )
            else:
                raise

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        embeddings = list(self.model.embed(texts))
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        return vectors

    def cluster_texts(self, texts: list[str], min_cluster_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.asarray([], dtype=np.int64)
        if len(texts) == 1:
            return np.asarray([0], dtype=np.int64)

        embeddings = self.embed_texts(texts)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min_cluster_size or get_min_cluster_size()),
            metric="euclidean",
            cluster_selection_method="eom",
        )
        return np.asarray(clusterer.fit_predict(embeddings), dtype=np.int64)

    def hybrid_rank(
        self,
        query_text: str,
        candidate_texts: list[str],
        *,
        limit: int | None = None,
    ) -> list[HybridRankedResult]:
        if not query_text or not candidate_texts:
            return []

        query_embedding = self.embed_texts([query_text])
        candidate_embeddings = self.embed_texts(candidate_texts)
        if query_embedding.size == 0 or candidate_embeddings.size == 0:
            return []

        top_k = min(limit or len(candidate_texts), len(candidate_texts))

        semantic_index = faiss.IndexFlatIP(candidate_embeddings.shape[1])
        semantic_index.add(candidate_embeddings)
        _, semantic_matches = semantic_index.search(query_embedding, top_k)
        semantic_ranks = {
            int(candidate_idx): rank + 1
            for rank, candidate_idx in enumerate(semantic_matches[0])
            if candidate_idx >= 0
        }

        tokenized_corpus = [_tokenize(candidate) for candidate in candidate_texts]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = np.asarray(bm25.get_scores(_tokenize(query_text)), dtype=np.float32)
        bm25_order = np.argsort(-bm25_scores)
        bm25_ranks = {int(candidate_idx): rank + 1 for rank, candidate_idx in enumerate(bm25_order)}

        ranked_results = []
        for candidate_idx in range(len(candidate_texts)):
            reciprocal_rank = 0.0
            if candidate_idx in semantic_ranks:
                reciprocal_rank += 1.0 / (get_rrf_k() + semantic_ranks[candidate_idx])
            if candidate_idx in bm25_ranks:
                reciprocal_rank += 1.0 / (get_rrf_k() + bm25_ranks[candidate_idx])
            if reciprocal_rank > 0:
                ranked_results.append(HybridRankedResult(index=candidate_idx, score=reciprocal_rank * 100.0))

        ranked_results.sort(key=lambda result: result.score, reverse=True)
        return ranked_results[:top_k]


@lru_cache(maxsize=1)
def get_semantic_models() -> SemanticFailureModels | None:
    if not _env_flag("RP_SEMANTIC_MODELS_ENABLED", "true"):
        return None
    if "PYTEST_CURRENT_TEST" in os.environ:
        return None
    if not SEMANTIC_DEPS_AVAILABLE:
        LOGGER.warning("Semantic analyzer dependencies are unavailable; using legacy analyzer logic.")
        return None
    try:
        return SemanticFailureModels()
    except Exception as exc:
        LOGGER.warning("Semantic analyzer models are unavailable; falling back to legacy logic.", exc_info=exc)
        return None