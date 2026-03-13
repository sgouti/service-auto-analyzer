from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np
from cachetools import LRUCache

from app.commons.model.launch_objects import SimilarityResult
from app.ml.runtime_settings import MlRuntimeSettings, get_runtime_settings

LOGGER = logging.getLogger("analyzerApp.semanticRuntime")
TOKEN_SPLIT_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def semantic_tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_SPLIT_PATTERN.split((text or "").lower()) if token]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[np.ndarray]:
        ...


class TextReranker(Protocol):
    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        ...


class HashingTextEmbedder:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[np.ndarray]:
        embeddings: list[np.ndarray] = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in semantic_tokenize(text):
                index = hash(("q" if is_query else "d", token)) % self.dimension
                vector[index] += 1.0
            embeddings.append(_normalize_vector(vector))
        return embeddings


class HashingTextReranker:
    def __init__(self, dimension: int) -> None:
        self._embedder = HashingTextEmbedder(dimension)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        queries = [query for query, _ in pairs]
        docs = [doc for _, doc in pairs]
        query_vectors = self._embedder.embed(queries, is_query=True)
        doc_vectors = self._embedder.embed(docs)
        return [
            float(np.dot(query_vector, doc_vector))
            for query_vector, doc_vector in zip(query_vectors, doc_vectors, strict=True)
        ]


class FastEmbedTextEmbedder:
    def __init__(self, settings: MlRuntimeSettings) -> None:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        self._batch_size = settings.semantic_embedding_batch_size
        self._model_name = settings.semantic_embedder_model_id
        specific_model_path = settings.semantic_embedder_model_path
        model_dir = os.path.abspath(specific_model_path)
        if not any(model.get("model") == self._model_name for model in TextEmbedding.list_supported_models()):
            TextEmbedding.add_custom_model(
                model=self._model_name,
                pooling=PoolingType.CLS,
                normalization=True,
                sources=ModelSource(hf=settings.semantic_embedder_model_id),
                dim=settings.semantic_vector_dimension,
                model_file=settings.semantic_embedder_model_file,
                size_in_gb=0.6,
            )
        self._model = TextEmbedding(
            model_name=self._model_name,
            cache_dir=settings.semantic_model_cache_dir,
            lazy_load=False,
            local_files_only=True,
            specific_model_path=model_dir,
        )

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> list[np.ndarray]:
        if not texts:
            return []
        if is_query:
            vectors = list(self._model.query_embed(texts, batch_size=self._batch_size))
        else:
            vectors = list(self._model.embed(texts, batch_size=self._batch_size))
        return [np.asarray(vector, dtype=np.float32) for vector in vectors]


class FastEmbedTextReranker:
    def __init__(self, settings: MlRuntimeSettings) -> None:
        from fastembed.common.model_description import ModelSource
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._batch_size = max(1, min(settings.reranker_candidate_window, 64))
        self._model_name = settings.semantic_reranker_model_id
        model_dir = os.path.abspath(settings.semantic_reranker_model_path)
        if not any(model.get("model") == self._model_name for model in TextCrossEncoder.list_supported_models()):
            TextCrossEncoder.add_custom_model(
                model=self._model_name,
                sources=ModelSource(hf=settings.semantic_reranker_model_id),
                model_file=settings.semantic_reranker_model_file,
                size_in_gb=0.6,
            )
        self._model = TextCrossEncoder(
            model_name=self._model_name,
            cache_dir=settings.semantic_model_cache_dir,
            lazy_load=False,
            local_files_only=True,
            specific_model_path=model_dir,
        )

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = list(self._model.rerank_pairs(pairs, batch_size=self._batch_size, normalize=True))
        return [float(score) for score in scores]


@dataclass
class SemanticScoreSet:
    dense_scores: list[float]
    rerank_scores: list[float]


class SemanticRuntime:
    _instance: Optional["SemanticRuntime"] = None
    _instance_lock = threading.Lock()

    def __init__(self, settings: MlRuntimeSettings) -> None:
        self.settings = settings
        self._cache: LRUCache[str, np.ndarray] = LRUCache(maxsize=4096)
        self._cache_lock = threading.RLock()
        self._model_lock = threading.Lock()
        self._embedder: Optional[TextEmbedder] = None
        self._reranker: Optional[TextReranker] = None

    @classmethod
    def get(cls, settings: Optional[MlRuntimeSettings] = None) -> "SemanticRuntime":
        runtime_settings = settings or get_runtime_settings()
        with cls._instance_lock:
            if cls._instance is None or cls._instance.settings != runtime_settings:
                cls._instance = cls(runtime_settings)
        return cls._instance

    def _build_embedder(self) -> TextEmbedder:
        if not self.settings.enable_semantic_embedding:
            return HashingTextEmbedder(self.settings.semantic_vector_dimension)
        try:
            return FastEmbedTextEmbedder(self.settings)
        except Exception as exc:
            LOGGER.warning("Falling back to hashing embedder: %s", exc)
            return HashingTextEmbedder(self.settings.semantic_vector_dimension)

    def _build_reranker(self) -> TextReranker:
        if not self.settings.enable_semantic_reranker:
            return HashingTextReranker(self.settings.semantic_vector_dimension)
        try:
            return FastEmbedTextReranker(self.settings)
        except Exception as exc:
            LOGGER.warning("Falling back to hashing reranker: %s", exc)
            return HashingTextReranker(self.settings.semantic_vector_dimension)

    @property
    def embedder(self) -> TextEmbedder:
        if self._embedder is None:
            with self._model_lock:
                if self._embedder is None:
                    self._embedder = self._build_embedder()
        return self._embedder

    @property
    def reranker(self) -> TextReranker:
        if self._reranker is None:
            with self._model_lock:
                if self._reranker is None:
                    self._reranker = self._build_reranker()
        return self._reranker

    def _cache_key(self, text: str, is_query: bool) -> str:
        return ("query:" if is_query else "doc:") + (text or "")

    def embed_texts(self, texts: Sequence[str], *, is_query: bool = False) -> list[np.ndarray]:
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        results: list[Optional[np.ndarray]] = [None] * len(texts)

        with self._cache_lock:
            for index, text in enumerate(texts):
                key = self._cache_key(text, is_query)
                cached = self._cache.get(key)
                if cached is not None:
                    results[index] = cached
                else:
                    missing_indices.append(index)
                    missing_texts.append(text)

        if missing_texts:
            embedded = self.embedder.embed(missing_texts, is_query=is_query)
            with self._cache_lock:
                for index, vector in zip(missing_indices, embedded, strict=True):
                    key = self._cache_key(texts[index], is_query)
                    self._cache[key] = vector
                    results[index] = vector

        return [
            vector if vector is not None
            else np.zeros(self.settings.semantic_vector_dimension)
            for vector in results
        ]

    def dense_scores(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        query_vector = self.embed_texts([query], is_query=True)[0]
        doc_vectors = self.embed_texts(documents)
        return [float(np.dot(query_vector, doc_vector)) for doc_vector in doc_vectors]

    def rerank_scores(self, query: str, documents: Sequence[str]) -> list[float]:
        pairs = [(query, document) for document in documents]
        if not pairs:
            return []
        return self.reranker.score(pairs)

    def score(self, query: str, documents: Sequence[str]) -> SemanticScoreSet:
        return SemanticScoreSet(
            dense_scores=self.dense_scores(query, documents),
            rerank_scores=self.rerank_scores(query, documents),
        )

    def calculate_similarity(self, base_text: Optional[str], other_texts: Sequence[str]) -> list[SimilarityResult]:
        if base_text is None or not other_texts:
            return []

        if not base_text.strip():
            return [
                SimilarityResult(similarity=0.0, both_empty=not candidate.strip())
                for candidate in other_texts
            ]

        dense_scores = self.dense_scores(base_text, list(other_texts))
        results: list[SimilarityResult] = []
        for candidate, score in zip(other_texts, dense_scores, strict=True):
            if not candidate.strip():
                results.append(SimilarityResult(similarity=0.0, both_empty=False))
                continue
            if base_text == candidate:
                results.append(SimilarityResult(similarity=1.0, both_empty=False))
                continue
            similarity = max(0.0, min(1.0, (score + 1.0) / 2.0))
            results.append(SimilarityResult(similarity=similarity, both_empty=False))
        return results


def get_semantic_runtime(settings: Optional[MlRuntimeSettings] = None) -> SemanticRuntime:
    return SemanticRuntime.get(settings)
