from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from app.commons.model.launch_objects import ApplicationConfig


@dataclass(frozen=True)
class MlRuntimeSettings:
    enable_semantic_embedding: bool = True
    enable_hybrid_retrieval: bool = True
    enable_semantic_reranker: bool = True
    enable_lightgbm_classifier: bool = True
    enable_optuna_tuning: bool = True
    enable_flaky_detection: bool = True
    enable_async_ml_pipeline: bool = True
    semantic_model_cache_dir: str = "res/model/runtime"
    semantic_embedder_model_id: str = "BAAI/bge-m3"
    semantic_reranker_model_id: str = "BAAI/bge-reranker-base"
    semantic_embedder_model_path: str = "res/model/runtime/bge-m3"
    semantic_reranker_model_path: str = "res/model/runtime/bge-reranker-base"
    semantic_embedder_model_file: str = "onnx/model.onnx"
    semantic_reranker_model_file: str = "onnx/model.onnx"
    hybrid_rrf_k: int = 60
    hybrid_candidate_pool_size: int = 50
    reranker_candidate_window: int = 30
    reranker_result_window: int = 10
    ml_pipeline_timeout_seconds: int = 5
    semantic_embedding_batch_size: int = 32
    optuna_max_trials: int = 15
    optuna_min_f1_score: float = 0.80
    flaky_quarantine_threshold: int = 75
    semantic_vector_dimension: int = 1024

    @classmethod
    def from_env(cls) -> "MlRuntimeSettings":
        def to_bool(value: object) -> bool:
            return str(value).strip().lower() in {"1", "true", "y", "yes"}
        return cls(
            enable_semantic_embedding=to_bool(os.getenv("AA_ENABLE_BGE_M3", "true")),
            enable_hybrid_retrieval=to_bool(os.getenv("AA_ENABLE_HYBRID_RETRIEVAL", "true")),
            enable_semantic_reranker=to_bool(os.getenv("AA_ENABLE_RERANKER", "true")),
            enable_lightgbm_classifier=to_bool(os.getenv("AA_ENABLE_LIGHTGBM", "true")),
            enable_optuna_tuning=to_bool(os.getenv("AA_ENABLE_OPTUNA", "true")),
            enable_flaky_detection=to_bool(os.getenv("AA_ENABLE_FLAKY_DETECTION", "true")),
            enable_async_ml_pipeline=to_bool(os.getenv("AA_ENABLE_ASYNC_PIPELINE", "true")),
            semantic_model_cache_dir=os.getenv("AA_MODEL_CACHE_DIR", "res/model/runtime").strip(),
            semantic_embedder_model_id=os.getenv("AA_BGE_M3_MODEL_ID", "BAAI/bge-m3").strip(),
            semantic_reranker_model_id=os.getenv("AA_BGE_RERANKER_MODEL_ID", "BAAI/bge-reranker-base").strip(),
            semantic_embedder_model_path=os.getenv("AA_BGE_M3_MODEL_PATH", "res/model/runtime/bge-m3").strip(),
            semantic_reranker_model_path=os.getenv(
                "AA_BGE_RERANKER_MODEL_PATH", "res/model/runtime/bge-reranker-base"
            ).strip(),
            semantic_embedder_model_file=os.getenv("AA_BGE_M3_MODEL_FILE", "onnx/model.onnx").strip(),
            semantic_reranker_model_file=os.getenv("AA_BGE_RERANKER_MODEL_FILE", "onnx/model.onnx").strip(),
            hybrid_rrf_k=int(os.getenv("AA_HYBRID_RRF_K", "60")),
            hybrid_candidate_pool_size=int(os.getenv("AA_HYBRID_CANDIDATE_POOL", "50")),
            reranker_candidate_window=int(os.getenv("AA_RERANK_CANDIDATES", "30")),
            reranker_result_window=int(os.getenv("AA_RERANK_RESULTS", "10")),
            ml_pipeline_timeout_seconds=int(os.getenv("AA_PIPELINE_TIMEOUT_SECONDS", "5")),
            semantic_embedding_batch_size=int(os.getenv("AA_EMBEDDING_BATCH_SIZE", "32")),
            optuna_max_trials=int(os.getenv("AA_OPTUNA_MAX_TRIALS", "15")),
            optuna_min_f1_score=float(os.getenv("AA_OPTUNA_MIN_F1", "0.80")),
            flaky_quarantine_threshold=int(os.getenv("AA_FLAKY_THRESHOLD", "75")),
            semantic_vector_dimension=int(os.getenv("AA_SEMANTIC_VECTOR_DIM", "1024")),
        )

    @classmethod
    def from_app_config(cls, app_config: ApplicationConfig) -> "MlRuntimeSettings":
        return cls(
            enable_semantic_embedding=app_config.enableSemanticEmbedding,
            enable_hybrid_retrieval=app_config.enableHybridRetrieval,
            enable_semantic_reranker=app_config.enableSemanticReranker,
            enable_lightgbm_classifier=app_config.enableLightgbmClassifier,
            enable_optuna_tuning=app_config.enableOptunaTuning,
            enable_flaky_detection=app_config.enableFlakyDetection,
            enable_async_ml_pipeline=app_config.enableAsyncMlPipeline,
            semantic_model_cache_dir=app_config.semanticModelCacheDir,
            semantic_embedder_model_id=app_config.semanticEmbedderModelId,
            semantic_reranker_model_id=app_config.semanticRerankerModelId,
            semantic_embedder_model_path=app_config.semanticEmbedderModelPath,
            semantic_reranker_model_path=app_config.semanticRerankerModelPath,
            semantic_embedder_model_file=app_config.semanticEmbedderModelFile,
            semantic_reranker_model_file=app_config.semanticRerankerModelFile,
            hybrid_rrf_k=app_config.hybridRrfK,
            hybrid_candidate_pool_size=app_config.hybridCandidatePoolSize,
            reranker_candidate_window=app_config.rerankerCandidateWindow,
            reranker_result_window=app_config.rerankerResultWindow,
            ml_pipeline_timeout_seconds=app_config.mlPipelineTimeoutSeconds,
            semantic_embedding_batch_size=app_config.semanticEmbeddingBatchSize,
            optuna_max_trials=app_config.optunaMaxTrials,
            optuna_min_f1_score=app_config.optunaMinF1Score,
            flaky_quarantine_threshold=app_config.flakyQuarantineThreshold,
            semantic_vector_dimension=app_config.semanticVectorDimension,
        )


def get_runtime_settings(app_config: Optional[ApplicationConfig] = None) -> MlRuntimeSettings:
    if app_config is not None:
        return MlRuntimeSettings.from_app_config(app_config)
    return MlRuntimeSettings.from_env()
