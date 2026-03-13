from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from app.ml.runtime_settings import get_runtime_settings

DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.onnx_data",
    "tokenizer.*",
    "vocab.*",
    "special_tokens_map.json",
    "sentencepiece.*",
    "onnx/*",
    "**/onnx/*",
    "**/*.onnx",
    "**/*.ort",
]
DEFAULT_IGNORE_PATTERNS = [
    "*.bin",
    "*.pt",
    "*.safetensors",
    "*.msgpack",
    "*.h5",
]


def bake_model(repo_id: str, target_dir: str, model_file: str, token: str | None) -> None:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_path),
        local_dir_use_symlinks=False,
        allow_patterns=DEFAULT_ALLOW_PATTERNS,
        ignore_patterns=DEFAULT_IGNORE_PATTERNS,
        token=token,
        resume_download=True,
    )
    model_path = target_path / model_file
    if not model_path.exists():
        allow_missing = os.getenv("AA_ALLOW_MISSING_MODEL_FILES", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        }
        if allow_missing:
            print(f"Skipping missing model artifact: {model_path}")
            return
        raise FileNotFoundError(f"Expected ONNX model file was not downloaded: {model_path}")


def main() -> None:
    if os.getenv("AA_BAKE_MODELS", "true").strip().lower() not in {"1", "true", "yes", "y"}:
        return

    settings = get_runtime_settings()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    bake_model(
        settings.semantic_embedder_model_id,
        settings.semantic_embedder_model_path,
        settings.semantic_embedder_model_file,
        token,
    )
    bake_model(
        settings.semantic_reranker_model_id,
        settings.semantic_reranker_model_path,
        settings.semantic_reranker_model_file,
        token,
    )


if __name__ == "__main__":
    main()
