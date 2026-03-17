"""
Shared helpers for the standalone parquet embedding pipeline.
"""

from __future__ import annotations

import os
import re
import threading
from collections import defaultdict, deque
from typing import Sequence

import certifi
import httpx

DEFAULT_BOOTSTRAP_MODEL = "perplexity-ai/pplx-embed-context-v1-0.6b"
DEFAULT_SEARCH_MODEL = "perplexity-ai/pplx-embed-context-v1-4b"
DEFAULT_CANDIDATE_ALIAS = "synthetic_candidate_profiles"
DEFAULT_CHUNK_ALIAS = "synthetic_candidate_chunks"
DEFAULT_DENSE_VECTOR_NAME = "dense"
DEFAULT_SPARSE_VECTOR_NAME = "bm25"
DEFAULT_BM25_MODEL = "Qdrant/bm25"

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def env_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_hf_http_clients() -> None:
    from huggingface_hub import close_session, set_async_client_factory, set_client_factory

    verify = False if env_enabled("HF_HUB_DISABLE_SSL_VERIFY") else certifi.where()
    timeout = httpx.Timeout(300.0, connect=30.0)

    def client_factory() -> httpx.Client:
        return httpx.Client(verify=verify, timeout=timeout, follow_redirects=True)

    def async_client_factory() -> httpx.AsyncClient:
        return httpx.AsyncClient(verify=verify, timeout=timeout, follow_redirects=True)

    set_client_factory(client_factory)
    set_async_client_factory(async_client_factory)
    close_session()


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def physical_collection_name(alias: str, model_name: str) -> str:
    return f"{alias}__{model_slug(model_name)}"


def choose_diverse_profile_ids(metadata_docs: list[dict], target_count: int) -> list[str]:
    if target_count <= 0 or len(metadata_docs) <= target_count:
        return [doc["profil_id"] for doc in metadata_docs]

    buckets = defaultdict(list)
    for doc in metadata_docs:
        key = (
            str(doc.get("family") or "unknown").lower(),
            str(doc.get("country") or "unknown").lower(),
            str(doc.get("seniority") or "unknown").lower(),
        )
        buckets[key].append(doc["profil_id"])

    grouped = {
        key: deque(sorted(profil_ids))
        for key, profil_ids in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0]))
    }

    selected: list[str] = []
    while grouped and len(selected) < target_count:
        empty_keys = []
        for key in list(grouped.keys()):
            if len(selected) >= target_count:
                break
            if grouped[key]:
                selected.append(grouped[key].popleft())
            if not grouped[key]:
                empty_keys.append(key)
        for key in empty_keys:
            grouped.pop(key, None)
    return selected


class StandaloneEmbedder:
    def __init__(self, model_name: str, device: str = "auto") -> None:
        configure_hf_http_clients()
        self.model_name = model_name
        self.device = device.lower()
        self._lock = threading.RLock()
        self._model = None
        self._dimension = None
        self._snapshot_path = None

    def prefetch(self) -> str:
        from huggingface_hub import snapshot_download

        with self._lock:
            if self._snapshot_path:
                return self._snapshot_path
            self._snapshot_path = snapshot_download(repo_id=self.model_name)
            return self._snapshot_path

    def _infer_dimension(self) -> int | None:
        if self._model is None:
            return None
        config = getattr(self._model, "config", None)
        for attribute in ("embedding_dimension", "projection_dim", "hidden_size", "d_model"):
            value = getattr(config, attribute, None)
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            model_path = self.prefetch()
            import numpy as np
            import torch
            from transformers import AutoModel

            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "low_cpu_mem_usage": True,
            }
            if self.device == "auto":
                if torch.cuda.is_available():
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["torch_dtype"] = torch.bfloat16
                else:
                    load_kwargs["torch_dtype"] = torch.float32
            elif self.device == "cuda":
                load_kwargs["device_map"] = "auto"
                load_kwargs["torch_dtype"] = torch.bfloat16
            else:
                load_kwargs["torch_dtype"] = torch.float32

            self._model = AutoModel.from_pretrained(model_path, **load_kwargs)
            self._dimension = self._infer_dimension()
            if self._dimension is None:
                sample = self._model.encode([["dimension probe"]])
                self._dimension = int(np.asarray(sample[0]).shape[-1])

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        import numpy as np

        embeddings = self._model.encode([[text] for text in texts])
        return [np.asarray(item[0]).tolist() for item in embeddings]

    def encode_contextual_documents(
        self,
        documents: Sequence[Sequence[str]],
    ) -> list[list[list[float]]]:
        if not documents:
            return []
        self._load()
        import numpy as np

        embeddings = self._model.encode([list(chunks) for chunks in documents])
        return [
            [np.asarray(chunk_vector).tolist() for chunk_vector in document_vectors]
            for document_vectors in embeddings
        ]

    def dimension(self) -> int:
        self._load()
        return int(self._dimension)
