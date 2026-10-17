"""Pinned embedding batch for Ragas reverse questions, never judge credentials."""
import hashlib
import json
import math

from .answering import sealed
from .embeddings import MODEL_ID, MODEL_REVISION, QUERY_PROMPT
from .ingestion import canonical, digest
from .reranking import checked_hash


def embedding_requests(texts):
    texts = sorted(set(texts))
    if not texts or len(texts) > 200 or any(not t.strip() or len(t) > 4000 for t in texts):
        raise ValueError("Semantic embeddings require 1 to 200 short nonempty texts")
    return sealed({"model": {"id": MODEL_ID, "revision": MODEL_REVISION, "query_prompt": QUERY_PROMPT},
                   "texts": {digest(t.encode()): t for t in texts}})


def validate_vectors(value):
    checked_hash(value)
    requests = value["requests"]
    checked_hash(requests)
    if requests != embedding_requests(requests["texts"].values()):
        raise ValueError("Semantic embedding request mismatch")
    if set(value["vectors"]) != set(requests["texts"]):
        raise ValueError("Semantic embedding coverage mismatch")
    for vector in value["vectors"].values():
        if len(vector) != 768 or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector) or sum(v*v for v in vector) == 0:
            raise ValueError("Invalid semantic vector")
    if value["truncated"]:
        raise ValueError("Truncated semantic questions must not be scored silently")


def encode_texts(requests, model_directory, device="cuda"):
    """Encode short questions with the same pinned query prompt as retrieval."""
    checked_hash(requests)
    if requests != embedding_requests(requests["texts"].values()):
        raise ValueError("Semantic embedding request mismatch")
    snapshot = json.loads((model_directory / "snapshot-manifest.json").read_text())
    if snapshot["id"] != MODEL_ID or snapshot["revision"] != MODEL_REVISION:
        raise ValueError("Embedding model revision mismatch")
    if not {"model.safetensors", "config.json", "tokenizer.json", "modules.json"}.issubset(snapshot["files"]):
        raise ValueError("Incomplete embedding snapshot")
    for name, expected in snapshot["files"].items():
        path = model_directory / name
        if not path.resolve().is_relative_to(model_directory.resolve()):
            raise ValueError("Unsafe model snapshot path")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError("Model file hash mismatch")
    import torch
    from sentence_transformers import SentenceTransformer
    from .runtime import configure
    runtime = configure(torch, device)
    model = SentenceTransformer(str(model_directory), device=device, local_files_only=True,
        trust_remote_code=False, config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32})
    model.max_seq_length = 512
    rows = sorted(requests["texts"].items())
    counts = {key: len(model.tokenizer(QUERY_PROMPT+text, truncation=False)["input_ids"]) for key, text in rows}
    vectors = model.encode([t for _, t in rows], prompt=QUERY_PROMPT, batch_size=4,
        normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    value = sealed({"requests": requests, "vectors": {k: v.tolist() for (k, _), v in zip(rows, vectors)},
        "token_counts": counts, "truncated": [k for k, count in counts.items() if count > 512],
        "runtime": runtime, "file_sha256": snapshot["files"]})
    validate_vectors(value)
    return value


def cached_embeddings(value=None):
    from ragas.embeddings.base import BaseRagasEmbedding
    if value is not None:
        validate_vectors(value)

    class FrozenEmbeddings(BaseRagasEmbedding):
        def __init__(self):
            super().__init__()
            self.pending = set()
            self.missing_calls = 0
        def embed_text(self, text, **kwargs):
            key = digest(text.encode())
            if value is not None and key in value["vectors"] and value["requests"]["texts"][key] == text:
                return value["vectors"][key]
            self.pending.add(text)
            self.missing_calls += 1
            # Collection-only sentinel; callers mark this metric pending and
            # discard its numeric value. It is never an evaluation score.
            return [1.0] + [0.0] * 767
        async def aembed_text(self, text, **kwargs):
            return self.embed_text(text)
    return FrozenEmbeddings()
