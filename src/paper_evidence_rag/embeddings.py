"""Encode a bounded corpus offline; model acquisition is a separate step."""
import importlib.metadata
import json
from pathlib import Path
import time

from .ingestion import canonical, digest, load_artifacts

MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
QUERY_PROMPT = "task: question answering | query: "
DOCUMENT_PROMPT = "title: none | text: "


def encode(artifacts, model_directory, max_length=512, batch_size=4, extra_question=None, device="cpu"):
    import torch
    print("Torch imported; loading sentence-transformers", flush=True)
    from sentence_transformers import SentenceTransformer

    manifest, contents = load_artifacts(Path(artifacts))
    if manifest["paper_count"] > 20 or manifest["passage_count"] > 2000:
        raise ValueError("Encoding is bounded to 20 papers and 2000 passages")
    if max_length < 32 or max_length > 2048 or batch_size < 1:
        raise ValueError("Invalid token or batch budget")
    model_directory = Path(model_directory)
    weights = model_directory / "model.safetensors"
    if not weights.is_file():
        raise ValueError("Pinned model snapshot is missing")
    snapshot = json.loads((model_directory / "snapshot-manifest.json").read_text(encoding="utf-8"))
    if snapshot.get("id") != MODEL_ID or snapshot.get("revision") != MODEL_REVISION:
        raise ValueError("Model snapshot revision mismatch")
    if not {"model.safetensors", "config.json", "tokenizer.json", "modules.json"}.issubset(snapshot["files"]):
        raise ValueError("Model snapshot manifest is incomplete")
    import hashlib
    model_files = {}
    for name, expected in snapshot["files"].items():
        path = model_directory / name
        if not path.resolve().is_relative_to(model_directory.resolve()):
            raise ValueError("Invalid model snapshot path")
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError("Model snapshot file hash mismatch")
        model_files[name] = expected
    from .runtime import configure, synchronize
    runtime = configure(torch, device)
    start = time.perf_counter()
    print("Snapshot hashes verified; loading text-only model on " + device, flush=True)
    model = SentenceTransformer(str(model_directory), device=device, local_files_only=True,
        trust_remote_code=False, config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32})
    model.max_seq_length = max_length
    print("Model loaded; counting input tokens", flush=True)
    passages = contents["passages.json"]
    questions = sorted(contents["questions.json"], key=lambda q: (q["paper_id"], q["question_id"]))[:20]
    if extra_question is not None:
        questions.append({"question": extra_question})
    document_vectors, query_vectors, truncated = {}, {}, {"passages": [], "queries": []}
    token_counts = {"passages": {}, "queries": {}}
    for label, rows, prompt, result in (
        ("passages", [(p["passage_id"], p["text"]) for p in passages], DOCUMENT_PROMPT, document_vectors),
        ("queries", [(digest(q["question"].encode()), q["question"]) for q in questions], QUERY_PROMPT, query_vectors),
    ):
        for key, text in rows:
            count = len(model.tokenizer(prompt + text, truncation=False)["input_ids"])
            token_counts[label][key] = count
            if count > max_length:
                truncated[label].append(key)
        # Group similar lengths to avoid padding every short paragraph to the
        # longest paragraph in an arbitrary source-order batch. IDs retain their
        # source positions; this scheduling choice does not change source text.
        rows.sort(key=lambda row: (token_counts[label][row[0]], row[0]))
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            vectors = model.encode([text for _, text in batch], prompt=prompt, batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
            for (key, _), values in zip(batch, vectors):
                result[key] = values.tolist()
            if offset % 64 == 0:
                print(f"Encoded {label}: {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
    synchronize(torch, device)
    result = {"corpus_revision": manifest["corpus_revision"], "model": {
        "id": MODEL_ID, "revision": MODEL_REVISION, "dimension": 768, "device": device,
        "runtime": runtime,
        "dtype": "float32", "max_length": max_length, "batch_size": batch_size,
        "query_prompt": QUERY_PROMPT, "document_prompt": DOCUMENT_PROMPT,
        "normalize_embeddings": True, "vision": False, "audio": False,
        "file_sha256": model_files, "dependencies": {n: importlib.metadata.version(n)
            for n in ("sentence-transformers", "transformers", "torch")}},
        "passages": document_vectors, "queries": query_vectors,
        "token_counts": token_counts, "truncated": truncated}
    result["sha256"] = digest(canonical(result).encode())
    return result, {"encode_seconds": time.perf_counter() - start}
