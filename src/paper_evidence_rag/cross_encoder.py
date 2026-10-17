"""Score frozen query-passage pairs offline on a pinned CPU model snapshot."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

from .ingestion import canonical, digest
from .reranking import checked_hash, pair_key, validate_model_pin


_DEVICE = re.compile(r"^(cpu|cuda)(?::([0-9]+))?$")


def _prepared_device_matches(actual, requested, current_cuda_index=None):
    """Compare a prepared tensor device with an explicit or current CUDA target."""
    if not isinstance(actual, str) or not isinstance(requested, str):
        return False
    actual_match, requested_match = _DEVICE.fullmatch(actual), _DEVICE.fullmatch(requested)
    if not actual_match or not requested_match:
        return False
    actual_type, actual_index = actual_match.groups()
    requested_type, requested_index = requested_match.groups()
    if actual_type != requested_type:
        return False
    if requested_type == "cpu":
        return actual_index is None and requested_index is None
    current = current_cuda_index if type(current_cuda_index) is int and current_cuda_index >= 0 else None
    expected_index = int(requested_index) if requested_index is not None else current
    actual_index = int(actual_index) if actual_index is not None else current
    return expected_index is not None and actual_index == expected_index


def score_candidates(candidates, model_directory, max_length=512, batch_size=8, device="cpu", prepared=None,
                     preserve_text=False):
    checked_hash(candidates)
    if not 32 <= max_length <= 512 or not 1 <= batch_size <= 32:
        raise ValueError("Invalid cross-encoder token or batch budget")
    if type(preserve_text) is not bool:
        raise ValueError("Invalid cross-encoder preservation policy")
    if len(candidates["questions"]) > 20 or len(candidates["rows"]) > 60:
        raise ValueError("Question budget exceeded")
    root = Path(model_directory)
    snapshot = json.loads((root / "snapshot-manifest.json").read_text(encoding="utf-8"))
    validate_model_pin(snapshot)
    if not {"model.safetensors", "config.json", "tokenizer.json"}.issubset(snapshot["files"]):
        raise ValueError("Reranker snapshot manifest incomplete")
    if prepared is None:
        for name, expected in snapshot["files"].items():
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Invalid model snapshot path")
            h = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError("Reranker snapshot hash mismatch")
    # No model libraries are imported by ordinary retrieval or fixture tests.
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from .runtime import configure, synchronize
    runtime = configure(torch, device)
    start = time.perf_counter()
    if prepared is None:
        tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(str(root), local_files_only=True,
            trust_remote_code=False, dtype=torch.float32).to(device).eval()
    else:
        tokenizer, model = prepared
        requested = str(device)
        current_cuda_index = torch.cuda.current_device() if requested.startswith("cuda") else None
        actual = str(next(model.parameters()).device)
        if not _prepared_device_matches(actual, requested, current_cuda_index):
            raise RuntimeError("Prepared reranker is not on the requested device")
    synchronize(torch, device)
    if model.config.num_labels != 1:
        raise ValueError("Expected a single relevance logit")
    if preserve_text:
        # Live BGE requests use its verified positional capacity, rather than
        # the smaller frozen-experiment budget. XLM-R reserves positions for
        # padding and its initial position. Never exceed this bounded lease.
        positions = getattr(model.config, "max_position_embeddings", None)
        padding = getattr(model.config, "pad_token_id", 0)
        if type(positions) is not int or type(padding) is not int:
            raise ValueError("Reranker positional capacity is unavailable")
        max_length = min(8192, positions - padding - 1)
        if max_length < 32:
            raise ValueError("Invalid reranker positional capacity")
    load_seconds = time.perf_counter() - start
    pairs = {}
    for row in candidates["rows"]:
        if len(row["candidates"]) > 100:
            raise ValueError("Candidate budget exceeded")
        for hit in row["candidates"]:
            passage = hit["passage"]
            key = pair_key(row["question"], passage["passage_id"])
            pair = (row["question"], passage["text"])
            if key in pairs and pairs[key] != pair:
                raise ValueError("Conflicting source text for pair")
            pairs[key] = pair
    counts, truncated, values = {}, [], {}
    # Preserve the whole query and truncate only the passage. Count the full
    # pair, including separator tokens, before applying the shared budget.
    for key, (query, passage) in pairs.items():
        query_count = len(tokenizer(query, add_special_tokens=False)["input_ids"])
        if query_count + tokenizer.num_special_tokens_to_add(pair=True) >= max_length:
            raise ValueError("Query exceeds token budget")
        counts[key] = len(tokenizer(query, passage, truncation=False)["input_ids"])
        if counts[key] > max_length:
            if preserve_text:
                raise ValueError("Complete candidate exceeds the reranker context capacity")
            truncated.append(key)
    ordered = sorted(pairs, key=lambda k: (counts[k], k))
    inference_start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(ordered), batch_size):
            keys = ordered[offset:offset + batch_size]
            inputs = tokenizer([pairs[k][0] for k in keys], [pairs[k][1] for k in keys],
                padding=True, truncation=False if preserve_text else "only_second",
                max_length=max_length, return_tensors="pt")
            inputs = inputs.to(device)
            logits = model(**inputs).logits[:, 0].tolist()
            values.update(zip(keys, logits))
    synchronize(torch, device)
    result = {"candidate_sha256": candidates["sha256"], "scores": values,
        "token_counts": counts, "truncated_pair_ids": sorted(truncated),
        "model": {"id": snapshot["id"], "revision": snapshot["revision"], "device": device, "dtype": "float32", "runtime": runtime,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "weight_file_bytes": (root / "model.safetensors").stat().st_size,
            "max_length": max_length, "batch_size": batch_size, "threads": 4, "seed": 0,
            "score_type": "raw relevance logit", "truncation": False if preserve_text else "only_second",
            "file_sha256": snapshot["files"], "dependencies": ({n: importlib.metadata.version(n)
                for n in ("torch", "transformers", "tokenizers")} if prepared is None else {})}}
    result["sha256"] = digest(canonical(result).encode())
    return result, {"model_load_seconds": load_seconds,
        "inference_seconds": time.perf_counter() - inference_start, "unique_pair_count": len(pairs)}
