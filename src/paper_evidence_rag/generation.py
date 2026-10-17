"""Pinned offline generation with explicit device, prompt and output budgets."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from .answering import MODEL_PINS, PROMPT_VERSION, messages, prompt_config, prompt_hash, sealed, validate_draft
from .reranking import checked_hash
from .schema import Passage


def rendered_prompt(row, passages, tokenizer, prompt_version=PROMPT_VERSION, model_id=None):
    # The fixed prefix is part of the model's input, not a response repair.
    # Count the same rendered string used for inference, including the prefix.
    options = {"enable_thinking": False} if model_id == "Qwen/Qwen3-1.7B" else {}
    return tokenizer.apply_chat_template(messages(row["question"], passages, prompt_version), tokenize=False,
        add_generation_prompt=True, **options) + prompt_config(prompt_version)["prefix"]


def pack_evidence(row, tokenizer, max_input_tokens=2048, prompt_version=PROMPT_VERSION, model_id=None):
    """Keep rank order and original character prefixes, recording every omission.

Token decoding can change whitespace. Prefixes are sliced from original text,
then the entire chat template is retokenized to verify the actual prompt budget.
"""
    def count(passages):
        return len(tokenizer(rendered_prompt(row, passages, tokenizer, prompt_version, model_id), add_special_tokens=False)["input_ids"])
    if count([]) > max_input_tokens:
        raise ValueError("Question and instructions exceed prompt budget")
    presented, omitted, truncated = [], [], []
    for raw in row["passages"]:
        passage = Passage.model_validate(raw).model_dump()
        if passage["paper_id"] != row["paper_id"]:
            raise ValueError("Cross-paper evidence")
        original_chars = len(passage["text"])
        # Both models receive identical 800-character source prefixes unless
        # the tokenizer budget requires further truncation. Compare artifacts
        # explicitly before describing a model experiment as paired.
        passage["text"] = passage["text"][:800]
        if count(presented + [passage]) <= max_input_tokens:
            presented.append(passage)
            if original_chars > len(passage["text"]):
                truncated.append({"passage_id": passage["passage_id"], "original_chars": original_chars,
                                  "presented_chars": len(passage["text"])})
            continue
        low, high = 0, len(passage["text"])
        while low < high:
            middle = (low + high + 1) // 2
            if count(presented + [{**passage, "text": passage["text"][:middle]}]) <= max_input_tokens:
                low = middle
            else:
                high = middle - 1
        prefix = passage["text"][:low]
        if len(prefix.strip()) >= 32:
            presented.append({**passage, "text": prefix})
            truncated.append({"passage_id": passage["passage_id"], "original_chars": original_chars, "presented_chars": low})
        else:
            omitted.append(passage["passage_id"])
    assert count(presented) <= max_input_tokens
    return presented, {"input_tokens": count(presented), "truncated": truncated, "omitted_ids": omitted}


def verify_snapshot(root, snapshot):
    """Verify all recorded files and complete sharded weights before model loading."""
    root = Path(root)
    if snapshot["id"] not in MODEL_PINS or snapshot["revision"] != MODEL_PINS[snapshot["id"]]:
        raise ValueError("Generator snapshot revision mismatch")
    files = set(snapshot["files"])
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= files:
        raise ValueError("Generator snapshot incomplete")
    for name, expected in snapshot["files"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Invalid generator snapshot path")
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError("Generator snapshot hash mismatch")
    if "model.safetensors.index.json" in files:
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        shards = set(index["weight_map"].values())
        if not shards or not shards <= files or "model.safetensors" in files:
            raise ValueError("Generator weight shards incomplete or ambiguous")
        if any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
            raise ValueError("Invalid generator shard path")
    elif "model.safetensors" not in files:
        raise ValueError("Generator weights missing")


def generate(requests, model_directory, max_input_tokens=2048, max_new_tokens=256, device="cpu"):
    checked_hash(requests)
    version = requests["prompt_version"]
    config = prompt_config(version)
    if not 512 <= max_input_tokens <= 2048 or not 16 <= max_new_tokens <= 256:
        raise ValueError("Invalid generation token budget")
    if len(requests["rows"]) > 20 or any(len(r["passages"]) > 5 for r in requests["rows"]):
        raise ValueError("Generation request budget exceeded")
    root = Path(model_directory)
    snapshot = json.loads((root / "snapshot-manifest.json").read_text(encoding="utf-8"))
    verify_snapshot(root, snapshot)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .runtime import configure, synchronize
    runtime = configure(torch, device)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(root), local_files_only=True,
        trust_remote_code=False, dtype=torch.float32).to(device).eval()
    synchronize(torch, device)
    load_seconds = time.perf_counter() - start
    rows, timings = [], []
    for row in requests["rows"]:
        started = time.perf_counter()
        presented, packing = pack_evidence(row, tokenizer, max_input_tokens, version, snapshot["id"])
        raw, output_tokens, limit_reached = "", 0, False
        if presented:
            # Greedy decoding has no temperature sampling. Decode only the
            # continuation, then retain the supplied assistant opening with it.
            # Models can still copy prompt placeholders; validation checks them.
            inputs = tokenizer(rendered_prompt(row, presented, tokenizer, version, snapshot["id"]), add_special_tokens=False,
                return_tensors="pt").to(device)
            assert inputs["input_ids"].shape[1] == packing["input_tokens"]
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id)
            new = output[0, inputs["input_ids"].shape[1]:]
            output_tokens = len(new)
            limit_reached = output_tokens == max_new_tokens and int(new[-1]) != tokenizer.eos_token_id
            raw = config["prefix"] + tokenizer.decode(new, skip_special_tokens=True)
        validated = validate_draft(raw, row["paper_id"], presented)
        if limit_reached:
            validated = {"status": "abstained", "reason": "output_limit", "claims": [], "answer": ""}
        rows.append({k: row[k] for k in ("paper_id", "question_id", "question")} | {
            "presented": presented, "packing": packing, "raw_output": raw,
            "output_tokens": output_tokens, "output_limit_reached": limit_reached, "result": validated})
        synchronize(torch, device)
        timings.append({"paper_id": row["paper_id"], "question_id": row["question_id"],
                        "seconds": time.perf_counter() - started})
    result = sealed({"request_sha256": requests["sha256"], "prompt_version": version,
        "prompt_sha256": prompt_hash(version), "assistant_prefix": config["prefix"],
        "rows": rows, "model": {"id": snapshot["id"], "revision": snapshot["revision"], "files": snapshot["files"],
            "device": device, "runtime": runtime, "dtype": "float32", "threads": 4, "seed": 0, "do_sample": False,
            "max_input_tokens": max_input_tokens, "max_new_tokens": max_new_tokens,
            "chat_template_options": {"enable_thinking": False} if snapshot["id"] == "Qwen/Qwen3-1.7B" else {},
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "dependencies": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "tokenizers")}}})
    return result, {"load_seconds": load_seconds, "questions": timings,
                    "inference_seconds": sum(r["seconds"] for r in timings)}
