"""Optional adapter for one injected Hugging Face ZeroGPU inference lease.

Importing this module does not import spaces, torch, or transformers. The host
calls ``HFZeroGPUAnswerer.prepare`` during startup, before ZeroGPU admission,
then injects one bounded ``spaces.GPU`` runner for each combined request.
"""
import hashlib
import json
from pathlib import Path
import time

from .answering import MODEL_PINS
from .hosted_answers import SYSTEM as ANSWER_SYSTEM, ANSWER_FORMAT
from .ingestion import canonical, digest
from .reranking import MODEL_PINS as RERANKER_PINS, pair_key, rerank

RERANKER_ID = "BAAI/bge-reranker-v2-m3"
GENERATOR_ID = "Qwen/Qwen3-1.7B"
MAX_CANDIDATES = 20
MAX_INPUT_TOKENS = 2048
MAX_NEW_TOKENS = 512
_PREPARED = None


def _verify_snapshot(directory, model_id, revision):
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / "snapshot-manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("id"), manifest.get("revision")) != (model_id, revision):
        raise ValueError(f"Snapshot pin mismatch for {model_id}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"Snapshot file manifest missing for {model_id}")
    for name, expected in files.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Invalid snapshot file path")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Snapshot file hash mismatch for {model_id}")
    return root


def prepare_gpu_models(reranker_directory, generator_directory):
    """Validate and load both local snapshots on CUDA before a GPU lease."""
    global _PREPARED
    reranker = _verify_snapshot(reranker_directory, RERANKER_ID, RERANKER_PINS[RERANKER_ID])
    generator = _verify_snapshot(generator_directory, GENERATOR_ID, MODEL_PINS[GENERATOR_ID])
    import torch
    from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("ZeroGPU startup did not provide CUDA")
    reranker_tokenizer = AutoTokenizer.from_pretrained(str(reranker), local_files_only=True,
        trust_remote_code=False)
    reranker_model = AutoModelForSequenceClassification.from_pretrained(str(reranker),
        local_files_only=True, trust_remote_code=False, dtype=torch.float32).to("cuda").eval()
    if reranker_model.config.num_labels != 1:
        raise ValueError("Expected a single BGE relevance logit")
    generator_tokenizer = AutoTokenizer.from_pretrained(str(generator), local_files_only=True,
        trust_remote_code=False)
    generator_model = AutoModelForCausalLM.from_pretrained(str(generator), local_files_only=True,
        trust_remote_code=False, dtype=torch.float32).to("cuda").eval()
    _PREPARED = {"paths": (str(reranker), str(generator)),
        "reranker": (reranker_tokenizer, reranker_model),
        "generator": (generator_tokenizer, generator_model)}
    return {"device": "cuda", "reranker": RERANKER_ID, "generator": GENERATOR_ID}


def _hosted_prompt(row, passages, tokenizer):
    messages = [{"role": "system", "content": ANSWER_SYSTEM +
                 " Keep the answer under 100 words, include at most two supporting quotes of at most 160 characters each, and keep limitations under 40 words."},
        {"role": "user", "content": "Question:\n" + row["question"] + "\n\nPassages:\n" +
         "\n\n".join(f"[{p['passage_id']}] {p['text']}" for p in passages)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)


def _fit_prompt_passages(originals, row, tokenizer):
    """Select complete passages only, preserving exact model/UI evidence."""
    prompt_passages = [dict(passage) for passage in originals]
    def encode(values):
        rendered = _hosted_prompt(row, values, tokenizer)
        return tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    encoded = encode(prompt_passages)
    while int(encoded["input_ids"].shape[1]) > MAX_INPUT_TOKENS and prompt_passages:
        prompt_passages.pop()
        encoded = encode(prompt_passages)
    if not prompt_passages:
        raise ValueError("The best-ranked complete passage exceeds the model context budget")
    return prompt_passages, prompt_passages, encoded


def answer_with_snapshots(payload):
    """Rerank then generate in the same lease using startup-loaded CUDA models."""
    if _PREPARED is None:
        raise RuntimeError("ZeroGPU models were not loaded during startup")
    if (_PREPARED["paths"] != (payload.get("reranker"), payload.get("generator"))):
        raise RuntimeError("ZeroGPU request snapshots differ from startup models")
    hits = payload.get("hits")
    if not isinstance(hits, list) or not 1 <= len(hits) <= MAX_CANDIDATES:
        raise ValueError("GPU request must contain between 1 and 20 candidates")
    from .cross_encoder import score_candidates
    q = {"paper_id": payload["paper_id"], "question": payload["question"],
         "question_id": "interactive-" + digest(payload["question"].encode("utf-8"))}
    candidates = {"questions": [q], "rows": [{**q, "mode": payload["mode"], "candidates": hits}],
        "corpus_revision": payload["revision"], "input_sha256": digest(canonical(hits).encode("utf-8")),
        "embedding_sha256": None, "candidate_count": len(hits)}
    candidates["sha256"] = digest(canonical(candidates).encode("utf-8"))
    scores, _ = score_candidates(candidates, Path(payload["reranker"]), batch_size=4,
        device="cuda", prepared=_PREPARED["reranker"], preserve_text=True)
    ranked = rerank(hits, q["question"], scores["scores"])
    presented = [dict(h["passage"]) for h in ranked[:5]]
    tokenizer, model = _PREPARED["generator"]
    row = {"paper_id": q["paper_id"], "question": q["question"]}
    presented, prompt_passages, encoded = _fit_prompt_passages(presented, row, tokenizer)
    encoded = encoded.to("cuda")
    import torch
    with torch.inference_mode():
        output = model.generate(**encoded, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id)
    generated = output[0, encoded["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    return {"answer_format": ANSWER_FORMAT, "prompt_version": "hf-zerogpu-hosted-v1",
        "presented": presented, "raw_output": raw,
        "output_limit_reached": len(generated) == MAX_NEW_TOKENS and
            int(generated[-1]) != tokenizer.eos_token_id,
        "ranked_hits": ranked, "model": {"reranker": RERANKER_ID,
            "reranker_revision": RERANKER_PINS[RERANKER_ID], "generator": GENERATOR_ID,
            "generator_revision": MODEL_PINS[GENERATOR_ID], "device": "cuda",
            "dtype": "float32", "do_sample": False, "max_input_tokens": MAX_INPUT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS, "prompt_passage_count": len(prompt_passages),
            "chat_template_options": {"enable_thinking": False}}}


class HFZeroGPUAnswerer:
    """Call an injected host-decorated GPU function once per answer."""
    device = "cuda"
    origin = "Qwen3 with BGE reranking on Hugging Face ZeroGPU"

    def __init__(self, reranker, generator, gpu_runner, max_seconds=180):
        if not callable(gpu_runner):
            raise TypeError("An injected bounded ZeroGPU runner is required")
        if type(max_seconds) is not int or not 30 <= max_seconds <= 300:
            raise ValueError("GPU request budget must be between 30 and 300 seconds")
        self.reranker = _verify_snapshot(reranker, RERANKER_ID, RERANKER_PINS[RERANKER_ID])
        self.generator = _verify_snapshot(generator, GENERATOR_ID, MODEL_PINS[GENERATOR_ID])
        self.gpu_runner, self.max_seconds = gpu_runner, max_seconds

    def prepare(self):
        return prepare_gpu_models(self.reranker, self.generator)

    def __call__(self, paper_id, question, hits, revision, mode):
        if not isinstance(hits, list) or not 1 <= len(hits) <= MAX_CANDIDATES:
            raise ValueError("GPU request must contain between 1 and 20 candidates")
        payload = {"paper_id": paper_id, "question": question, "hits": hits,
            "revision": revision, "mode": mode, "reranker": str(self.reranker),
            "generator": str(self.generator), "budget_seconds": self.max_seconds}
        started = time.monotonic()
        result = self.gpu_runner(payload)
        if time.monotonic() - started > self.max_seconds:
            raise TimeoutError("ZeroGPU answer exceeded its configured request budget")
        if not isinstance(result, dict):
            raise RuntimeError("ZeroGPU runner returned an invalid answer")
        return result
