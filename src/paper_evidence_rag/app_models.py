"""Optional offline model adapters. Nothing downloads weights automatically."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .answering import sealed
from .embeddings import DOCUMENT_PROMPT, MODEL_ID, MODEL_REVISION, QUERY_PROMPT
from .ingestion import canonical, digest


class LocalEncoder:
    """Lazily load one verified CPU encoder and retain it across requests."""
    def __init__(self, directory, metadata):
        self.directory, self.metadata, self.model = Path(directory), metadata, None
        if (metadata["id"], metadata["revision"], metadata["dimension"]) != (MODEL_ID, MODEL_REVISION, 768):
            raise ValueError("Encoder metadata does not match the selected model")
        if metadata["query_prompt"] != QUERY_PROMPT or metadata["document_prompt"] != DOCUMENT_PROMPT:
            raise ValueError("Encoder prompts do not match the stored vectors")
        if metadata.get("dtype") != "float32" or metadata.get("normalize_embeddings") is not True:
            raise ValueError("Encoder normalization and precision must match the corpus")
        if type(metadata.get("max_length")) is not int or not 32 <= metadata["max_length"] <= 2048:
            raise ValueError("Invalid encoder token budget")

    def _load(self):
        if self.model is not None:
            return
        snapshot = json.loads((self.directory / "snapshot-manifest.json").read_text(encoding="utf-8"))
        if (snapshot["id"], snapshot["revision"]) != (MODEL_ID, MODEL_REVISION):
            raise ValueError("Encoder snapshot revision mismatch")
        if snapshot["files"] != self.metadata["file_sha256"]:
            raise ValueError("Encoder files do not match the frozen corpus model")
        for name, expected in snapshot["files"].items():
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError("Invalid encoder snapshot path")
            with path.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError("Encoder snapshot file changed")
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2)
        self.model = SentenceTransformer(str(self.directory), device="cpu", local_files_only=True,
            trust_remote_code=False, config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"torch_dtype": torch.float32})
        self.model.max_seq_length = self.metadata["max_length"]

    def _encode(self, texts, prompt):
        self._load()
        counts = [len(self.model.tokenizer(prompt + t, truncation=False)["input_ids"]) for t in texts]
        if any(n > self.model.max_seq_length for n in counts):
            kind = "Question" if prompt == QUERY_PROMPT else "Prepared document"
            raise ValueError(kind + " exceeds the encoder token budget.")
        return self.model.encode(texts, prompt=prompt, normalize_embeddings=True, batch_size=4,
            show_progress_bar=False, convert_to_numpy=True).tolist()

    def encode_question(self, text):
        return self._encode([text], QUERY_PROMPT)[0]

    def encode_documents(self, texts):
        return self._encode(texts, DOCUMENT_PROMPT)

    def split_documents(self, texts):
        """Split passages by this snapshot's real token budget, without decoding.

        Returned offsets are zero-based, end-exclusive character positions in
        each original input. Adjacent pieces cover every input character once.
        """
        self._load()
        tokenizer = self.model.tokenizer
        limit = self.model.max_seq_length
        result = []
        for text in texts:
            pieces, start = [], 0
            while start < len(text):
                if len(tokenizer(DOCUMENT_PROMPT + text[start:], truncation=False)["input_ids"]) <= limit:
                    pieces.append({"text": text[start:], "start": start, "end": len(text)})
                    break
                lo, hi = start + 1, len(text)
                best = start
                while lo <= hi:
                    end = (lo + hi) // 2
                    count = len(tokenizer(DOCUMENT_PROMPT + text[start:end], truncation=False)["input_ids"])
                    if count <= limit:
                        best, lo = end, end + 1
                    else:
                        hi = end - 1
                if best == start:
                    raise ValueError("Encoder prompt leaves no token budget for document text")
                end = best
                if end < len(text):
                    # Prefer a nearby sentence or word boundary while retaining
                    # the exact contiguous source slice and respecting the limit.
                    floor = start + max(1, (end - start) * 3 // 4)
                    candidates = []
                    for pattern in (r"\n\s*\n", r"[.!?][\"')\]]*\s+", r"\s+"):
                        candidates = [m.end() for m in re.finditer(pattern, text[start:end])
                                      if start + m.end() >= floor]
                        if candidates:
                            break
                    if candidates:
                        preferred_end = start + candidates[-1]
                        preferred_count = len(tokenizer(
                            DOCUMENT_PROMPT + text[start:preferred_end], truncation=False)["input_ids"])
                        if preferred_count <= limit:
                            end = preferred_end
                pieces.append({"text": text[start:end], "start": start, "end": end})
                start = end
            result.append(pieces)
        return result


class SnapshotAnswerer:
    """Bound one live BGE/Qwen3 request in an isolated process.

    The CPU query encoder persists in the server. These two larger models load
    within each worker, trading latency for releasing memory after a request.
    A later hosting adapter may keep them warm after memory measurements.
    """
    def __init__(self, reranker, generator, device="cpu", timeout=180):
        if device not in ("cpu", "cuda"):
            raise ValueError("Invalid inference device")
        self.reranker, self.generator = str(Path(reranker).resolve()), str(Path(generator).resolve())
        for directory, expected in ((self.reranker, "BAAI/bge-reranker-v2-m3"), (self.generator, "Qwen/Qwen3-1.7B")):
            if json.loads((Path(directory) / "snapshot-manifest.json").read_text())["id"] != expected:
                raise ValueError("Live app requires the selected BGE and Qwen3 snapshots")
        self.device, self.timeout = device, timeout

    def __call__(self, paper_id, question, hits, revision, mode):
        with tempfile.TemporaryDirectory(prefix="paper-answer-") as directory:
            root = Path(directory)
            payload = {"paper_id": paper_id, "question": question, "hits": hits, "revision": revision,
                       "mode": mode, "reranker": self.reranker, "generator": self.generator, "device": self.device}
            (root / "input.json").write_text(canonical(payload), encoding="utf-8")
            try:
                result = subprocess.run([sys.executable, "-m", __name__, str(root)],
                    capture_output=True, timeout=self.timeout, check=False)
            except subprocess.TimeoutExpired:
                raise TimeoutError("Live answer budget exceeded") from None
            if result.returncode != 0 or not (root / "output.json").exists():
                raise RuntimeError("Offline answer worker failed")
            return json.loads((root / "output.json").read_text(encoding="utf-8"))


def _answer_worker(root):
    from .answering import prepare_requests, PROMPT_VERSION
    from .cross_encoder import score_candidates
    from .generation import generate
    value = json.loads((root / "input.json").read_text(encoding="utf-8"))
    q = {"paper_id": value["paper_id"], "question": value["question"],
         "question_id": "interactive-" + digest(value["question"].encode())}
    candidates = sealed({"questions": [q], "rows": [{**q, "mode": value["mode"], "candidates": value["hits"]}],
        "corpus_revision": value["revision"], "input_sha256": digest(canonical(value["hits"]).encode()),
        "embedding_sha256": None, "candidate_count": 20})
    scores, _ = score_candidates(candidates, Path(value["reranker"]), device=value["device"])
    request = prepare_requests(candidates, scores, value["mode"], prompt_version=PROMPT_VERSION)
    output, _ = generate(request, Path(value["generator"]), device=value["device"])
    (root / "output.json").write_text(canonical(output["rows"][0]), encoding="utf-8")


if __name__ == "__main__":
    _answer_worker(Path(sys.argv[1]))
