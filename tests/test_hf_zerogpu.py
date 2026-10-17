"""Model-free contract tests for the optional HF ZeroGPU answer adapter."""
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from paper_evidence_rag import hf_zerogpu as adapter
from paper_evidence_rag.answering import MODEL_PINS
from paper_evidence_rag.app_service import DemoService
from paper_evidence_rag.reranking import MODEL_PINS as RERANKER_PINS, pair_key


def make_snapshot(root, model_id, revision):
    root.mkdir()
    file = root / "config.json"
    file.write_text('{"fixture": true}', encoding="utf-8")
    manifest = {"id": model_id, "revision": revision,
        "files": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()}}
    (root / "snapshot-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


@pytest.fixture
def snapshots(tmp_path):
    return (
        make_snapshot(tmp_path / "bge", adapter.RERANKER_ID,
                      RERANKER_PINS[adapter.RERANKER_ID]),
        make_snapshot(tmp_path / "qwen", adapter.GENERATOR_ID,
                      MODEL_PINS[adapter.GENERATOR_ID]),
    )


def test_snapshot_pin_and_file_integrity(snapshots):
    bge, _ = snapshots
    assert adapter._verify_snapshot(bge, adapter.RERANKER_ID,
                                    RERANKER_PINS[adapter.RERANKER_ID]) == bge.resolve()
    with pytest.raises(ValueError, match="pin mismatch"):
        adapter._verify_snapshot(bge, adapter.RERANKER_ID, "0" * 40)
    (bge / "config.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        adapter._verify_snapshot(bge, adapter.RERANKER_ID,
                                 RERANKER_PINS[adapter.RERANKER_ID])


def test_snapshot_manifest_rejects_path_traversal(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    manifest = {"id": adapter.RERANKER_ID,
        "revision": RERANKER_PINS[adapter.RERANKER_ID],
        "files": {"../outside.json": hashlib.sha256(outside.read_bytes()).hexdigest()}}
    (root / "snapshot-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="path"):
        adapter._verify_snapshot(root, adapter.RERANKER_ID,
                                 RERANKER_PINS[adapter.RERANKER_ID])


def test_startup_prepare_precedes_one_runner_call(snapshots, monkeypatch):
    bge, qwen = snapshots
    prepared = []
    monkeypatch.setattr(adapter, "prepare_gpu_models", lambda *paths: prepared.append(paths) or {"device": "cuda"})
    calls = []
    def runner(payload):
        assert len(prepared) == 1
        calls.append(payload)
        return {"raw_output": "draft"}
    answerer = adapter.HFZeroGPUAnswerer(bge, qwen, runner)
    assert answerer.prepare() == {"device": "cuda"}
    assert len(prepared) == 1
    hits = [{"passage": {"passage_id": "p:1", "paper_id": "paper", "text": "evidence"}}]
    assert answerer("paper", "question", hits, "revision", "dense")["raw_output"] == "draft"
    assert len(calls) == 1
    assert calls[0]["budget_seconds"] == 180


def test_candidate_and_generation_budgets_are_explicit(snapshots):
    bge, qwen = snapshots
    calls = []
    answerer = adapter.HFZeroGPUAnswerer(bge, qwen, lambda payload: calls.append(payload) or {})
    hits = [{"passage": {"passage_id": str(i), "paper_id": "paper", "text": "evidence"}}
            for i in range(adapter.MAX_CANDIDATES + 1)]
    with pytest.raises(ValueError, match="20 candidates"):
        answerer("paper", "question", hits, "revision", "dense")
    assert calls == []
    assert adapter.MAX_INPUT_TOKENS == 2048
    assert adapter.MAX_NEW_TOKENS == 512


def test_inference_requires_models_prepared_at_startup(monkeypatch):
    monkeypatch.setattr(adapter, "_PREPARED", None)
    with pytest.raises(RuntimeError, match="not loaded during startup"):
        adapter.answer_with_snapshots({})


class FakeTensor:
    def __init__(self, length):
        self.shape = (1, length)

    def to(self, device):
        assert device == "cuda"
        return self


class FakeBatch(dict):
    def __init__(self, length):
        super().__init__(input_ids=FakeTensor(length))

    def to(self, device):
        assert device == "cuda"
        return self


class FakeTokenizer:
    eos_token_id = 0
    last_rendered = None

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        self.last_rendered = messages[-1]["content"]
        return " ".join(message["content"] for message in messages)

    def __call__(self, text, **kwargs):
        return FakeBatch(len(text))

    def decode(self, tokens, **kwargs):
        return '{"answer":"Readable draft","claims":[{"text":"invented quote",' \
            '"passage_id":"p:1"}],"limitations":"Review the sources."}'


class FakeOutput:
    def __getitem__(self, key):
        _, continuation = key
        return [1] * (adapter.MAX_NEW_TOKENS if continuation.stop is None else
                      continuation.stop - continuation.start)


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return FakeOutput()


def test_combined_inference_is_bounded_and_preserves_original_sources(snapshots, monkeypatch):
    bge, qwen = snapshots
    tokenizer, model = FakeTokenizer(), FakeModel()
    monkeypatch.setattr(adapter, "_PREPARED", {"paths": (str(bge.resolve()), str(qwen.resolve())),
        "reranker": (None, None), "generator": (tokenizer, model)})

    def score(candidates, *args, **kwargs):
        assert kwargs["preserve_text"] is True
        query = candidates["questions"][0]["question"]
        scores = {pair_key(query, hit["passage"]["passage_id"]): float(index)
                  for index, hit in enumerate(candidates["rows"][0]["candidates"])}
        return {"scores": scores}, {}

    monkeypatch.setattr("paper_evidence_rag.cross_encoder.score_candidates", score)
    monkeypatch.setitem(__import__("sys").modules, "torch",
                        SimpleNamespace(inference_mode=nullcontext))
    hits = [{"passage": {"passage_id": f"p:{i}", "paper_id": "paper",
             "text": ("original evidence " + str(i) + " ") * 45}, "rank": i, "score": 0.0}
            for i in range(2)]
    payload = {"reranker": str(bge.resolve()), "generator": str(qwen.resolve()),
        "hits": hits, "paper_id": "paper", "question": "question", "revision": "r", "mode": "dense"}
    draft = adapter.answer_with_snapshots(payload)
    assert draft["presented"] == [hit["passage"] for hit in draft["ranked_hits"][:len(draft["presented"])]]
    assert all(len(passage["text"]) > 800 for passage in draft["presented"])
    assert len(draft["presented"]) < len(draft["ranked_hits"])
    assert all(passage["text"] in tokenizer.last_rendered for passage in draft["presented"])
    assert all(hit["passage"]["text"] not in tokenizer.last_rendered
               for hit in draft["ranked_hits"] if hit["passage"] not in draft["presented"])
    assert model.kwargs["max_new_tokens"] == adapter.MAX_NEW_TOKENS
    assert model.kwargs["input_ids"].shape[1] <= adapter.MAX_INPUT_TOKENS
    assert draft["output_limit_reached"] is True
    assert "invented quote" in draft["raw_output"]


def test_best_passage_over_context_budget_fails_without_mutating_sources():
    tokenizer = FakeTokenizer()
    source = {"passage_id": "p:1", "paper_id": "paper", "text": "evidence " * 500}
    original = [dict(source)]
    with pytest.raises(ValueError, match="best-ranked complete passage"):
        adapter._fit_prompt_passages(original, {"paper_id": "paper", "question": "q"}, tokenizer)
    assert original == [source]


def test_service_keeps_readable_draft_and_marks_invalid_quote_unverified():
    passage = {"passage_id": "p:1", "paper_id": "paper", "text": "actual source text"}
    hits = [{"passage": passage, "rank": 1, "score": 0.5}]
    draft = {"answer_format": "hosted-grounded-v1", "prompt_version": "test",
        "presented": [passage], "raw_output": json.dumps({"answer": "Readable draft",
            "claims": [{"text": "invented quote", "passage_id": "p:1"}],
            "limitations": "Review the sources."})}
    checked = DemoService._checked_live(draft, "paper", hits)
    assert checked["answer"]["status"] == "unverified"
    assert checked["answer"]["answer"] == "Readable draft"
    assert checked["answer"]["invalid_claim_count"] == 1
    assert checked["presented"] == [passage]


def test_quote_in_context_dropped_passage_is_not_validated():
    best = {"passage_id": "p:best", "paper_id": "paper", "text": "the selected evidence"}
    dropped = {"passage_id": "p:dropped", "paper_id": "paper", "text": "quote from omitted passage"}
    hits = [{"passage": best, "rank": 1, "score": 1.0},
            {"passage": dropped, "rank": 2, "score": 0.0}]
    draft = {"answer_format": "hosted-grounded-v1", "presented": [best],
        "raw_output": json.dumps({"answer": "Readable draft", "claims": [
            {"text": "quote from omitted passage", "passage_id": "p:dropped"}],
            "limitations": "Review the sources."})}
    checked = DemoService._checked_live(draft, "paper", hits)
    assert checked["presented"] == [best]
    assert checked["answer"]["status"] == "unverified"
    assert checked["answer"]["invalid_claim_reasons"] == ["unknown_or_cross_paper_source"]
