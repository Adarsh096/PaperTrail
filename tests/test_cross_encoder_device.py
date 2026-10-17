"""Prepared-model device checks do not require CUDA or model weights."""
import pytest
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

from paper_evidence_rag.cross_encoder import _prepared_device_matches


@pytest.mark.parametrize("actual,requested,current,expected", [
    ("cuda:0", "cuda", 0, True),
    ("cuda:1", "cuda", 1, True),
    ("cuda:0", "cuda", 1, False),
    ("cuda:1", "cuda:0", 0, False),
    ("cuda:0", "cuda:0", 0, True),
    ("cpu", "cpu", None, True),
    ("cuda:0", "cpu", 0, False),
    ("cpu", "cuda", 0, False),
    ("mps", "cpu", None, False),
    ("cuda:0", "cuda", None, False),
    ("cuda:0", "cuda:-1", 0, False),
])
def test_prepared_device_matches_current_or_explicit_target(actual, requested, current, expected):
    assert _prepared_device_matches(actual, requested, current) is expected


@pytest.mark.parametrize("actual,requested", [(None, "cuda"), ("cuda:0", None), ("cuda:0", "cuda:zero")])
def test_prepared_device_rejects_invalid_device_values(actual, requested):
    assert not _prepared_device_matches(actual, requested, 0)


@pytest.mark.parametrize("preserve,length,expected_truncation", [
    (True, 700, False), (False, 700, "only_second"),
])
def test_live_reranker_preserves_candidates_beyond_experiment_budget(
        tmp_path, monkeypatch, preserve, length, expected_truncation):
    scores, tokenizer = _score_fixture(tmp_path, monkeypatch, length, preserve)
    assert tokenizer.batch_options["truncation"] == expected_truncation
    assert tokenizer.documents == ["e" * length]
    assert scores["model"]["max_length"] == (8192 if preserve else 512)
    assert len(scores["truncated_pair_ids"]) == (0 if preserve else 1)


def test_live_reranker_rejects_over_capacity_without_silent_truncation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Complete candidate exceeds"):
        _score_fixture(tmp_path, monkeypatch, 9000, True)


def _score_fixture(tmp_path, monkeypatch, length, preserve):
    from paper_evidence_rag import cross_encoder, runtime
    from paper_evidence_rag.answering import sealed
    from paper_evidence_rag.reranking import DEFAULT_MODEL_ID, MODEL_PINS
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    snapshot = {"id": DEFAULT_MODEL_ID, "revision": MODEL_PINS[DEFAULT_MODEL_ID],
                "files": {name: "fixture" for name in ("model.safetensors", "config.json", "tokenizer.json")}}
    (tmp_path / "snapshot-manifest.json").write_text(json.dumps(snapshot))

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        def num_special_tokens_to_add(self, pair=True):
            return 3

        def __call__(self, query, passage=None, **kwargs):
            if isinstance(query, list):
                self.batch_options, self.documents = kwargs, passage
                return Batch(input_ids=query)
            return {"input_ids": [0] * (len(query) + (len(passage) + 3 if passage else 0))}

    class Logits:
        def __getitem__(self, key):
            return self

        def tolist(self):
            return [0.5]

    class Model:
        config = SimpleNamespace(num_labels=1, max_position_embeddings=8194, pad_token_id=1)

        def parameters(self):
            return iter([SimpleNamespace(device="cpu", numel=lambda: 1)])

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=Logits())

    tokenizer = Tokenizer()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=None, AutoModelForSequenceClassification=None))
    monkeypatch.setattr(runtime, "configure", lambda *args: {})
    monkeypatch.setattr(runtime, "synchronize", lambda *args: None)
    query = "question"
    candidates = sealed({"questions": [{"question": query}], "rows": [{"question": query,
        "candidates": [{"passage": {"passage_id": "p", "text": "e" * length}}]}]})
    result, _ = cross_encoder.score_candidates(candidates, tmp_path,
        prepared=(tokenizer, Model()), preserve_text=preserve)
    return result, tokenizer
