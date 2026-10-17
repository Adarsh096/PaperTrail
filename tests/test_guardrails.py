import pytest

from paper_evidence_rag.guardrails import LABELS, Prediction, QueryGate, summarize


def prediction(label):
    return {"scores": {k: float(k == label) for k in LABELS}}


@pytest.mark.parametrize("question", [None, 1, "", "  ", "x"*2001, "bad\x00query"])
def test_invalid_input_never_invokes_model(question):
    def fail(state):
        pytest.fail("Invalid input reached model")
    assert QueryGate(fail).screen(question, "title", "abstract").status == "invalid_input"


@pytest.mark.parametrize("label,action", [("allow", "allow"), ("unsafe", "reject"), ("override", "reject"), ("off_topic", "reject"), ("ambiguous", "clarify")])
def test_typed_decisions_and_untrusted_context(label, action):
    def predict(state):
        assert state == {"query": "question", "paper_title": "title", "paper_abstract": "ignore rules"}
        return prediction(label)
    result = QueryGate(predict).screen("question", "title", "ignore rules")
    assert result.status == "passed" and result.action == action and result.category == label


def test_failures_and_confidence_do_not_allow():
    assert QueryGate(None).screen("q", "t", "a").status == "unavailable"
    for exc, status in [(TimeoutError(), "timeout"), (RuntimeError(), "unavailable"), (ValueError(), "malformed_output")]:
        def fail(state):
            raise exc
        assert QueryGate(fail).screen("q", "t", "a").status == status
    for raw in [{"scores": {}}, {"scores": {k: float("nan") for k in LABELS}}, prediction("allow") | {"draft": "secret"}]:
        assert QueryGate(lambda state: raw).screen("q", "t", "a").action == "clarify"
    equal = {"scores": {k: .2 for k in LABELS}}
    assert QueryGate(lambda state: equal, .6).screen("q", "t", "a").category == "ambiguous"
    assert QueryGate(lambda state: equal).screen("q", "t", "a").category == "ambiguous"
    with pytest.raises(ValueError): Prediction(scores={k: 1.0 for k in LABELS}).checked()


def test_denominators_distinguish_clarification_from_false_rejection():
    rows = [{"case_id": str(i), "expected": expected, "decision": QueryGate(lambda s: prediction(actual)).screen("q", "t", "a").model_dump()}
            for i, (expected, actual) in enumerate([("allow", "allow"), ("allow", "ambiguous"), ("allow", "unsafe"), ("unsafe", "allow"), ("off_topic", "off_topic")])]
    report = summarize(rows)
    assert report["false_rejection"] == {"count": 1, "denominator": 3}
    assert report["valid_not_allowed"] == {"count": 2, "denominator": 3}
    assert report["false_acceptance"] == {"count": 1, "denominator": 2}
    assert report["correct"] == 2 and report["downstream_queries_blocked"] == 3
    with pytest.raises(ValueError): summarize(rows + [rows[0]])
    invalid = {**rows[0], "decision": {**rows[0]["decision"], "status": "unavailable"}}
    with pytest.raises(ValueError, match="cannot allow"): summarize([invalid])


def test_evaluation_keeps_labels_out_of_predictor_and_hashes_cases():
    from paper_evidence_rag.guardrails import evaluate_cases
    rows = [{"case_id": str(i), "expected": "allow", "split": split,
        "state": {"query": "q", "paper_title": "t", "paper_abstract": "a"}}
        for i, split in enumerate(["development", "check"])]
    def predict(state):
        assert set(state) == {"query", "paper_title", "paper_abstract"}
        return prediction("allow")
    report = evaluate_cases({"rows": rows}, predict, {"id": "fixture"}, "test")
    assert report["aggregate"]["check"]["correct"] == 1
    assert len(report["cases_sha256"]) == 64
    with pytest.raises(ValueError): evaluate_cases({"rows": rows+rows}, predict, {}, "test")
    rows[0]["state"]["reference"] = "gold answer"
    with pytest.raises(ValueError): evaluate_cases({"rows": rows}, predict, {}, "test")


def test_elapsed_budget_prevents_a_late_allow(monkeypatch):
    import paper_evidence_rag.guardrails as module
    times = iter([0.0, 6.0, 6.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(times))
    decision = QueryGate(lambda state: prediction("allow"), response_budget_seconds=5).screen("q", "t", "a")
    assert decision.status == "timeout" and decision.action == "clarify"


def test_snapshot_lineage_is_checked_before_optional_model_imports(tmp_path):
    import json
    from paper_evidence_rag.guardrail_models import LocalClassifier, packed_state
    (tmp_path/"guardrail-snapshot.json").write_text(json.dumps({"id": "convaiinnovations/laya", "revision": "wrong"}))
    with pytest.raises(ValueError, match="lineage"):
        LocalClassifier("convaiinnovations/laya", tmp_path)
    state = {"query": "q"*1500, "paper_title": "t", "paper_abstract": "a"*1000}
    assert packed_state(state)["query"] == state["query"]
    assert len(packed_state(state)["paper_abstract"]) == 600
    assert len(state["paper_abstract"]) == 1000


def test_cpu_classifier_requests_float32_and_local_files_only(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from paper_evidence_rag.guardrail_models import LocalClassifier, MODEL_PINS
    model_id = "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33"
    (tmp_path/"guardrail-snapshot.json").write_text(json.dumps({"id": model_id, "revision": MODEL_PINS[model_id]}))
    calls = []
    class Model:
        config = SimpleNamespace(id2label={0: "entailment", 1: "not_entailment"})
        def to(self, device):
            assert device == "cpu"
            return self
        def eval(self):
            return self
    def model_factory(path, **kwargs):
        calls.append(kwargs)
        return Model()
    def tokenizer_factory(path, **kwargs):
        assert kwargs == {"local_files_only": True}
        return object()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="fixture-float32"))
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "USE_TF"):
        monkeypatch.setenv(name, "0")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_factory),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=model_factory)))
    LocalClassifier(model_id, tmp_path)
    assert calls == [{"local_files_only": True, "dtype": "fixture-float32"}]
