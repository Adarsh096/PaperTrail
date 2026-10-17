"""Important routing and provider-failure behavior, without network or weights."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from paper_evidence_rag.cloud_pipeline import (SAFETY_MODEL, RERANK_MODEL, GENERATION_MODELS,
    OpenRouterClient, PacedBudget, SafetyFirstGate, CloudAnswerer, checked_ranking, parse_safety)
from paper_evidence_rag.guardrails import QueryGate, LABELS
from paper_evidence_rag.provider_status import ProviderUnavailable


def completion(text, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": text}}]}


def relevant():
    return QueryGate(lambda _: {"scores": {k: float(k == "allow") for k in LABELS}})


@pytest.mark.parametrize("response", [completion("User Safety: unsafe"), completion(None),
    completion("User Safety: safe", "length"), completion("unsafe"),
    completion("User Safety: safe\nUser Safety: unsafe"), {"choices": [None]}])
def test_safety_blocks_before_relevance_or_any_evidence_work(response):
    client = SimpleNamespace(request=Mock(return_value=response))
    relevance = Mock()
    decision = SafetyFirstGate(client, relevance).screen("What does this paper study?", "Paper", "Study")
    assert decision.action != "allow"
    relevance.screen.assert_not_called()
    assert client.request.call_count == 1


def test_safety_safe_passes_to_relevance_only_after_native_decision():
    calls = []
    class Client:
        def request(self, stage, payload):
            calls.append(stage)
            assert payload["model"] == SAFETY_MODEL and "response_format" not in payload
            return completion("User Safety: safe")
    class Relevance:
        def screen(self, *args):
            calls.append("relevance")
            return relevant().screen(*args)
    gate = SafetyFirstGate(Client(), Relevance())
    assert gate.screen("What is studied?", "Paper", "Study").action == "allow"
    assert calls == ["safety", "relevance"]
    assert gate.screen("", "Paper", "Study").status == "invalid_input"
    assert calls == ["safety", "relevance"]
    assert parse_safety("User Safety: unsafe\nSafety Categories: Illegal Activity") == "unsafe"


def test_safety_budget_failure_names_openrouter_and_stops_before_groq_or_search(tmp_path):
    from paper_evidence_rag.app_service import DemoService
    from test_workbench import cache
    root, req, gen, _, _ = cache.__wrapped__(tmp_path)
    class Client:
        def request(self, *args):
            raise ProviderUnavailable("safety")
    relevance = Mock()
    service = DemoService(root, req, gen, gate=SafetyFirstGate(Client(), relevance))
    try:
        service.viewer.retriever.search = Mock(side_effect=AssertionError("CPU search started"))
        result = service.ask("p", "new research question", "bm25", generate=True)
        assert "OpenRouter safety screening failed" in result["guardrail"]["reason"]
        assert "Groq was not called" in result["guardrail"]["reason"]
        assert "CPU search was not started" in result["guardrail"]["reason"]
        assert result["hits"] == []
        relevance.screen.assert_not_called()
    finally:
        service.close()


def test_transport_fixed_destination_zero_price_and_single_attempt():
    budget = SimpleNamespace(reserve=Mock())
    http = Mock()
    http.post.return_value.json.return_value = completion("User Safety: safe")
    client = OpenRouterClient("fixture-key", budget, http)
    client.request("safety", {"model": SAFETY_MODEL, "messages": []})
    url = http.post.call_args.args[0]
    body = http.post.call_args.kwargs["json"]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert body["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0}
    assert body["provider"]["only"] == ["nvidia"] and not body["provider"]["allow_fallbacks"]
    with pytest.raises(ValueError): client.request("generation", {"model": "paid/model"})
    with pytest.raises(ValueError): client.request("safety", {"model": SAFETY_MODEL, "plugins": []})
    http.post.side_effect = RuntimeError("untrusted provider diagnostic")
    with pytest.raises(RuntimeError, match="^Hosted model request unavailable$"):
        client.request("safety", {"model": SAFETY_MODEL})
    assert http.post.call_count == 2 and budget.reserve.call_count == 2


def test_provider_429_is_distinct_sanitized_and_has_bounded_retry_after():
    from paper_evidence_rag.cloud_pipeline import HostedRequestError
    http = Mock()
    response = http.post.return_value
    response.status_code = 429
    response.headers = {"retry-after": "15"}
    response.raise_for_status.side_effect = RuntimeError("secret=provider-body")
    client = OpenRouterClient("fixture-key", SimpleNamespace(reserve=Mock()), http)
    with pytest.raises(HostedRequestError) as caught:
        client.request("generation", {"model": GENERATION_MODELS[0]})
    assert caught.value.status_code == 429 and caught.value.stage == "generation"
    assert caught.value.retry_after == 15 and "secret" not in str(caught.value)
    response.headers = {"retry-after": "999999999"}
    with pytest.raises(HostedRequestError) as caught:
        client.request("rerank", {"model": RERANK_MODEL})
    assert caught.value.status_code == 429 and caught.value.retry_after is None


@pytest.mark.parametrize("stage", ["reranking", "generation"])
def test_cloud_failure_after_search_keeps_hits_and_names_failed_stage(tmp_path, stage):
    from test_workbench import cache
    from paper_evidence_rag.app_service import DemoService
    from paper_evidence_rag.cloud_pipeline import HostedRequestError
    root, req, gen, _, _ = cache.__wrapped__(tmp_path)
    class FailedCloud:
        def __call__(self, *args):
            raise HostedRequestError(429, stage, 11)
    service = DemoService(root, req, gen, gate=relevant(), cloud_answerer=FailedCloud())
    try:
        result = service.ask("p", "documents", "bm25", generate=True, execution_mode="cpu")
        assert result["hits"] and result["state"] == "retrieval_only"
        assert ("CPU search and reranking completed" if stage == "generation" else "CPU search completed") in result["notice"]
        assert "OpenRouter rate limit rejected " + stage in result["notice"]
        assert "11 seconds" in result["notice"] and "No fresh answer was produced" in result["notice"]
        assert "secret" not in result["notice"]
    finally:
        service.close()


def test_shared_request_budget_survives_restart_and_paces_stages(tmp_path):
    clock = [172800.0]
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
    path = tmp_path / "usage.sqlite"
    budget = PacedBudget(path, daily_limit=2, clock=lambda: clock[0], sleep=sleep)
    budget.reserve()
    budget.reserve()
    assert sleeps == [6.0]
    with pytest.raises(RuntimeError):
        PacedBudget(path, daily_limit=2, clock=lambda: clock[0], sleep=sleep).reserve()


def hits():
    return [{"rank": n + 1, "score": 1.0, "passage": {"paper_id": "p", "passage_id": pid,
        "kind": "paragraph", "section_name": "Methods", "text": text}} for n, (pid, text) in enumerate([
            ("p:b", "We studied 200 documents."), ("p:a", "A secondary result."), ("p:c", "Unrelated detail.")])]


def ranking():
    return {"results": [{"index": 2, "relevance_score": -1}, {"index": 0, "relevance_score": 2},
                        {"index": 1, "relevance_score": 2}]}


def test_rerank_indices_preserve_source_text_and_ties_use_stable_ids():
    original = hits()
    ranked = checked_ranking(ranking(), original)
    assert [h["passage"]["passage_id"] for h in ranked] == ["p:a", "p:b", "p:c"]
    assert original == hits()
    for rows in ([{"index": 0, "relevance_score": 1}] * 3,
                 [{"index": i, "relevance_score": float("nan")} for i in range(3)],
                 [{"index": i, "relevance_score": 1} for i in (0, 1, 9)], [None] * 3):
        with pytest.raises(ValueError): checked_ranking({"results": rows}, original)


def test_cloud_answerer_batches_reranking_and_preserves_malformed_draft():
    calls = []
    class Client:
        def request(self, stage, payload):
            calls.append((stage, payload))
            return ranking() if stage == "rerank" else completion("malformed draft", "length")
    result = CloudAnswerer(Client())("p", "How many documents?", hits(), "revision", "dense")
    assert [c[0] for c in calls] == ["rerank", "generation"]
    assert calls[0][1]["documents"] == [h["passage"]["text"] for h in hits()]
    assert calls[0][1]["top_n"] == 3 and calls[0][1]["model"] == RERANK_MODEL
    assert calls[1][1]["model"] == GENERATION_MODELS[0]
    assert calls[1][1]["max_tokens"] == 1024
    assert [m["role"] for m in calls[1][1]["messages"]] == ["system", "user"]
    assert result["raw_output"] == "malformed draft" and result["output_limit_reached"]
    assert len(result["presented"]) == 3
    evidence = json.loads(calls[1][1]["messages"][1]["content"])["evidence"]
    assert evidence[0]["kind"] == "paragraph" and evidence[0]["section_name"] == "Methods"
    assert all(e["text"] == p["text"] for e, p in zip(evidence, result["presented"], strict=True))
    assert result["answer_format"] == "hosted-grounded-v1" and result["prompt_version"]


def test_bad_rerank_never_reaches_generation():
    client = SimpleNamespace(request=Mock(return_value={"results": []}))
    from paper_evidence_rag.provider_status import PipelineStageFailure
    with pytest.raises(PipelineStageFailure, match="Hosted pipeline stage failed") as caught:
        CloudAnswerer(client)("p", "question", hits(), "rev", "dense")
    assert caught.value.stage == "reranking"
    assert client.request.call_count == 1


def test_cpu_mode_never_calls_gpu_and_rejects_mutated_rerank_sources(tmp_path):
    from test_workbench import cache
    from paper_evidence_rag.app_service import DemoService
    root, req, gen, _, _ = cache.__wrapped__(tmp_path)
    gpu = Mock(device="cuda")
    def cloud(paper, question, candidates, revision, mode):
        passage = candidates[0]["passage"]
        ranked = copy.deepcopy(candidates)
        ranked[0]["passage"]["text"] = "fabricated replacement"
        return {"raw_output": json.dumps({"abstain": False, "claims": [{"text": "200 documents", "passage_id": passage["passage_id"]}]}),
                "presented": [passage], "ranked_hits": ranked}
    service = DemoService(root, req, gen, gate=relevant(), answerer=gpu, cloud_answerer=cloud)
    try:
        result = service.ask("p", "documents", "bm25", generate=True, execution_mode="cpu")
        gpu.assert_not_called()
        assert result["state"] == "retrieval_only" and "answer" not in result
        assert result["hits"][0]["passage"]["text"] != "fabricated replacement"
    finally:
        service.close()


def test_ui_has_one_execution_control_and_no_manual_generation_checkbox(tmp_path):
    pytest.importorskip("gradio")
    from test_workbench import cache
    from paper_evidence_rag.app_service import DemoService
    from paper_evidence_rag.gradio_app import build_app
    root, req, gen, _, _ = cache.__wrapped__(tmp_path)
    service = DemoService(root, req, gen)
    try:
        app = build_app(service)
        labels = [c["props"].get("label", "") for c in app.config["components"]]
        assert labels.count("Use HF GPU") == 1
        assert "Generate a new cited answer" not in labels
        assert any("How to use this app" == label for label in labels)
    finally:
        service.close()
