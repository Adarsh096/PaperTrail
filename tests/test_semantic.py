import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from test_workbench import cache
from paper_evidence_rag.answering import sealed
from paper_evidence_rag.ingestion import load_artifacts
from paper_evidence_rag.semantic_evaluation import METRICS, aggregate_semantic, metric_jobs, prepare_semantic, validate_semantic
from paper_evidence_rag.semantic_embeddings import embedding_requests, validate_vectors


def test_semantic_freeze_replays_sources_and_keeps_rejected_text_separate(cache):
    root, _, _, request, generated = cache
    manifest, contents = load_artifacts(root)
    result = prepare_semantic(contents, manifest, request, generated, "fixture")
    validate_semantic(result)
    assert result["rows"][0]["accepted_response"] == "200 documents"
    assert result["rows"][1]["accepted_response"] is None
    assert result["rows"][1]["diagnostic_response"] is None
    assert "<img" in result["rows"][1]["raw_output"]
    assert result["rows"][0]["contexts"][0]["text"] == request["rows"][0]["passages"][0]["text"]
    modified = copy.deepcopy(generated)
    modified["rows"][0]["presented"][0]["text"] = "fabricated"
    modified = sealed({k: v for k, v in modified.items() if k != "sha256"})
    with pytest.raises(ValueError): prepare_semantic(contents, manifest, request, modified, "fixture")


def test_reference_alternatives_exclusions_and_missing_answers(cache):
    root, _, _, request, generated = cache
    manifest, contents = load_artifacts(root)
    row = prepare_semantic(contents, manifest, request, generated, "fixture")["rows"][0]
    row["references"] = ["200 documents", "two hundred documents", "200 documents"]
    jobs = {j["metric"]: j for j in metric_jobs(row, "accepted")}
    assert len(jobs["factual_correctness"]["inputs"]) == 2
    assert jobs["answer_relevancy"]["inputs"] == [{"user_input": row["question"], "response": "200 documents"}]
    row["accepted_response"] = None
    assert next(j for j in metric_jobs(row, "accepted") if j["metric"] == "faithfulness")["not_applicable"] == "no_response"
    assert all(j["not_applicable"] == "no_diagnostic_response" for j in metric_jobs(row, "diagnostic"))
    row["excluded_reason"] = "unanswerable"
    assert next(j for j in metric_jobs(row, "accepted") if j["metric"] == "context_recall")["not_applicable"] == "unanswerable"


def test_denominators_failures_and_negative_cosine():
    def row(reason, status, value):
        return {"excluded_reason": reason, "scores": {track: {metric: {"status": status,
            "value": value if metric != "answer_relevancy" else -0.2} for metric in METRICS}
            for track in ("accepted", "diagnostic")}}
    result = aggregate_semantic([row(None, "scored", 0.5), row(None, "failed", None), row("unanswerable", "scored", 1)])
    assert result["accepted/eligible"]["faithfulness"] == {"mean": 0.5, "scored_count": 1,
        "question_count": 2, "status_counts": {"scored": 1, "failed": 1}}
    assert result["accepted/eligible"]["answer_relevancy"]["mean"] == -0.2
    with pytest.raises(ValueError): aggregate_semantic([row(None, "scored", float("nan"))])


def test_embeddings_bound_coverage_model_and_truncation():
    request = embedding_requests(["Question?", "Question?"])
    key = next(iter(request["texts"]))
    artifact = sealed({"requests": request, "vectors": {key: [1.0] + [0.0] * 767}, "truncated": []})
    validate_vectors(artifact)
    artifact["truncated"] = [key]
    artifact = sealed({k: v for k, v in artifact.items() if k != "sha256"})
    with pytest.raises(ValueError): validate_vectors(artifact)
    with pytest.raises(ValueError): embedding_requests([""])


def test_judge_adapter_cache_budget_and_malformed_response(tmp_path):
    pytest.importorskip("ragas")
    from pydantic import BaseModel
    from paper_evidence_rag.semantic_judge import JudgeFailure, JudgeLedger, JudgeLimit, make_judge
    class Verdict(BaseModel):
        verdict: bool
    class Client:
        def __init__(self, text):
            self.text, self.calls = text, 0
            self.chat = SimpleNamespace(completions=self)
        async def create(self, **kwargs):
            self.calls += 1
            self.options = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15), model="fixture")
    config = {"model": "fixture", "max_output_tokens": 100}
    ledger = JudgeLedger(tmp_path/"ledger.json", config, max_calls=1)
    client = Client('{"verdict":true}')
    judge = make_judge(client, ledger)
    assert asyncio.run(judge.agenerate("check", Verdict)).verdict
    assert asyncio.run(judge.agenerate("check", Verdict)).verdict and client.calls == 1
    with pytest.raises(JudgeLimit): asyncio.run(judge.agenerate("different", Verdict))
    replay = make_judge(None, JudgeLedger(tmp_path/"ledger.json", config, offline=True))
    assert asyncio.run(replay.agenerate("check", Verdict)).verdict
    with pytest.raises(JudgeLimit): asyncio.run(replay.agenerate("missing", Verdict))
    assert ledger.usage()["reported_prompt_tokens"] == 12
    bad = make_judge(Client("not json"), JudgeLedger(tmp_path/"bad.json", config))
    with pytest.raises(JudgeFailure): asyncio.run(bad.agenerate("bad", Verdict))
    assert "not json" in (tmp_path/"bad.json").read_text()
    assert "api_key" not in (tmp_path/"ledger.json").read_text()
    tampered = json.loads((tmp_path/"ledger.json").read_text())
    tampered["calls"] = []
    (tmp_path/"ledger.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError): JudgeLedger(tmp_path/"ledger.json", config)
    with pytest.raises(ValueError): JudgeLedger(tmp_path/"new.json", config, max_calls=0)
    native_client = Client('{"verdict":true}')
    native = make_judge(native_client, JudgeLedger(tmp_path/'native.json',
        {**config, 'structured_output':'native-json-schema'}))
    assert asyncio.run(native.agenerate('native', Verdict)).verdict
    schema = native_client.options['response_format']['json_schema']
    assert schema['strict'] and schema['schema']['additionalProperties'] is False
    assert schema['schema']['required'] == ['verdict']


def test_pending_relevance_embeddings_and_vector_identity(tmp_path):
    pytest.importorskip("ragas")
    from paper_evidence_rag.semantic_embeddings import cached_embeddings
    from paper_evidence_rag.ingestion import digest
    collector = cached_embeddings()
    asyncio.run(collector.aembed_text("Original question?"))
    asyncio.run(collector.aembed_texts(["Reverse question?", "Original question?"]))
    request = embedding_requests(collector.pending)
    assert len(request["texts"]) == 2
    artifact = sealed({"requests": request, "vectors": {k: [1.0]+[0.0]*767 for k in request["texts"]}, "truncated": []})
    frozen = cached_embeddings(artifact)
    assert asyncio.run(frozen.aembed_text("Original question?")) == artifact["vectors"][digest(b"Original question?")]
    assert not frozen.pending
    asyncio.run(frozen.aembed_text('Missing question?'))
    misses = frozen.missing_calls
    asyncio.run(frozen.aembed_text('Original question?'))
    assert frozen.missing_calls == misses and frozen.pending == {'Missing question?'}
