import copy
import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from paper_evidence_rag.answer_batches import merge_generated, split_requests
from paper_evidence_rag.answering import DEFAULT_MODEL_ID, MODEL_PINS, PROMPT_VERSION, prompt_hash, sealed, validate_draft
from paper_evidence_rag.ingestion import build, canonical, digest, load_artifacts
from paper_evidence_rag.workbench import Workbench, make_server
from paper_evidence_rag.runtime import configure


@pytest.fixture
def cache(tmp_path):
    text = 'We studied 200 documents. <script>alert("source")</script>'
    raw = {pid: {"title": f"Study {pid}", "abstract": text, "full_text": [], "figures_and_tables": [],
        "qas": [{"question_id": "q", "question": "How many documents?", "answers": [{"annotation_id": "a",
            "answer": {"unanswerable": False, "yes_no": None, "extractive_spans": ["200 documents"],
                       "free_form_answer": "", "evidence": [text]}}]}]} for pid in ("p", "z")}
    source = tmp_path / "source.json"; source.write_text(canonical(raw))
    artifact = tmp_path / "build"; manifest = build(source, artifact, digest(source.read_bytes()))
    _, contents = load_artifacts(artifact)
    rows = [{"paper_id": p["paper_id"], "question_id": "q", "question": "How many documents?", "passages": [p]}
            for p in contents["passages.json"]]
    requests = sealed({"corpus_revision": manifest["corpus_revision"], "input_sha256": manifest["input_sha256"],
        "candidate_sha256": "c", "scores_sha256": "s", "mode": "dense", "prompt_version": PROMPT_VERSION, "rows": rows})
    outputs = []
    for row in rows:
        draft = (json.dumps({"abstain": False, "claims": [{"text": "200 documents", "passage_id": row["passages"][0]["passage_id"]}]})
                 if row["paper_id"] == "p" else '<img src=x onerror="alert(1)"> malformed draft')
        outputs.append({k: row[k] for k in ("paper_id", "question_id", "question")} | {
            "presented": row["passages"], "raw_output": draft, "output_limit_reached": False,
            "result": validate_draft(draft, row["paper_id"], row["passages"])})
    generated = sealed({"request_sha256": requests["sha256"], "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_hash(), "model": {"id": DEFAULT_MODEL_ID, "revision": MODEL_PINS[DEFAULT_MODEL_ID]}, "rows": outputs})
    req, gen = tmp_path / "requests.json", tmp_path / "generated.json"
    req.write_text(canonical(requests)); gen.write_text(canonical(generated))
    return artifact, req, gen, requests, generated


def test_query_gate_blocks_retrieval_and_saved_answer(cache):
    from paper_evidence_rag.guardrails import LABELS, QueryGate
    root, req, gen, _, _ = cache
    gate = QueryGate(lambda state: {"scores": {k: float(k == "override") for k in LABELS}})
    w = Workbench(root, req, gen, query_gate=gate)
    try:
        def fail(*args):
            pytest.fail("Rejected query reached retrieval")
        w.retriever.search = fail
        assert w.query("p", "new question")["state"] == "query_reject"
        assert w.query("p", "How many documents?")["state"] == "query_reject"
        w.query_gate = QueryGate(None)
        assert w.query("p", "new question")["state"] == "query_clarify"
        w.query_gate = QueryGate(lambda state: {"scores": {k: float(k == "allow") for k in LABELS}})
        assert w.query("p", "How many documents?")["state"] == "saved_answer"
    finally:
        w.close()


def test_cache_replay_privacy_raw_opt_in_and_paper_isolation(cache):
    root, req, gen, _, _ = cache
    w = Workbench(root, req, gen)
    try:
        catalog = canonical(w.catalog())
        assert "annotations" not in catalog and "evidence" not in catalog
        good = w.query("p", "How many documents?")
        assert good["answer"]["answer"] == "200 documents" and "diagnostic" not in good
        assert w.passage(good["answer"]["claims"][0]["passage_id"])["passage"]["paper_id"] == "p"
        bad = w.query("z", "How many documents?", True)
        assert bad["answer"]["reason"] == "invalid_schema"
        assert bad["diagnostic"]["raw_output"].startswith("<img")
        assert w.query("p", "documents")["state"] == "needs_generation"
        assert w.query("p", "zzzzxyxyxy")["state"] == "no_evidence"
        with pytest.raises(LookupError): w.query("missing", "documents")
        with pytest.raises(LookupError): w.passage("../../secrets")
        with pytest.raises(ValueError): w.query("p", " ")
        with pytest.raises(ValueError): w.query("p", "a" * 2001)
    finally:
        w.close()


@pytest.mark.parametrize("mutation", ["corpus", "model", "source", "validation"])
def test_stale_or_modified_cache_rejected(cache, mutation):
    root, req, gen, request, output = cache
    if mutation == "corpus": request["corpus_revision"] = "stale"
    if mutation == "model": output["model"]["revision"] = "0" * 40
    if mutation == "source": output["rows"][0]["presented"][0]["text"] = "fabricated"
    if mutation == "validation": output["rows"][0]["result"]["answer"] = "fabricated"
    request = sealed({k: v for k, v in request.items() if k != "sha256"})
    output["request_sha256"] = request["sha256"]
    output = sealed({k: v for k, v in output.items() if k != "sha256"})
    req.write_text(canonical(request)); gen.write_text(canonical(output))
    with pytest.raises(ValueError): Workbench(root, req, gen)


def test_generation_partition_identity_and_missing_or_incompatible_outputs(cache):
    _, _, _, request, output = cache
    batches = split_requests(request, 1)
    results = [sealed({**{k: v for k, v in output.items() if k not in {"sha256", "rows", "request_sha256"}},
        "request_sha256": b["sha256"], "rows": output["rows"][i:i + 1]}) for i, b in enumerate(batches)]
    assert merge_generated(request, batches, results) == output
    with pytest.raises(ValueError, match="cover"): merge_generated(request, batches[:1], results[:1])
    with pytest.raises(ValueError): merge_generated(request, batches[::-1], results)
    other = copy.deepcopy(results); other[1]["model"] = {**other[1]["model"], "revision": "different"}
    other[1] = sealed({k: v for k, v in other[1].items() if k != "sha256"})
    with pytest.raises(ValueError, match="runtime"): merge_generated(request, batches, other)
    with pytest.raises(ValueError): split_requests(request, 7)


def test_bounded_http_routes_origin_and_raw_display(cache):
    root, req, gen, _, _ = cache
    workbench = Workbench(root, req, gen); server = make_server(workbench, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    def call(path, value=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        connection.request("GET" if value is None else "POST", path,
            body=None if value is None else json.dumps(value),
            headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse(); body = response.read(); connection.close()
        return response.status, response.headers, body
    try:
        status, headers, body = call("/api/catalog")
        assert status == 200 and headers["Cache-Control"] == "no-store"
        query = {"paper_id": "z", "question": "How many documents?"}
        assert "diagnostic" not in json.loads(call("/api/query", query)[2])
        raw = json.loads(call("/api/query", {**query, "include_raw": True})[2])
        assert raw["diagnostic"]["raw_output"].startswith("<img")
        assert call("/api/query", query, {"Origin": "https://foreign.example"})[0] == 403
        assert call("/api/catalog", headers={"Host": "foreign.example"})[0] == 403
        assert call("/api/query", {**query, "question": "a" * 9000})[0] == 413
        assert call("/api/query", {**query, "include_raw": "true"})[0] == 400
        assert call("/../../README.md")[0] == 404
        assert "script-src 'self'" in call("/")[1]["Content-Security-Policy"]
    finally:
        server.shutdown(); worker.join(5); server.server_close(); workbench.close()


def test_expanded_manifest_rebuild_and_original_default(cache, tmp_path):
    root, _, _, _, _ = cache
    source = tmp_path / "source.json"; raw = json.loads(source.read_text())
    raw.update({f"x{i}": copy.deepcopy(raw["p"]) for i in range(9)})
    source.write_text(canonical(raw)); checksum = digest(source.read_bytes())
    one, two = tmp_path / "one", tmp_path / "two"
    m1 = build(source, one, checksum, paper_limit=10); m2 = build(source, two, checksum, paper_limit=10)
    assert m1 == m2 and m1["paper_count"] == 10
    assert {p.name: p.read_bytes() for p in one.iterdir()} == {p.name: p.read_bytes() for p in two.iterdir()}
    assert build(source, root, checksum)["paper_count"] == 5
    assert m1["split"] == "train" and "10 training" in m1["selection"]
    with pytest.raises(ValueError): build(source, root, checksum, paper_limit=21)


def test_explicit_gpu_request_cannot_silently_use_cpu():
    unavailable = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    with pytest.raises(ValueError, match="no CPU fallback"): configure(unavailable, "cuda")
    with pytest.raises(ValueError, match="device"): configure(unavailable, "automatic")
