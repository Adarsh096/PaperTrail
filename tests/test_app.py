"""App checks use tiny synthetic documents and fake providers, never model weights."""
import copy
import io
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("gradio")
pytest.importorskip("pypdf")

from paper_evidence_rag.app_service import DemoService
from paper_evidence_rag.answering import sealed
from paper_evidence_rag.guardrails import LABELS, QueryGate
from paper_evidence_rag.hosted_guardrail import CallBudget, GroqPredictor
from paper_evidence_rag.pdf_ingestion import CHUNK_CHARS, UploadedPaper, chunk_page, extract_pdf
from paper_evidence_rag.ingestion import build, canonical, digest, load_artifacts


@pytest.fixture
def app_inputs(tmp_path):
    # Reuse the public synthetic cache fixture, not downloaded research evidence.
    from test_workbench import cache
    return cache.__wrapped__(tmp_path)


def allow_gate():
    return QueryGate(lambda state: {"scores": {k: float(k == "allow") for k in LABELS}})


def synthetic_vectors(root, tmp_path):
    manifest, contents = load_artifacts(root)
    value = sealed({"corpus_revision": manifest["corpus_revision"],
        "model": {"id": "fixture", "revision": "a" * 40, "dimension": 2},
        "passages": {p["passage_id"]: [1., .5] for p in contents["passages.json"]},
        "queries": {digest(b"How many documents?"): [1., .5]}})
    path = tmp_path / "vectors.json"
    path.write_text(canonical(value))
    return path


def test_examples_dense_bm25_raw_and_gate_separation(app_inputs, tmp_path):
    root, req, gen, _, _ = app_inputs
    service = DemoService(root, req, gen, synthetic_vectors(root, tmp_path))
    try:
        dense = service.example("p", "How many documents?", "dense")
        assert dense["origin"] == "saved" and len(dense["hits"]) == 1
        assert dense["answer"]["answer"] == "200 documents"
        assert "diagnostic" not in dense
        assert "raw_output" in service.example("z", "How many documents?", include_raw=True)["diagnostic"]
        assert service.example("p", "How many documents?", "bm25")["method"] == "bm25"
        with pytest.raises(ValueError): service.example("p", "arbitrary new query")
        # Even exact saved text typed into the free-query endpoint must be gated.
        assert service.ask("p", "How many documents?")["state"] == "query_clarify"
        service.gate = allow_gate()
        assert service.ask("p", "documents", "bm25")["state"] == "retrieval_only"
        assert service.ask("p", "zzzzxyx", "bm25")["state"] == "no_evidence"
        with pytest.raises(ValueError): service.ask("missing", "documents")
        with pytest.raises(ValueError): service.passage("z", dense["hits"][0]["passage"]["passage_id"])
    finally:
        service.close()


def test_gate_blocks_before_embedding_retrieval_or_generation(app_inputs):
    root, req, gen, _, _ = app_inputs
    encoder, answerer = Mock(), Mock()
    service = DemoService(root, req, gen, encoder=encoder, answerer=answerer)
    try:
        service.viewer.retriever.search = Mock(side_effect=AssertionError("Retrieval bypassed gate"))
        for query in ("new question", "", "a" * 2001):
            assert service.ask("p", query, generate=True)["state"] in ("query_clarify", "query_reject")
        encoder.encode_question.assert_not_called()
        answerer.assert_not_called()
    finally:
        service.close()


def test_new_query_vector_is_temporary_and_source_cache_stays_sealed(app_inputs, tmp_path):
    root, req, gen, _, _ = app_inputs
    encoder = SimpleNamespace(encode_question=lambda q: [1., .5])
    service = DemoService(root, req, gen, synthetic_vectors(root, tmp_path), allow_gate(), encoder)
    try:
        original = copy.deepcopy(service.embeddings)
        assert service.ask("p", "new query")["hits"]
        assert service.embeddings == original
        service.dense.search = Mock(side_effect=RuntimeError("fixture failure"))
        with pytest.raises(RuntimeError): service.ask("p", "another query")
        assert service.embeddings == original
    finally:
        service.close()


def test_upload_isolation_cached_document_vectors_and_original_corpus(app_inputs):
    root, req, gen, _, _ = app_inputs
    p = {"passage_id": "upload-a:paragraph:0:0", "paper_id": "upload-a", "kind": "paragraph",
         "section_index": 0, "position": 0, "section_name": "Page 1", "text": "We studied 321 documents.", "source_file": "uploaded.pdf"}
    a = UploadedPaper("upload-a", "A", "Research paper", "rev-a", [p], 1)
    b = UploadedPaper("upload-b", "B", "Research paper", "rev-b", [{**p, "paper_id": "upload-b", "passage_id": "upload-b:paragraph:0:0"}], 1)
    encoder = SimpleNamespace(metadata={"id": "fixture", "revision": "a" * 40, "dimension": 2},
        encode_question=Mock(return_value=[1., .5]), encode_documents=Mock(return_value=[[1., .5]]),
        split_documents=Mock(side_effect=lambda texts: [[{"text": text, "start": 0, "end": len(text)}]
            for text in texts]))
    service = DemoService(root, req, gen, gate=allow_gate(), encoder=encoder)
    try:
        for query in ("documents", "study"):
            result = service.ask(a.paper_id, query, uploaded=a)
            assert result["hits"][0]["passage"]["paper_id"] == a.paper_id
        encoder.encode_documents.assert_called_once()
        assert b.dense_cache is None
        assert len(service.papers) == 2 and a.paper_id not in service.papers
        with pytest.raises(ValueError): service.passage(b.paper_id, p["passage_id"], b)
        assert service.ask(b.paper_id, "documents", "bm25", b)["hits"][0]["passage"]["paper_id"] == b.paper_id
    finally:
        service.close()


def test_dense_upload_indexes_token_fragments_with_exact_page_locations(app_inputs):
    root, req, gen, _, _ = app_inputs
    text = "alpha evidence. beta evidence."
    split_at = text.index("beta")
    passage = {"passage_id": "upload-fragment:paragraph:0:0", "paper_id": "upload-fragment",
        "kind": "paragraph", "section_index": 0, "position": 0, "section_name": "Page 1",
        "text": text, "source_file": "uploaded.pdf"}
    uploaded = UploadedPaper("upload-fragment", "Fixture", "Abstract", "revision-fragment",
        [passage], 1, locations={passage["passage_id"]: {"page": 1, "char_start": 100,
        "char_end": 100 + len(text), "offsets": "zero-based, end-exclusive in extracted page text"}})
    metadata = {"id": "fixture", "revision": "f" * 40, "dimension": 2}
    encoder = SimpleNamespace(metadata=metadata,
        split_documents=Mock(return_value=[[{"text": text[:split_at], "start": 0, "end": split_at},
            {"text": text[split_at:], "start": split_at, "end": len(text)}]]),
        encode_documents=Mock(return_value=[[1., 0.], [0., 1.]]),
        encode_question=Mock(return_value=[0., 1.]))
    service = DemoService(root, req, gen, gate=allow_gate(), encoder=encoder)
    try:
        result = service.ask(uploaded.paper_id, "beta evidence", "dense", uploaded=uploaded)
        assert result["hits"][0]["passage"]["passage_id"].endswith(":token:1")
        fragments = uploaded.dense_passages
        assert "".join(item["text"] for item in fragments) == text
        assert [item["passage_id"] for item in fragments] == [
            passage["passage_id"] + ":token:0", passage["passage_id"] + ":token:1"]
        assert uploaded.dense_locations[fragments[1]["passage_id"]]["char_start"] == 100 + split_at
        assert uploaded.dense_locations[fragments[1]["passage_id"]]["char_end"] == 100 + len(text)
        assert service.passage(uploaded.paper_id, fragments[1]["passage_id"], uploaded)["text"] == text[split_at:]
        assert service.browse(uploaded.paper_id, uploaded) == [passage]
        assert len(uploaded.dense_cache["passages"]) == 2
        encoder.split_documents.assert_called_once_with([text])
        encoder.encode_documents.assert_called_once_with([text[:split_at], text[split_at:]])
    finally:
        service.close()


def test_non_sample_upload_reaches_cloud_gate_retrieval_and_answerer(app_inputs):
    root, req, gen, _, _ = app_inputs
    passage = {"passage_id": "upload-x:paragraph:0:0", "paper_id": "upload-x", "kind": "paragraph",
        "section_index": 0, "position": 0, "section_name": "Page 1", "text": "The arXiv paper evaluates XGBoost."}
    uploaded = UploadedPaper("upload-x", "XGBoost paper", "Research paper", "rev-x", [passage], 1)
    gate = allow_gate()
    gate.requires_public_data = True
    gate_calls = Mock(wraps=gate.screen)
    gate.screen = gate_calls
    def hosted_answer(paper_id, question, hits, revision, mode):
        source = hits[0]["passage"]
        return {"raw_output": json.dumps({"answer": "The paper discusses XGBoost.",
            "limitations": "", "claims": [{"text": "XGBoost", "passage_id": source["passage_id"]}]}),
            "answer_format": "hosted-grounded-v1", "presented": [source]}
    cloud = Mock(side_effect=hosted_answer)
    cloud.requires_public_data = True
    service = DemoService(root, req, gen, gate=gate, cloud_answerer=cloud)
    try:
        result = service.ask(uploaded.paper_id, "What does the paper say about XGBoost?", "bm25",
            uploaded=uploaded, generate=True, execution_mode="cpu")
        gate_calls.assert_called_once()
        cloud.assert_called_once()
        assert result["state"] == "live_answer" and result["answer"]["status"] == "answered"
        assert result["hits"][0]["passage"]["paper_id"] == uploaded.paper_id
        assert all(hit["passage"]["paper_id"] == uploaded.paper_id for hit in result["hits"])
        assert uploaded.public_source is False
    finally:
        service.close()


def test_live_draft_revalidated_against_only_supplied_prefixes(app_inputs):
    root, req, gen, _, _ = app_inputs
    def answerer(paper, question, hits, revision, mode):
        return {"raw_output": json.dumps({"abstain": False, "claims": [{"text": "200 documents", "passage_id": hits[0]["passage"]["passage_id"]}]}),
                "presented": [hits[0]["passage"]]}
    service = DemoService(root, req, gen, gate=allow_gate(), answerer=answerer)
    try:
        result = service.ask("p", "documents", "bm25", generate=True)
        assert result["state"] == "live_answer" and result["answer"]["status"] == "answered"
        result["presented"][0]["text"] = "fabricated"
        with pytest.raises(ValueError): service._checked_live(result | {"raw_output": result["_raw"]}, "p", service.ask("p", "documents", "bm25")["hits"])
        service.answerer = lambda *args: {"raw_output": "malformed draft", "presented": [args[2][0]["passage"]]}
        bad = service.ask("p", "documents", "bm25", generate=True)
        assert bad["answer"]["reason"] == "invalid_schema" and bad["_raw"] == "malformed draft"
        service.answerer = Mock(side_effect=TimeoutError())
        assert service.ask("p", "documents", "bm25", generate=True)["state"] == "retrieval_only"
        service.answerer = lambda *args: {"raw_output": "malformed provider result"}
        malformed = service.ask("p", "documents", "bm25", generate=True)
        assert malformed["state"] == "retrieval_only" and malformed["hits"]
        assert "CPU search completed" in malformed["notice"]
        assert "No fresh answer was produced" in malformed["notice"]
        assert "Retrieved sources remain available" in malformed["notice"]
    finally:
        service.close()


def pdf_bytes(text="We studied 200 documents.", pages=1, encrypted=False):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(("BT /F1 12 Tf 40 740 Td (" + text + ") Tj ET").encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    if encrypted:
        writer.encrypt("fixture-password")
    out = io.BytesIO(); writer.write(out)
    return out.getvalue()


def test_pdf_bounds_stable_locations_and_no_ocr():
    data = pdf_bytes()
    first, second = extract_pdf(data), extract_pdf(data)
    assert first == second
    assert first.locations[first.passages[0]["passage_id"]]["page"] == 1
    assert first.passages[0]["text"] == "We studied 200 documents."
    for invalid in (b"not pdf", b"%PDF-" + b"x" * (5 * 1024**2), pdf_bytes(encrypted=True), pdf_bytes(pages=21), pdf_bytes(text="")):
        with pytest.raises(ValueError): extract_pdf(invalid)


def test_paragraph_sentence_overlap_exact_coverage_and_size():
    text = "First paragraph explains a method. " * 30 + "\n\n" + "Second paragraph evaluates the method. " * 80
    chunks = list(chunk_page(text))
    assert len(chunks) > 1 and chunks == list(chunk_page(text))
    assert chunks[0][0] == 0 and chunks[-1][1] == len(text)
    for index, (start, end, value) in enumerate(chunks):
        assert value == text[start:end] and len(value) <= CHUNK_CHARS
        if index:
            assert chunks[index - 1][0] < start <= chunks[index - 1][1]
    assert list(chunk_page(" \n\t")) == []


def test_budget_restart_spacing_daily_limit_and_atomic_reservations(tmp_path):
    clock = [172800.0]
    path = tmp_path / "budget.sqlite"
    budget = CallBudget(path, daily_limit=2, clock=lambda: clock[0])
    budget.reserve()
    with pytest.raises(RuntimeError): budget.reserve()
    clock[0] += 6
    CallBudget(path, daily_limit=2, clock=lambda: clock[0]).reserve()
    clock[0] += 6
    with pytest.raises(RuntimeError): budget.reserve()
    clock[0] += 86400
    def reserve(_):
        try:
            budget.reserve()
            return True
        except RuntimeError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(8))) == 1


def test_provider_none_content_rate_error_and_truncation_fail_closed(tmp_path):
    for finish, content in (("stop", None), ("length", '{"scores":{}}'), ("stop", "bad json")):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))])))))
        predictor = GroqPredictor(client, SimpleNamespace(reserve=lambda: None))
        decision = QueryGate(predictor).screen("question", "title", "abstract")
        assert decision.action == "clarify" and decision.status == "malformed_output"
    predictor.budget = SimpleNamespace(reserve=Mock(side_effect=RuntimeError("exhausted")))
    client.chat.completions.create.reset_mock()
    assert QueryGate(predictor).screen("question", "title", "abstract").action == "clarify"
    client.chat.completions.create.assert_not_called()


def test_guardrail_budget_and_provider_rate_limit_are_distinct_and_short_circuit(tmp_path):
    from paper_evidence_rag.provider_status import LocalCallLimit
    from paper_evidence_rag.app_service import DemoService
    root, req, gen, _, _ = __import__("test_workbench").cache.__wrapped__(tmp_path)
    calls = Mock()
    predictor = GroqPredictor(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=calls))),
        SimpleNamespace(reserve=Mock(side_effect=LocalCallLimit("daily"))))
    service = DemoService(root, req, gen, gate=QueryGate(predictor))
    try:
        service.viewer.retriever.search = Mock(side_effect=AssertionError("Search passed a failed gate"))
        result = service.ask("p", "new research question", "bm25", generate=True)
        reason = result["guardrail"]["reason"]
        assert "daily" in reason and "no Groq screening request was sent" in reason
        assert "CPU search and generation were not started" in reason
        calls.assert_not_called()
        predictor.budget.reserve.side_effect = None
        # A status-bearing provider failure is classified without exposing its text.
        class RateError(Exception):
            status_code = 429
            response = SimpleNamespace(status_code=429, headers={"retry-after": "12"})
        predictor.client.chat.completions.create.side_effect = RateError("secret=hidden")
        decision = QueryGate(predictor).screen("new research question", "title", "abstract")
        assert decision.category == "provider_rate_limit"
        assert "rejected the screening request" in decision.reason and "12 seconds" in decision.reason
        assert "secret" not in decision.reason and "CPU search and generation were not started" in decision.reason
        assert "No request" not in decision.reason
    finally:
        service.close()


def test_gradio_configuration_is_bounded_and_saved_examples_are_allowlisted(app_inputs):
    from paper_evidence_rag.gradio_app import build_app, launch_options
    root, req, gen, _, _ = app_inputs
    service = DemoService(root, req, gen)
    try:
        app = build_app(service)
        api_names = {event["api_name"] for event in app.config["dependencies"]}
        assert {"open_example", "ask_paper", "process_pdf", "ask_upload", "remove_upload"} <= api_names
        assert {"select_question_suggestion", "select_paper"} <= api_names
        ask_event = next(event for event in app.config["dependencies"] if event["api_name"] == "ask_paper")
        assert len(ask_event["inputs"]) == 4
        paper_event = next(event for event in app.config["dependencies"] if event["api_name"] == "select_paper")
        assert len(paper_event["outputs"]) >= 4
        options = launch_options(service)
        assert options["share"] is False and options["server_name"] == "127.0.0.1"
        assert options["run_history"] is False and options["max_file_size"] == "5mb"
        assert options["state_session_capacity"] == 32
        assert options["blocked_paths"]
        from paper_evidence_rag.gradio_app import THEME_JS, TOGGLE_JS
        assert "body.classList.toggle('dark', dark)" in THEME_JS
        assert "body.classList.toggle('dark', dark)" in TOGGLE_JS
    finally:
        service.close()
