import json

from paper_evidence_rag.hosted_answers import validate_hosted_draft


def sources():
    return [{"paper_id": "paper", "passage_id": "paper:p:1", "kind": "paragraph",
             "section_name": "Results", "text": "We tested 20 papers."}]


def raw(answer="We tested 20 papers.", claims=None, limitations=""):
    return json.dumps({"answer": answer, "claims": claims if claims is not None else [
        {"text": "20 papers", "passage_id": "paper:p:1"}], "limitations": limitations})


def test_invalid_quote_does_not_discard_valid_claim_or_readable_answer():
    value = validate_hosted_draft(raw(claims=[
        {"text": "20 papers", "passage_id": "paper:p:1"},
        {"text": "30 papers", "passage_id": "paper:p:1"}]), "paper", sources())
    assert value["status"] == "partial" and len(value["claims"]) == 1
    assert value["invalid_claim_count"] == 1 and value["answer"] == "We tested 20 papers."


def test_unknown_source_never_validates():
    value = validate_hosted_draft(raw(claims=[{"text": "20 papers", "passage_id": "other:p:1"}]), "paper", sources())
    assert value["status"] == "unverified" and value["claims"] == []


def test_no_claims_or_invalid_quotes_keep_explanation_and_answer():
    value = validate_hosted_draft(raw(claims=[], limitations="The passages do not give the sample size."), "paper", sources())
    assert value["status"] == "unverified" and value["answer"]
    assert "do not give" in value["limitations"]


def test_malformed_empty_and_truncated_outputs_are_unverified_not_blocked():
    for draft, limited in (("{\"answer\":", True), ("", False), ("not JSON", False)):
        value = validate_hosted_draft(draft, "paper", sources(), output_limit_reached=limited)
        assert value["status"] == "unverified" and value["answer"]


def test_duplicate_keys_rejected_without_fuzzy_repair():
    value = validate_hosted_draft('{"answer":"a","answer":"b","claims":[],"limitations":""}', "paper", sources())
    assert value["status"] == "unverified" and value["claims"] == []


def test_duplicate_quote_does_not_inflate_verified_claims():
    value = validate_hosted_draft(raw(claims=[{"text": "20 papers", "passage_id": "paper:p:1"}] * 2), "paper", sources())
    assert len(value["claims"]) == 1 and value["status"] == "partial"
    assert value["invalid_claim_reasons"] == ["duplicate_claim"]


def test_live_service_retains_partial_answer_and_ui_does_not_withhold(tmp_path):
    from test_workbench import cache
    from test_cloud_pipeline import relevant
    from paper_evidence_rag.app_service import DemoService
    from paper_evidence_rag.gradio_app import _result_text
    root, req, gen, _, _ = cache.__wrapped__(tmp_path)
    def answerer(paper, question, candidates, revision, mode):
        source = candidates[0]["passage"]
        return {"answer_format": "hosted-grounded-v1", "presented": [source], "raw_output": json.dumps({
            "answer": "They studied 200 documents.", "limitations": "", "claims": [
                {"text": "200 documents", "passage_id": source["passage_id"]},
                {"text": "300 documents", "passage_id": source["passage_id"]}]})}
    service = DemoService(root, req, gen, gate=relevant(), cloud_answerer=answerer)
    try:
        result = service.ask("p", "documents", "bm25", generate=True, execution_mode="cpu")
        assert result["answer"]["status"] == "partial" and len(result["answer"]["claims"]) == 1
        status, text = _result_text(result)
        assert text == "They studied 200 documents." and "withheld" not in status.lower()
        assert "Question allowed" in status
    finally:
        service.close()


def test_hosted_generation_keeps_relevant_text_beyond_old_prefix_limit():
    from paper_evidence_rag.cloud_pipeline import CloudAnswerer
    text = "Background detail. " * 100 + "Our contribution is a survey of event extraction."
    captured = []
    class Client:
        def request(self, stage, payload):
            if stage == "rerank":
                return {"results": [{"index": 0, "relevance_score": 1.0}]}
            captured.append(json.loads(payload["messages"][1]["content"]))
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "answer": "The paper surveys event extraction.", "limitations": "", "claims": [
                    {"text": "a survey of event extraction", "passage_id": "paper:p:1"}]})}}]}
    hit = {"rank": 1, "score": 0.5, "passage": {**sources()[0], "text": text}}
    result = CloudAnswerer(Client())("paper", "What is its contribution?", [hit], "revision", "dense")
    assert len(text) > 1200 and captured[0]["evidence"][0]["text"] == text
    value = validate_hosted_draft(result["raw_output"], "paper", result["presented"])
    assert value["status"] == "answered" and value["claims"][0]["text"] in text[1200:]
