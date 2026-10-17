import pytest

from paper_evidence_rag.evaluation_hardening import AnswerVerdict, summarize_verdict, verdict_prompt


def verdict(**changes):
    data = dict(relevant=True, facts=[
        dict(reference_quote="12 samples", description="sample count", answered=True, supporting_ids=["count"]),
        dict(reference_quote="Paris", description="location", answered=True, supporting_ids=["place"])],
        contexts=[dict(passage_id="count", useful=True), dict(passage_id="place", useful=True)],
        incorrect_response_quotes=[], reason="Both required facts are present.")
    data.update(changes)
    return AnswerVerdict.model_validate(data)


CONTEXTS = [dict(passage_id="count", text="12 samples"), dict(passage_id="place", text="Paris")]


def test_complementary_contexts_and_fragment():
    result = summarize_verdict(verdict(), "12 samples in Paris", "12 samples in Paris", CONTEXTS)
    assert result["correctness"] == "correct"
    assert result["completeness"] == result["context_usefulness_ap"] == 1


def test_partial_answer_has_full_available_evidence():
    value = verdict()
    value.facts[1].answered = False
    result = summarize_verdict(value, "12 samples", "12 samples in Paris", CONTEXTS)
    assert result["correctness"] == "partial"
    assert result["completeness"] == .5
    assert result["available_fact_fraction"] == 1


def test_contradiction_is_incorrect_despite_complete_answer():
    result = summarize_verdict(verdict(incorrect_response_quotes=["99"]),
                               "12 samples in Paris and 99 samples", "12 samples in Paris", CONTEXTS)
    assert result["correctness"] == "incorrect"


@pytest.mark.parametrize("change", [
    dict(incorrect_response_quotes=["invented"]),
    dict(contexts=[dict(passage_id="place", useful=True), dict(passage_id="count", useful=True)]),
    dict(facts=[]),
])
def test_invalid_verdict_is_explicit(change):
    with pytest.raises(ValueError):
        summarize_verdict(verdict(**change), "12 samples in Paris", "12 samples in Paris", CONTEXTS)


def test_context_ap_counts_rank_and_partial_support():
    value = verdict()
    value.contexts.insert(0, dict_to_context("noise", False))
    contexts = [dict(passage_id="noise", text="unrelated"), *CONTEXTS]
    result = summarize_verdict(value, "12 samples in Paris", "12 samples in Paris", contexts)
    assert result["context_usefulness_ap"] == pytest.approx((.5 + 2 / 3) / 2)


def dict_to_context(pid, useful):
    from paper_evidence_rag.evaluation_hardening import ContextVerdict
    return ContextVerdict(passage_id=pid, useful=useful)


def test_prompt_keeps_original_fragment_and_has_no_expected_labels():
    prompt = verdict_prompt("Which dataset?", "GeoQA dataset", "GeoQA dataset", CONTEXTS)
    assert '"response": "GeoQA dataset"' in prompt
    assert "expected" not in prompt


def test_evidence_trace_keeps_mapping_failures_and_packing_stages():
    from paper_evidence_rag.evaluation_hardening import trace_evidence
    row = {"contexts": [{"passage_id": "packed", "text": "first"}],
           "validation": {"status": "abstained"}, "evidence_mappings": [
        {"annotation_id": "a", "match": "exact", "text": "first then second",
         "passage_ids": ["absent", "dropped", "packed"]},
        {"annotation_id": "b", "match": "unmatched", "text": "unknown", "passage_ids": []}]}
    result = trace_evidence(row, [{"passage": {"passage_id": p}} for p in ("dropped", "packed")],
                            [{"passage_id": "packed", "text": "first then second"}])
    assert [p["stage"] for p in result["evidence"][0]["passages"]] == [
        "absent_from_candidates", "not_selected", "evidence_paragraph_truncated"]
    assert result["evidence"][1]["status"] == "unmapped"
    row["contexts"][0]["text"] = "fabricated"
    with pytest.raises(ValueError, match="prefix"):
        trace_evidence(row, [], [{"passage_id": "packed", "text": "first then second"}])


def test_frozen_population_exclusions_and_alternative_references(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from paper_evidence_rag.answering import sealed
    from paper_evidence_rag.evaluation_hardening import evaluate_frozen
    from paper_evidence_rag import semantic_judge, evaluation_hardening
    calls = []

    class FakeJudge:
        async def agenerate(self, prompt, model):
            calls.append(prompt)
            return verdict()

    monkeypatch.setattr(semantic_judge, "make_judge", lambda client, ledger: FakeJudge())
    monkeypatch.setattr(evaluation_hardening, "version", lambda name: "fixture")
    common = dict(paper_id="paper", question="How many and where?", contexts=CONTEXTS,
                  references=["12 samples in Paris", "12 samples in Paris"],
                  accepted_response="12 samples in Paris", diagnostic_response=None)
    rows = [{**common, "question_id": "1", "excluded_reason": None},
            {**common, "question_id": "2", "excluded_reason": None, "accepted_response": None},
            {**common, "question_id": "3", "excluded_reason": "unanswerable"},
            {**common, "question_id": "4", "excluded_reason": "incomplete_evidence_mapping"},
            {**common, "question_id": "5", "excluded_reason": "unanswerable", "accepted_response": None}]
    inputs = sealed({"protocol": "ragas-debug-v1", "rows": rows})
    report = asyncio.run(evaluate_frozen(inputs, SimpleNamespace(config={}, usage=lambda: {}), None, "fixture"))
    assert len(report["rows"]) == 5
    assert len(calls) == 2  # Duplicate references deduplicated; excluded incomplete row is diagnostic scoring.
    assert report["aggregate"]["accepted"]["eligible_count"] == 2
    assert report["aggregate"]["accepted"]["scored_count"] == 1
    assert report["rows"][2]["scores"]["accepted"]["status"] == "unanswerable_response"
    assert report["rows"][4]["scores"]["accepted"]["status"] == "not_applicable"
