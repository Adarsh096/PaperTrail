import copy
import math

import pytest

from paper_evidence_rag.ingestion import canonical, digest
from paper_evidence_rag.reranking import assess, freeze_candidates, pair_key, rerank, MODEL_PINS, validate_model_pin, candidate_replay_matches


def hashed(value):
    value["sha256"] = digest(canonical({k: v for k, v in value.items() if k != "sha256"}).encode())
    return value


def test_model_allowlist_requires_exact_revision():
    for model, revision in MODEL_PINS.items():
        validate_model_pin({"id": model, "revision": revision})
        with pytest.raises(ValueError, match="revision mismatch"):
            validate_model_pin({"id": model, "revision": "main"})
    with pytest.raises(ValueError): validate_model_pin({"id": "unregistered", "revision": "a" * 40})


@pytest.fixture
def experiment():
    passages = [{"passage_id": f"p:paragraph:0:{i}", "paper_id": "p", "kind": "paragraph",
                 "position": i, "section_index": 0, "section_name": "Method", "text": f"zebra {i}"}
                for i in range(12)]
    annotation = {"annotation_id": "a", "unanswerable": False, "answer_type": "extractive",
        "evidence": ["zebra 11"], "extractive_spans": ["11"], "free_form_answer": "", "yes_no": None}
    contents = {"passages.json": passages, "questions.json": [{"paper_id": "p", "question_id": "q",
                "question": "zebra", "annotations": [annotation]}]}
    manifest = {"paper_ids": ["p"], "corpus_revision": "r", "input_sha256": "input"}
    embeddings = hashed({"corpus_revision": "r", "model": {"id": "fixture", "revision": "a" * 40,
        "dimension": 2}, "passages": {p["passage_id"]: [0, 1] for p in passages},
        "queries": {digest(b"zebra"): [1, 0]}, "truncated": {"passages": [], "queries": []}})
    return contents, manifest, embeddings


def test_rerank_logits_ties_and_bad_scores():
    hits = [{"passage": {"passage_id": pid}, "rank": i, "score": 5.0}
            for i, pid in enumerate(("b", "a", "c"), 1)]
    scores = {pair_key("q", pid): value for pid, value in (("a", -2), ("b", -2), ("c", 3))}
    result = rerank(hits, "q", scores)
    assert [h["passage"]["passage_id"] for h in result] == ["c", "a", "b"]
    assert [h["rank"] for h in result] == [1, 2, 3]
    scores[pair_key("q", "a")] = math.nan
    with pytest.raises(ValueError, match="finite"): rerank(hits, "q", scores)
    with pytest.raises(ValueError, match="Duplicate"): rerank(hits + hits[:1], "q", scores)


def test_freeze_is_label_free_and_deterministic(experiment):
    contents, manifest, embeddings = experiment
    first, _ = freeze_candidates(contents, manifest, embeddings, 10)
    other = copy.deepcopy(contents)
    other["questions.json"][0]["annotations"][0]["evidence"] = ["private answer"]
    second, _ = freeze_candidates(other, manifest, embeddings, 10)
    assert first == second
    assert "private answer" not in canonical(first)
    assert "annotations" not in canonical(first)
    with pytest.raises(ValueError): freeze_candidates(contents, manifest, embeddings, 9)


def test_portable_dense_replay_keeps_exact_ranks_text_and_hashes(experiment):
    contents, manifest, embeddings = experiment
    frozen, _ = freeze_candidates(contents, manifest, embeddings, 10, ('dense',))
    assert frozen['selection_rule'].endswith('1 training paper')
    portable = copy.deepcopy(frozen)
    portable['rows'][0]['candidates'][0]['score'] += 5e-7
    hashed(portable)
    assert candidate_replay_matches(frozen, portable)
    for change in ('large_score', 'rank', 'text', 'nonfinite', 'unhashed'):
        altered = copy.deepcopy(portable)
        hit = altered['rows'][0]['candidates'][0]
        if change == 'large_score': hit['score'] += 1e-5
        if change == 'rank': hit['rank'] = 2
        if change == 'text': hit['passage']['text'] = 'fabricated source'
        if change == 'nonfinite': hit['score'] = math.inf
        if change == 'unhashed':
            hit['score'] += 1e-8
            with pytest.raises(ValueError, match='hash'): candidate_replay_matches(frozen, altered)
            continue
        hashed(altered)
        assert not candidate_replay_matches(frozen, altered)
    lexical, _ = freeze_candidates(contents, manifest, embeddings, 10, ('bm25',))
    changed = copy.deepcopy(lexical)
    changed['rows'][0]['candidates'][0]['score'] += 5e-7
    hashed(changed)
    assert not candidate_replay_matches(lexical, changed)


def test_candidate_ceiling_and_score_integrity(experiment):
    contents, manifest, embeddings = experiment
    candidates, _ = freeze_candidates(contents, manifest, embeddings, 10)
    scores = hashed({"candidate_sha256": candidates["sha256"], "model": {"id": "fixture"},
        "scores": {pair_key(row["question"], h["passage"]["passage_id"]): 0.0
                   for row in candidates["rows"] for h in row["candidates"]}})
    report = assess(contents, manifest, embeddings, candidates, scores, "test")
    for result in report["results"].values():
        row = result["reranked"]["questions"][0]
        # Lexical ID order places :11 inside the ten-item candidate pool.
        assert row["candidate_recall"] == 1
        assert row["metrics"]["reciprocal_first_hit_rank"] == pytest.approx(1 / 4)
    contents["questions.json"][0]["annotations"][0]["evidence"] = ["zebra 9"]
    report = assess(contents, manifest, embeddings, candidates, scores, "test")
    assert report["results"]["dense"]["reranked"]["candidate_recall"] == 0
    assert report["results"]["dense"]["reranked"]["aggregate"]["reciprocal_first_hit_rank"] == 0
    scores["scores"].pop(next(iter(scores["scores"])))
    hashed(scores)
    with pytest.raises(ValueError, match="exactly"): assess(contents, manifest, embeddings, candidates, scores, "test")


def test_exclusions_survive_ablation(experiment):
    contents, manifest, embeddings = experiment
    candidates, _ = freeze_candidates(contents, manifest, embeddings, 10)
    scores = hashed({"candidate_sha256": candidates["sha256"], "model": {}, "scores": {
        pair_key(row["question"], h["passage"]["passage_id"]): 0 for row in candidates["rows"] for h in row["candidates"]}})
    for unanswerable, evidence, reason in ((True, ["zebra 1"], "unanswerable"), (False, ["absent"], "incomplete_evidence")):
        annotation = contents["questions.json"][0]["annotations"][0]
        annotation.update(unanswerable=unanswerable, evidence=evidence)
        report = assess(contents, manifest, embeddings, candidates, scores, "test")
        row = report["results"]["dense"]["reranked"]["questions"][0]
        assert row["excluded_reason"] == reason and row["metrics"] is None
