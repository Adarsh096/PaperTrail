import copy
import json
import subprocess
import sys

import pytest

from paper_evidence_rag.evaluation import evaluate, map_evidence, metrics
from paper_evidence_rag.ingestion import build, canonical, digest, load_artifacts, normalize
from paper_evidence_rag.retrieval import Retriever, tokenize


@pytest.fixture
def raw():
    def answer(evidence, unanswerable=False):
        return {"annotation_id": "a1", "answer": {
            "unanswerable": unanswerable, "yes_no": None,
            "extractive_spans": [], "free_form_answer": "gold-answer-only",
            "evidence": evidence,
        }}
    return {"p1": {"title": "Sample", "abstract": "Introduction overview.",
        "full_text": [{"section_name": "Method", "paragraphs": [
            "Zebras eat grass.", "", "Café experiment.", "Zebras eat grass.",
            "Other unrelated material.", "More different tokens.", "Last paragraph."]}],
        "figures_and_tables": [{"caption": "Figure 1: results.", "file": "figure.png"}],
        "qas": [{"question_id": "q1", "question": "zebras", "answers": [answer(["Zebras eat grass."])]},
                {"question_id": "q2", "question": "experiment", "answers": [answer(["Missing evidence."])]},
                {"question_id": "q3", "question": "unanswerable", "answers": [answer([], True)]} ]}}


def make_retriever(raw):
    papers, passages, questions = normalize(raw)
    return Retriever(passages, [p["paper_id"] for p in papers], "revision"), questions


def test_normalization_preserves_ids_text_and_labels(raw):
    papers, passages, questions = normalize(raw)
    ids = {p["passage_id"] for p in passages}
    assert "p1:paragraph:0:1" not in ids
    assert "p1:paragraph:0:2" in ids
    assert "p1:caption:0" in ids
    assert next(p for p in passages if p["passage_id"] == "p1:paragraph:0:2")["text"] == "Café experiment."
    assert "gold-answer-only" not in canonical(papers) + canonical(passages)
    assert questions[0]["annotations"][0]["free_form_answer"] == "gold-answer-only"


@pytest.mark.parametrize("mutation", ["bad_paragraph", "missing_title", "duplicate_question", "missing_annotations"])
def test_malformed_input(raw, mutation):
    if mutation == "bad_paragraph":
        raw["p1"]["full_text"][0]["paragraphs"] = [42]
    elif mutation == "missing_title":
        del raw["p1"]["title"]
    elif mutation == "duplicate_question":
        raw["p1"]["qas"].append(raw["p1"]["qas"][0])
    else:
        raw["p1"]["qas"][0]["answers"] = []
    with pytest.raises((ValueError, KeyError)):
        normalize(raw)


def test_deterministic_selection(raw):
    sources = {f"p{i}": copy.deepcopy(raw["p1"]) for i in reversed(range(7))}
    papers, _, _ = normalize(sources)
    assert [p["paper_id"] for p in papers] == ["p0", "p1", "p2", "p3", "p4"]


def test_retrieval_scope_order_and_lookup(raw):
    raw["p2"] = copy.deepcopy(raw["p1"])
    r, _ = make_retriever(raw)
    hits = r.search("p1", "ZEBRAS")
    assert [h.passage.passage_id for h in hits[:2]] == ["p1:paragraph:0:0", "p1:paragraph:0:3"]
    assert all(h.passage.paper_id == "p1" for h in hits)
    assert r.inspect(hits[0].passage.passage_id).text == "Zebras eat grass."
    assert r.search("p1", "") == []
    assert r.search("p1", "?!") == []
    assert r.search("p1", "unknowntoken") == []
    with pytest.raises(ValueError, match="Unknown paper"):
        r.search("missing", "zebras")
    with pytest.raises(ValueError, match="Unknown passage"):
        r.inspect("missing")
    with pytest.raises(ValueError, match="positive"):
        r.search("p1", "zebras", 0)
    assert tokenize("CAFÉ 中文_词 2026") == ["café", "中文_词", "2026"]


def test_evidence_mapping(raw):
    r, _ = make_retriever(raw)
    assert map_evidence("Zebras eat grass.", r.papers["p1"])["passage_ids"] == ["p1:paragraph:0:0", "p1:paragraph:0:3"]
    assert map_evidence("Cafe\u0301   experiment.", r.papers["p1"])["match"] == "normalized"
    assert map_evidence("Figure 1: results.", r.papers["p1"])["passage_ids"] == ["p1:caption:0"]
    assert map_evidence("Zebras eat grasses.", r.papers["p1"])["match"] == "unmatched"


def test_metrics_are_hand_calculated():
    m = metrics(["x", "a", "b", "c", "d", "z"], ["a", "z", "z"])
    assert m == {"recall_at_5": 0.5, "recall_at_10": 1.0, "reciprocal_first_hit_rank": 0.5}
    assert metrics(["x"], ["a"])["reciprocal_first_hit_rank"] == 0
    with pytest.raises(ValueError):
        metrics([], [])


def test_first_hit_rank_is_not_capped_at_100():
    passages = [{"passage_id": f"p:paragraph:0:{i:03}", "paper_id": "p", "kind": "paragraph",
                 "section_index": 0, "position": i, "section_name": "S", "text": "shared"}
                for i in range(120)]
    # All duplicate text is mapped by design, so use an exact caption as gold
    # with the same tokens and therefore the same BM25 score.
    passages[-1]["text"] = "shared!"
    r = Retriever(passages, ["p"], "revision")
    questions = [{"paper_id": "p", "question_id": "q", "question": "shared", "annotations": [{
        "annotation_id": "a", "unanswerable": False, "answer_type": "free_form",
        "evidence": ["shared!"], "extractive_spans": [], "free_form_answer": "answer", "yes_no": None}]}]
    result = evaluate(r, questions, {"corpus_revision": "revision", "input_sha256": "hash"}, "commit")
    assert result["aggregate"]["reciprocal_first_hit_rank"] == 1 / 120
    assert result["aggregate"]["recall_at_10"] == 0


def test_empty_paper_and_zero_token_passages(raw):
    raw["p1"]["abstract"] = ""
    raw["p1"]["full_text"] = []
    raw["p1"]["figures_and_tables"] = []
    r, _ = make_retriever(raw)
    assert r.search("p1", "anything") == []
    raw["p1"]["abstract"] = "?!"
    r, _ = make_retriever(raw)
    assert r.search("p1", "anything") == []


def test_incomplete_annotation_excludes_question(raw):
    a = copy.deepcopy(raw["p1"]["qas"][0]["answers"][0])
    a["answer"]["evidence"] = []
    raw["p1"]["qas"][0]["answers"].append(a)
    r, questions = make_retriever(raw)
    result = evaluate(r, questions, {"corpus_revision": "revision", "input_sha256": "hash"}, "commit")
    assert result["eligible_count"] == 0
    assert result["aggregate"]["recall_at_5"] is None


def test_evaluation_annotations_and_exclusions(raw):
    second = copy.deepcopy(raw["p1"]["qas"][0]["answers"][0])
    second["annotation_id"] = "a2"
    second["answer"]["evidence"] = ["Figure 1: results."]
    raw["p1"]["qas"][0]["answers"].append(second)
    r, questions = make_retriever(raw)
    report = evaluate(r, questions, {"corpus_revision": "revision", "input_sha256": "hash"}, "commit")
    assert report["eligible_count"] == 1
    assert report["excluded_counts"] == {"incomplete_evidence": 1, "unanswerable": 1}
    assert len(report["questions"][0]["gold_passage_ids"]) == 3
    assert report["evidence_mapping_counts"] == {"exact": 2, "unmatched": 1}
    assert report["questions"][1]["retrieved"]


def test_build_reproducibility_integrity_and_cli(raw, tmp_path):
    source = tmp_path / "train.json"
    source.write_text(canonical(raw), encoding="utf-8")
    sha = digest(source.read_bytes())
    first, second = tmp_path / "first", tmp_path / "second"
    build(source, first, sha)
    build(source, second, sha)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    manifest, contents = load_artifacts(first)
    assert manifest["paper_count"] == 1
    with pytest.raises(ValueError, match="input"):
        build(source, second, "wrong")
    command = [sys.executable, "-m", "paper_evidence_rag.cli", "search", "--artifacts", str(first), "--paper", "p1", "--question", "zebras"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0
    assert json.loads(result.stdout)[0]["passage"]["passage_id"] == "p1:paragraph:0:0"
    result = subprocess.run(command[:-3] + ["missing", "--question", "zebras"], capture_output=True, text=True)
    assert result.returncode == 2
    (first / "passages.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact"):
        load_artifacts(first)
