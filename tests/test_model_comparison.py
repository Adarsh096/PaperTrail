import copy

import pytest

from paper_evidence_rag.model_comparison import compare_models
from paper_evidence_rag.reranking import MODEL_PINS


def fixture():
    reports, times = {}, {}
    for model, revision in MODEL_PINS.items():
        reports[model] = {"model": {"id": model, "revision": revision}, "candidate_sha256": "c",
            "corpus_revision": "r", "input_sha256": "i", "code_revision": "commit", "results": {
            "dense": {"baseline": {"questions": ["same population"]}, "reranked": {
                "aggregate": {"recall_at_5": .6, "recall_at_10": .7, "reciprocal_first_hit_rank": .5},
                "candidate_recall": .8}}}}
        times[model] = {"inference_seconds": 1.0 if "MiniLM" in model else 10.0}
    return reports, times


def test_quality_precedes_speed_and_ties_use_speed():
    reports, times = fixture()
    result = compare_models(reports, times)
    assert "MiniLM" in result["recommended_model"]
    bge = "BAAI/bge-reranker-v2-m3"
    reports[bge]["results"]["dense"]["reranked"]["aggregate"]["recall_at_5"] = .8
    assert compare_models(reports, times)["recommended_model"] == bge


@pytest.mark.parametrize("field", ["candidate_sha256", "code_revision", "baseline"])
def test_comparison_rejects_unpaired_inputs(field):
    reports, times = fixture()
    key = next(iter(reports))
    reports[key] = copy.deepcopy(reports[key])
    if field == "baseline": reports[key]["results"]["dense"]["baseline"] = {"different": True}
    else: reports[key][field] = "changed"
    with pytest.raises(ValueError): compare_models(reports, times)
