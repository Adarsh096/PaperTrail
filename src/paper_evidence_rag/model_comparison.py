"""Compare rerankers on identical pools without comparing raw logit scales."""
import math

from .reranking import validate_model_pin


def compare_models(reports, timings, selection_mode="dense"):
    if len(reports) < 2 or set(reports) != set(timings):
        raise ValueError("Need paired reports and timings for at least two models")
    reference = next(iter(reports.values()))
    for name, report in reports.items():
        validate_model_pin(report["model"])
        if name != report["model"]["id"]:
            raise ValueError("Model identity mismatch")
        for key in ("candidate_sha256", "corpus_revision", "input_sha256", "code_revision"):
            if report[key] != reference[key]:
                raise ValueError("Comparison inputs or code revisions differ")
        if set(report["results"]) != set(reference["results"]):
            raise ValueError("Candidate modes differ")
        for mode in report["results"]:
            # Baseline records include gold mappings, exclusions and source IDs.
            if report["results"][mode]["baseline"] != reference["results"][mode]["baseline"]:
                raise ValueError("Evaluation populations or baseline rankings differ")
        seconds = timings[name]["inference_seconds"]
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Inference timing must be positive and finite")
    if selection_mode not in reference["results"]:
        raise ValueError("Unknown selection mode")
    def selection_key(name):
        metrics = reports[name]["results"][selection_mode]["reranked"]["aggregate"]
        if any(value is None for value in metrics.values()):
            raise ValueError("Cannot select a model without eligible questions")
        return (-metrics["recall_at_5"], -metrics["recall_at_10"],
                -metrics["reciprocal_first_hit_rank"], timings[name]["inference_seconds"], name)
    ordered = sorted(reports, key=selection_key)
    return {"code_revision": reference["code_revision"], "candidate_sha256": reference["candidate_sha256"],
        "selection_mode": selection_mode, "recommended_model": ordered[0],
        "selection_rule": "Dense candidates: maximize Recall@5, then Recall@10, then pool MRR; use lower batch inference time for a quality tie",
        "models": {name: {"model": report["model"], "timings": timings[name], "results": {
            mode: {"aggregate": data["reranked"]["aggregate"], "candidate_recall": data["reranked"]["candidate_recall"]}
            for mode, data in report["results"].items()}} for name, report in reports.items()},
        "interpretation": "Provisional choice on five training debugging questions; one timing run per model. Tokenizers differ, raw logits are not comparable, and batched timing is not interactive latency."}
