"""Compare structure compliance separately from lexical answer scores."""
from .answering import sealed
from .reranking import checked_hash


def compare_answers(runs):
    if len(runs) != 2:
        raise ValueError("Exactly two generator runs required")
    first = runs[0][0]
    comparison = []
    ids = set()
    for generated, report, timing in runs:
        checked_hash(generated)
        checked_hash(report)
        if report["generation_sha256"] != generated["sha256"]:
            raise ValueError("Generation report mismatch")
        if generated["model"]["id"] in ids:
            raise ValueError("Duplicate comparison model")
        ids.add(generated["model"]["id"])
        for field in ("request_sha256", "prompt_version", "prompt_sha256"):
            if generated[field] != first[field]:
                raise ValueError("Unpaired generator inputs")
        settings = ("device", "dtype", "threads", "seed", "do_sample", "max_input_tokens", "max_new_tokens")
        if any(generated["model"][k] != first["model"][k] for k in settings):
            raise ValueError("Unpaired generator settings")
        presentation = lambda g: [{k: r[k] for k in ("paper_id", "question_id", "question", "presented")}
                                 for r in g["rows"]]
        if presentation(generated) != presentation(first):
            raise ValueError("Models received different evidence")
        if report["code_revision"] != runs[0][1]["code_revision"]:
            raise ValueError("Comparison code revisions differ")
        comparison.append({"model": generated["model"], "raw_schema_valid_count": report["raw_schema_valid_count"],
            "question_count": report["question_count"], "raw_schema_valid_rate": report["raw_schema_valid_rate"],
            "status_counts": report["status_counts"], "abstention_reasons": report["abstention_reasons"],
            "aggregate": report["aggregate"], "by_answer_type": report["by_answer_type"], "timings": timing,
            "questions": [{k: r[k] for k in ("question_id", "result", "raw_schema_valid", "answer_metrics",
                "missing_gold_ids", "unanswerable_abstained")} for r in report["questions"]]})
    return sealed({"request_sha256": first["request_sha256"], "code_revision": runs[0][1]["code_revision"],
        "models": comparison, "selection_rule": "Prefer raw schema compliance, then manually reviewed responsive supported answers; latency is a tradeoff. Do not select from token F1 alone.",
        "interpretation": "Six training questions, five eligible. One CPU timing run; no production quality claim or automatic model promotion."})
