"""Frozen evaluation inputs and explicit semantic score denominators.

Gold annotations enter here after generation. They never enter retrieval or the
answer prompt. Rejected drafts are diagnostics, never accepted answers.
"""
from collections import Counter
import math

from .answer_evaluation import evaluate_answers
from .answering import parse_draft, sealed
from .reranking import checked_hash

PROTOCOL = "ragas-debug-v1"
METRICS = ("faithfulness", "answer_relevance", "answer_relevancy", "context_precision", "context_recall", "factual_correctness")


def prepare_semantic(contents, manifest, requests, generated, code_revision):
    """Reuse all existing source, population, hash and validation replay checks."""
    baseline = evaluate_answers(contents, manifest, requests, generated, code_revision)
    outputs = {(r["paper_id"], r["question_id"]): r for r in generated["rows"]}
    rows = []
    for evaluated in baseline["questions"]:
        output = outputs[evaluated["paper_id"], evaluated["question_id"]]
        diagnostic = None
        if evaluated["result"]["status"] != "answered":
            try:
                draft = parse_draft(output["raw_output"])
                if not draft.abstain and not output["output_limit_reached"]:
                    diagnostic = " ".join(c.text for c in draft.claims).strip() or None
            except (ValueError, TypeError):
                pass  # Malformed drafts remain available as raw text, not repaired.
        rows.append({**{k: evaluated[k] for k in ("paper_id", "question_id", "question", "answer_types",
            "references", "excluded_reason", "evidence_mappings", "gold_passage_ids")},
            "contexts": [{"passage_id": p["passage_id"], "text": p["text"]} for p in output["presented"]],
            "accepted_response": evaluated["result"]["answer"] or None,
            "validation": evaluated["result"], "diagnostic_response": diagnostic,
            "raw_output": output["raw_output"]})
    return sealed({"protocol": PROTOCOL, "code_revision": code_revision,
        "corpus_revision": manifest["corpus_revision"], "input_sha256": manifest["input_sha256"],
        "request_sha256": requests["sha256"], "generation_sha256": generated["sha256"],
        "baseline": baseline, "rows": rows,
        "reference_policy": "Score each distinct nonempty annotation reference separately; report the maximum per reference-based metric. Never concatenate alternative answers.",
        "context_policy": "Exact prefixes presented to the generator, in their original retrieval order."})


def metric_jobs(row, track):
    """Metrics unavailable for empty answers are explicit, not fabricated zeros."""
    response = row["accepted_response" if track == "accepted" else "diagnostic_response"]
    contexts = [p["text"] for p in row["contexts"]]
    references = sorted(set(r for r in row["references"] if r.strip()))
    common = {"user_input": row["question"], "retrieved_contexts": contexts}
    jobs = []
    for metric in METRICS:
        inputs, reason = [], None
        if track == "diagnostic" and not response:
            reason = "no_diagnostic_response"
        elif metric in ("faithfulness", "answer_relevance", "answer_relevancy"):
            if not response:
                reason = "no_response"
            elif metric == "faithfulness" and not contexts:
                reason = "no_context"
            else:
                inputs = ([{"user_input": row["question"], "response": response}] if metric == "answer_relevancy"
                          else [{**common, "response": response}])
        elif not references:
            reason = "no_reference"
        elif row["excluded_reason"] == "unanswerable":
            reason = "unanswerable"
        elif metric == "factual_correctness":
            if response:
                inputs = [{"response": response, "reference": reference} for reference in references]
            else:
                reason = "no_response"
        elif not contexts:
            reason = "no_context"
        else:
            inputs = [{**common, "reference": reference} for reference in references]
        jobs.append({"metric": metric, "inputs": inputs, "not_applicable": reason})
    return jobs


def aggregate_semantic(rows):
    """Separate accepted, diagnostics and excluded strata with per-metric counts.

    Exclusions preserve phase 5's denominator policy. Judge failures stay visible
    and never become zeros. Coverage must be read beside every mean.
    """
    groups = {}
    for track in ("accepted", "diagnostic"):
        for stratum in ("eligible", "unanswerable", "incomplete_evidence"):
            items = [r for r in rows if (r["excluded_reason"] or "eligible") == stratum]
            metrics = {}
            for name in METRICS:
                results = [r["scores"][track][name] for r in items]
                values = [r["value"] for r in results if r["status"] == "scored"]
                lower = -1 if name == "answer_relevancy" else 0
                if any(type(v) not in (int, float) or not math.isfinite(v) or not lower <= v <= 1 for v in values):
                    raise ValueError("Semantic score must be finite and between zero and one")
                metrics[name] = {"mean": sum(values) / len(values) if values else None,
                    "scored_count": len(values), "question_count": len(items),
                    "status_counts": dict(Counter(r["status"] for r in results))}
            groups[f"{track}/{stratum}"] = metrics
    return groups


def validate_semantic(inputs):
    checked_hash(inputs)
    if inputs["protocol"] != PROTOCOL or not 1 <= len(inputs["rows"]) <= 20:
        raise ValueError("Unsupported semantic protocol or population")
    keys = [(r["paper_id"], r["question_id"]) for r in inputs["rows"]]
    if keys != sorted(set(keys)):
        raise ValueError("Semantic rows must be unique and sorted")
