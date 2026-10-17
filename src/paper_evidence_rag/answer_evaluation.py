"""Replay answer checks and join gold labels only after offline generation."""
from collections import Counter
import re
import unicodedata

from .answering import MODEL_PINS, parse_draft, prompt_hash, sealed, validate_draft
from .evaluation import map_evidence
from .ingestion import digest
from .reranking import checked_hash
from .schema import Passage, Question


def tokens(text):
    words = re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())
    return [w for w in words if w not in {"a", "an", "the"}]


def answer_metrics(answer, references):
    """Best annotation-level normalized exact match and bag-of-token F1.

Different reference answers are alternatives, not one concatenated gold answer.
Extractive spans within an annotation are joined in their supplied order.
"""
    if not references:
        return None
    predicted = tokens(answer)
    values = []
    for reference in references:
        gold = tokens(reference)
        overlap = sum((Counter(predicted) & Counter(gold)).values())
        f1 = 2 * overlap / (len(predicted) + len(gold)) if predicted or gold else 1.0
        values.append({"exact_match": float(predicted == gold), "token_f1": f1})
    return {key: max(v[key] for v in values) for key in ("exact_match", "token_f1")}


def evaluate_answers(contents, manifest, requests, generated, code_revision):
    checked_hash(requests)
    checked_hash(generated)
    if requests["corpus_revision"] != manifest["corpus_revision"] or requests["input_sha256"] != manifest["input_sha256"]:
        raise ValueError("Answer corpus revision mismatch")
    if generated["request_sha256"] != requests["sha256"]:
        raise ValueError("Answer request hash mismatch")
    if generated["prompt_version"] != requests["prompt_version"]:
        raise ValueError("Answer prompt revision mismatch")
    if generated["prompt_sha256"] != prompt_hash(requests["prompt_version"]):
        raise ValueError("Answer instruction hash mismatch")
    if generated["model"]["id"] not in MODEL_PINS or generated["model"]["revision"] != MODEL_PINS[generated["model"]["id"]]:
        raise ValueError("Answer model revision mismatch")
    questions = sorted((Question.model_validate(q) for q in contents["questions.json"]),
        key=lambda q: (q.paper_id, q.question_id))[:20]
    key = lambda r: (r["paper_id"], r["question_id"])
    expected = {(q.paper_id, q.question_id) for q in questions}
    for rows in (requests["rows"], generated["rows"]):
        if len(rows) != len(expected) or {key(r) for r in rows} != expected:
            raise ValueError("Answer question population mismatch")
    req = {key(r): r for r in requests["rows"]}
    outputs = {key(r): r for r in generated["rows"]}
    source = {p["passage_id"]: Passage.model_validate(p).model_dump() for p in contents["passages.json"]}
    rows, coverage = [], Counter()
    for q in questions:
        request, output = req[q.paper_id, q.question_id], outputs[q.paper_id, q.question_id]
        if request["question"] != q.question or output["question"] != q.question:
            raise ValueError("Answer question text mismatch")
        if len(request["passages"]) > 5 or len({p["passage_id"] for p in request["passages"]}) != len(request["passages"]):
            raise ValueError("Invalid answer evidence population")
        for p in request["passages"]:
            if source.get(p["passage_id"]) != p or p["paper_id"] != q.paper_id:
                raise ValueError("Request source text or paper mismatch")
        lookup = {p["passage_id"]: p for p in request["passages"]}
        presented_ids = [p["passage_id"] for p in output["presented"]]
        if len(set(presented_ids)) != len(presented_ids):
            raise ValueError("Duplicate presented evidence")
        if presented_ids != [p["passage_id"] for p in request["passages"] if p["passage_id"] in presented_ids]:
            raise ValueError("Presented evidence order changed")
        for p in output["presented"]:
            original = lookup.get(p["passage_id"])
            if original is None or not original["text"].startswith(p["text"]) or not p["text"].strip() or {
                k: v for k, v in p.items() if k != "text"} != {k: v for k, v in original.items() if k != "text"}:
                raise ValueError("Presented evidence is not an original source prefix")
        replay = validate_draft(output["raw_output"], q.paper_id, output["presented"])
        if output["output_limit_reached"]:
            replay = {"status": "abstained", "reason": "output_limit", "claims": [], "answer": ""}
        if replay != output["result"]:
            raise ValueError("Answer validation replay mismatch")
        try:
            parse_draft(output["raw_output"])
            schema_valid = True
        except (ValueError, TypeError):
            schema_valid = False
        mappings = [{"annotation_id": a.annotation_id, **map_evidence(e,
            [Passage.model_validate(p) for p in source.values() if p["paper_id"] == q.paper_id])}
            for a in q.annotations for e in a.evidence]
        coverage.update(m["match"] for m in mappings)
        gold = {pid for m in mappings for pid in m["passage_ids"]}
        unanswerable = any(a.unanswerable for a in q.annotations)
        incomplete = not gold or any(not a.evidence for a in q.annotations) or any(m["match"] == "unmatched" for m in mappings)
        reason = "unanswerable" if unanswerable else "incomplete_evidence" if incomplete else None
        references = [" ".join(a.extractive_spans) if a.answer_type == "extractive" else
            a.free_form_answer if a.answer_type == "free_form" else "yes" if a.yes_no else "no"
            for a in q.annotations if not a.unanswerable]
        references = [r for r in references if r.strip()]
        cited = {c["passage_id"] for c in replay["claims"]}
        available = {p["passage_id"] for p in output["presented"]}
        rows.append({"paper_id": q.paper_id, "question_id": q.question_id, "question": q.question,
            "answer_types": sorted({a.answer_type for a in q.annotations}), "result": replay,
            "raw_schema_valid": schema_valid, "output_limit_reached": output["output_limit_reached"],
            "references": references, "excluded_reason": reason, "evidence_mappings": mappings,
            "gold_passage_ids": sorted(gold), "presented_ids": presented_ids,
            "missing_gold_ids": sorted(gold - available),
            "answer_metrics": answer_metrics(replay["answer"], references) if reason is None else None,
            "citation_evidence_precision": len(cited & gold) / len(cited) if cited and reason is None else None,
            "citation_evidence_recall": len(cited & gold) / len(gold) if reason is None else None,
            "unanswerable_abstained": replay["status"] == "abstained" if unanswerable else None,
            "manual_review_required": replay["status"] == "answered"})
    eligible = [r for r in rows if r["answer_metrics"] is not None]
    def averages(items):
        return {k: sum(r["answer_metrics"][k] for r in items) / len(items) if items else None
                for k in ("exact_match", "token_f1")}
    return sealed({"code_revision": code_revision, "corpus_revision": manifest["corpus_revision"],
        "input_sha256": manifest["input_sha256"], "request_sha256": requests["sha256"],
        "generation_sha256": generated["sha256"], "question_count": len(rows), "eligible_count": len(eligible),
        "excluded_counts": dict(Counter(r["excluded_reason"] for r in rows if r["excluded_reason"])),
        "evidence_mapping_counts": dict(coverage), "status_counts": dict(Counter(r["result"]["status"] for r in rows)),
        "abstention_reasons": dict(Counter(r["result"]["reason"] for r in rows if r["result"]["reason"])),
        "raw_schema_valid_count": sum(r["raw_schema_valid"] for r in rows),
        "raw_schema_valid_rate": sum(r["raw_schema_valid"] for r in rows) / len(rows) if rows else None,
        "aggregate": averages(eligible), "by_answer_type": {t: {"count": len(items), **averages(items)}
            for t in sorted({t for r in eligible for t in r["answer_types"]})
            if (items := [r for r in eligible if t in r["answer_types"]])}, "questions": rows,
        "interpretation": "Training debugging sample. Exact quotes verify provenance, not question relevance. Token F1 is lexical overlap, not factual correctness. Gold evidence is incomplete; inspect every answered row manually."})
