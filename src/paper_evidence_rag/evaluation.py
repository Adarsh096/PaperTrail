import importlib.metadata
import time
import unicodedata
from collections import Counter

from .schema import Question


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def map_evidence(text, passages):
    exact = [p.passage_id for p in passages if p.text == text]
    if exact:
        return {"text": text, "match": "exact", "passage_ids": sorted(exact)}
    matches = [p.passage_id for p in passages if normalized(p.text) == normalized(text)] if normalized(text) else []
    return {"text": text, "match": "normalized" if matches else "unmatched", "passage_ids": sorted(matches)}


def metrics(ranked, gold):
    gold = set(gold)
    if not gold:
        raise ValueError("Metrics require mapped gold evidence")
    return {
        "recall_at_5": len(set(ranked[:5]) & gold) / len(gold),
        "recall_at_10": len(set(ranked[:10]) & gold) / len(gold),
        "reciprocal_first_hit_rank": next((1 / i for i, p in enumerate(ranked, 1) if p in gold), 0.0),
    }


def evaluate(retriever, questions, manifest, code_revision):
    start = time.perf_counter()
    rows, mapping_counts, types = [], Counter(), Counter()
    ordered = sorted((Question.model_validate(q) for q in questions), key=lambda q: (q.paper_id, q.question_id))[:20]
    for q in ordered:
        unanswerable = any(a.unanswerable for a in q.annotations)
        mappings = [{"annotation_id": a.annotation_id, **map_evidence(e, retriever.papers[q.paper_id])}
                    for a in q.annotations for e in a.evidence]
        mapping_counts.update(m["match"] for m in mappings)
        types.update(set(a.answer_type for a in q.annotations))
        gold = sorted({p for m in mappings for p in m["passage_ids"]})
        reason = ("unanswerable" if unanswerable else "incomplete_evidence" if not gold or any(not a.evidence for a in q.annotations) or any(
            m["match"] == "unmatched" for m in mappings) else None)
        # Retrieve the entire paper for an uncensored first-hit rank.
        hits = retriever.search(q.paper_id, q.question, max(1, len(retriever.papers[q.paper_id])))
        ranked = [h.passage.passage_id for h in hits]
        rows.append({"paper_id": q.paper_id, "question_id": q.question_id, "question": q.question,
                     "answer_types": sorted(set(a.answer_type for a in q.annotations)),
                     "evidence_mappings": mappings, "gold_passage_ids": gold,
                     "excluded_reason": reason, "retrieved": [h.model_dump() for h in hits[:10]],
                     "metrics": metrics(ranked, gold) if reason is None else None})
    eligible = [r["metrics"] for r in rows if r["metrics"] is not None]
    aggregate = {key: sum(m[key] for m in eligible) / len(eligible) if eligible else None
                 for key in ("recall_at_5", "recall_at_10", "reciprocal_first_hit_rank")}
    dependencies = ("pydantic", "rank-bm25", "numpy") if not hasattr(retriever, "mode") else ("pydantic", "qdrant-client", "numpy")
    return {"corpus_revision": manifest["corpus_revision"], "input_sha256": manifest["input_sha256"],
            "code_revision": code_revision, "dependencies": {name: importlib.metadata.version(name)
            for name in dependencies}, "question_count": len(rows),
            "eligible_count": len(eligible), "excluded_counts": dict(Counter(r["excluded_reason"] for r in rows if r["excluded_reason"])),
            "answer_type_counts": dict(types), "evidence_mapping_counts": dict(mapping_counts),
            "aggregate": aggregate, "questions": rows, "elapsed_seconds": time.perf_counter() - start,
            "interpretation": "Bounded training debugging sample; not a general quality benchmark"}
