"""Freeze label-free candidates, then evaluate a cached cross-encoder ablation."""
import math
import time

from .evaluation import evaluate, metrics
from .ingestion import canonical, digest
from .qdrant_retrieval import QdrantRetriever

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
MODEL_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
MODEL_PINS = {
    MODEL_ID: MODEL_REVISION,
    "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
}
# The paired five-question run tied dense Recall@5, while BGE improved
# Recall@10 and MRR. This quality-first default is provisional; MiniLM took
# about 1/24.5 of the CPU batch time and remains an explicit pinned option.
# See README for both results and the selection rule in model_comparison.py.
DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DENSE_SCORE_ATOL = 1e-6


def validate_model_pin(snapshot):
    if snapshot.get("id") not in MODEL_PINS or snapshot.get("revision") != MODEL_PINS[snapshot["id"]]:
        raise ValueError("Reranker snapshot revision mismatch")


def checked_hash(value):
    expected = digest(canonical({k: v for k, v in value.items() if k != "sha256"}).encode())
    if value.get("sha256") != expected:
        raise ValueError("Artifact hash mismatch")


def freeze_candidates(contents, manifest, embeddings, candidate_count=50, modes=("bm25", "dense", "hybrid")):
    if not 10 <= candidate_count <= 100:
        raise ValueError("Candidate count must be between 10 and 100")
    if len(manifest["paper_ids"]) > 20:
        raise ValueError("This experiment is bounded to 20 papers")
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"bm25", "dense", "hybrid"}:
        raise ValueError("Invalid candidate retrieval modes")
    questions = sorted(contents["questions.json"], key=lambda q: (q["paper_id"], q["question_id"]))[:20]
    rows, timings = [], {}
    for mode in modes:
        retriever = QdrantRetriever(contents["passages.json"], manifest["paper_ids"],
                                     manifest["corpus_revision"], embeddings, mode)
        start = time.perf_counter()
        try:
            for question in questions:
                hits = retriever.search(question["paper_id"], question["question"], candidate_count)
                # Scoring artifacts contain query/source text only. Gold labels
                # are joined later by evaluation and cannot influence scores.
                rows.append({"mode": mode, "paper_id": question["paper_id"],
                    "question_id": question["question_id"], "question": question["question"],
                    "candidates": [h.model_dump() for h in hits]})
        finally:
            retriever.close()
        timings[mode] = time.perf_counter() - start
    result = {"corpus_revision": manifest["corpus_revision"], "input_sha256": manifest["input_sha256"],
        "embedding_sha256": embeddings["sha256"], "candidate_count": candidate_count,
        "selection_rule": "First 20 questions ordered by paper ID and question ID; " + (
            "five training papers" if len(manifest["paper_ids"]) == 5 else
            f"{len(manifest['paper_ids'])} training " + ("paper" if len(manifest["paper_ids"]) == 1 else "papers")),
        "questions": [{k: q[k] for k in ("paper_id", "question_id", "question")} for q in questions],
        "rows": rows}
    result["sha256"] = digest(canonical(result).encode())
    return result, timings


def pair_key(question, passage_id):
    return digest(canonical([question, passage_id]).encode())


def candidate_replay_matches(expected, saved):
    """Verify both hashes and exact input/rank/source identity.

    Float32 cosine computation can vary across CPU libraries. Only dense hit
    scores allow an absolute 1e-6 replay tolerance. Keep saved scores and their
    hash untouched; this does not round scores, allow rank swaps or alter text.
    """
    checked_hash(expected)
    checked_hash(saved)
    metadata = lambda value: {k: v for k, v in value.items() if k not in {"sha256", "rows"}}
    if metadata(expected) != metadata(saved) or len(expected["rows"]) != len(saved["rows"]):
        return False
    for a, b in zip(expected["rows"], saved["rows"]):
        row = lambda value: {k: v for k, v in value.items() if k != "candidates"}
        if row(a) != row(b) or len(a["candidates"]) != len(b["candidates"]):
            return False
        for hit, original in zip(a["candidates"], b["candidates"]):
            content = lambda value: {k: v for k, v in value.items() if k != "score"}
            if content(hit) != content(original):
                return False
            x, y = hit["score"], original["score"]
            if type(x) not in {int, float} or type(y) not in {int, float} or not math.isfinite(x) or not math.isfinite(y):
                return False
            tolerance = DENSE_SCORE_ATOL if a["mode"] == "dense" else 0.0
            if not math.isclose(x, y, rel_tol=0.0, abs_tol=tolerance):
                return False
    return True


def rerank(candidates, question, scores):
    ids = [h["passage"]["passage_id"] for h in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate candidate passage ID")
    values = [(float(scores[pair_key(question, pid)]), hit) for pid, hit in zip(ids, candidates)]
    if any(not math.isfinite(score) for score, _ in values):
        raise ValueError("Reranker scores must be finite")
    # Raw logits are ordered within one question, never mixed with BM25/cosine.
    values.sort(key=lambda item: (-item[0], item[1]["passage"]["passage_id"]))
    return [{**hit, "rank": rank, "score": score} for rank, (score, hit) in enumerate(values, 1)]


def assess(contents, manifest, embeddings, candidates, scores, code_revision):
    checked_hash(candidates)
    checked_hash(scores)
    modes = tuple(dict.fromkeys(r["mode"] for r in candidates["rows"]))
    expected, _ = freeze_candidates(contents, manifest, embeddings, candidates["candidate_count"], modes)
    if not candidate_replay_matches(expected, candidates) or scores["candidate_sha256"] != candidates["sha256"]:
        raise ValueError("Candidate inputs changed since freeze")
    keys = {pair_key(row["question"], h["passage"]["passage_id"])
            for row in candidates["rows"] for h in row["candidates"]}
    if set(scores["scores"]) != keys:
        raise ValueError("Scores must cover exactly the frozen query-passage pairs")
    results = {}
    for mode in modes:
        retriever = QdrantRetriever(contents["passages.json"], manifest["paper_ids"],
                                     manifest["corpus_revision"], embeddings, mode)
        try:
            baseline = evaluate(retriever, contents["questions.json"], manifest, code_revision)
        finally:
            retriever.close()
        baseline.pop("elapsed_seconds")
        pools = {(r["paper_id"], r["question_id"]): r for r in candidates["rows"] if r["mode"] == mode}
        rows = []
        for base in baseline["questions"]:
            pool = pools[base["paper_id"], base["question_id"]]
            hits = rerank(pool["candidates"], pool["question"], scores["scores"])
            ranked = [h["passage"]["passage_id"] for h in hits]
            gold = set(base["gold_passage_ids"])
            eligible = base["excluded_reason"] is None
            rows.append({**base, "retrieved": hits[:10],
                "candidate_ids": [h["passage"]["passage_id"] for h in pool["candidates"]],
                "candidate_baseline_metrics": metrics([h["passage"]["passage_id"] for h in pool["candidates"]], gold) if eligible else None,
                "missing_gold_ids": sorted(gold - set(ranked)),
                "candidate_recall": len(gold & set(ranked)) / len(gold) if eligible else None,
                "metrics": metrics(ranked, gold) if eligible else None})
        eligible = [r for r in rows if r["metrics"] is not None]
        aggregate = {k: sum(r["metrics"][k] for r in eligible) / len(eligible) if eligible else None
                     for k in baseline["aggregate"]}
        pool_aggregate = {k: sum(r["candidate_baseline_metrics"][k] for r in eligible) / len(eligible) if eligible else None
                          for k in baseline["aggregate"]}
        results[mode] = {"baseline": baseline, "reranked": {"questions": rows, "aggregate": aggregate,
            "candidate_recall": sum(r["candidate_recall"] for r in eligible) / len(eligible) if eligible else None},
            "candidate_baseline_aggregate": pool_aggregate,
            "metric_deltas": {k: aggregate[k] - pool_aggregate[k] if eligible else None for k in aggregate}}
    return {"code_revision": code_revision, "corpus_revision": manifest["corpus_revision"],
        "input_sha256": manifest["input_sha256"], "candidate_sha256": candidates["sha256"],
        "scores_sha256": scores["sha256"], "model": scores["model"], "results": results,
        "interpretation": "Bounded training debugging ablation; MRR after reranking is bounded to candidates. No default is selected from this sample."}
