"""Paired comparisons on the fixed phase-one questions and source IDs."""
import importlib.metadata
import time

from .evaluation import evaluate
from .qdrant_retrieval import CONFIG, QdrantRetriever
from .retrieval import Retriever


def compare(contents, manifest, code_revision, embeddings=None):
    results, timings, statistics = {}, {}, None
    modes = ["legacy", "bm25"] + (["dense", "hybrid"] if embeddings is not None else [])
    for mode in modes:
        start = time.perf_counter()
        retriever = (Retriever(contents["passages.json"], manifest["paper_ids"], manifest["corpus_revision"])
                     if mode == "legacy" else QdrantRetriever(contents["passages.json"], manifest["paper_ids"],
                         manifest["corpus_revision"], embeddings, mode))
        build_seconds = time.perf_counter() - start
        try:
            report = evaluate(retriever, contents["questions.json"], manifest, code_revision)
            timings[mode] = {"build_seconds": build_seconds, "evaluation_seconds": report.pop("elapsed_seconds")}
            results[mode] = report
            if mode != "legacy":
                statistics = retriever.statistics
            for row in report["questions"]:
                for hit in row["retrieved"]:
                    if retriever.inspect(hit["passage"]["passage_id"]).model_dump() != hit["passage"]:
                        raise ValueError("Returned passage failed source resolution")
        finally:
            if hasattr(retriever, "close"):
                retriever.close()
    differences = []
    for before, after in zip(results["legacy"]["questions"], results["bm25"]["questions"]):
        old_ids = [h["passage"]["passage_id"] for h in before["retrieved"]]
        new_ids = [h["passage"]["passage_id"] for h in after["retrieved"]]
        differences.append({"paper_id": before["paper_id"], "question_id": before["question_id"],
            "same_top_10": old_ids == new_ids, "legacy_ids": old_ids, "qdrant_ids": new_ids})
    return {"code_revision": code_revision, "corpus_revision": manifest["corpus_revision"],
        "configuration": CONFIG, "statistics": statistics,
        "embedding_sha256": embeddings["sha256"] if embeddings else None,
        "model": embeddings["model"] if embeddings else None,
        "truncated": embeddings["truncated"] if embeddings else None,
        "dependencies": {n: importlib.metadata.version(n) for n in ("qdrant-client", "numpy", "rank-bm25")},
        "results": results, "ranking_differences": differences,
        "difference_explanation": "Tokenization, k1, b and paper-local statistics are held fixed. Qdrant uses positive log(1+(N-df+0.5)/(df+0.5)) IDF; rank_bm25 uses log((N-df+0.5)/(df+0.5)) with epsilon smoothing. Qdrant sparse search omits zero-overlap passages; legacy returns them. Dense inputs may be truncated, while source passages remain whole.",
        "interpretation": "Fixed training debugging sample; no general quality claim"}, timings
