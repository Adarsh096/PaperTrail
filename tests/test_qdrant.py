import copy
import math
import socket

import pytest

from paper_evidence_rag.ingestion import canonical, digest
from paper_evidence_rag.qdrant_retrieval import QdrantRetriever, fuse


@pytest.fixture
def passages():
    return [{"passage_id": f"{paper}:paragraph:0:{i}", "paper_id": paper, "kind": "paragraph",
        "position": i, "section_index": 0, "section_name": "Method", "text": text}
        for paper, i, text in [("p1", 0, "zebra grass"), ("p1", 1, "zebra grass"),
            ("p1", 2, "other words"), ("p2", 0, "zebra zebra zebra grass")]]


def embeddings(passages):
    value = {"corpus_revision": "revision", "model": {"id": "fixture-only", "revision": "a" * 40,
        "dimension": 2}, "passages": {p["passage_id"]: ([1, 0] if p["position"] == 2 else [0, 1])
        for p in passages}, "queries": {digest(b"zebra"): [1, 0]}, "truncated": {"passages": [], "queries": []}}
    value["sha256"] = digest(canonical(value).encode())
    return value


def test_qdrant_bm25_score_scope_and_ties(passages):
    r = QdrantRetriever(passages, ["p1", "p2"], "revision")
    try:
        hits = r.search("p1", "ZEBRA")
        assert [h.passage.passage_id for h in hits] == ["p1:paragraph:0:0", "p1:paragraph:0:1"]
        # N=3, df=2, TF weight=1 because every document has length two.
        assert hits[0].score == pytest.approx(math.log(1 + 1.5 / 2.5))
        assert r.search("p1", "zebra zebra")[0].score == pytest.approx(2 * hits[0].score)
        assert r.statistics["p1"]["average_tokens"] == 2
        assert r.statistics["p2"]["average_tokens"] == 4
        assert r.inspect(hits[0].passage.passage_id).model_dump() == hits[0].passage.model_dump()
        assert r.search("p1", "") == r.search("p1", "?!") == r.search("p1", "unknown") == []
        with pytest.raises(ValueError, match="Unknown paper"): r.search("absent", "zebra")
        with pytest.raises(ValueError, match="Unknown passage"): r.inspect("absent")
        with pytest.raises(ValueError, match="positive"): r.search("p1", "zebra", 0)
    finally:
        r.close()


def test_dense_and_hybrid_differ_without_labels(passages):
    e = embeddings(passages)
    dense = QdrantRetriever(passages, ["p1", "p2"], "revision", e, "dense")
    hybrid = QdrantRetriever(passages, ["p1", "p2"], "revision", e, "hybrid")
    try:
        assert dense.search("p1", "zebra")[0].passage.passage_id.endswith(":2")
        hits = hybrid.search("p1", "zebra")
        assert hits[0].passage.passage_id.endswith(":0")
        assert all(h.passage.paper_id == "p1" for h in hits)
        assert not any("answer" in key or "query" in key for key in hybrid.client.scroll("paper_0")[0][0].payload)
        assert set(hybrid.client.scroll("paper_0")[0][0].payload) == {"paper_id", "passage_id"}
        with pytest.raises(ValueError, match="Query embedding absent"): dense.search("p1", "not cached")
    finally:
        dense.close(); hybrid.close()


def test_fusion_is_hand_calculated_and_deduplicated():
    assert fuse([("a", 10), ("b", 5)], [("b", 0.9), ("a", 0.5)]) == [
        ("a", 1 / 61 + 1 / 62), ("b", 1 / 61 + 1 / 62)]
    assert fuse([], [("b", 0.9)]) == [("b", 1 / 61)]


@pytest.mark.parametrize("mutation", ["revision", "hash", "dimension", "nan", "zero", "ids"])
def test_embedding_integrity(passages, mutation):
    e = embeddings(passages)
    if mutation == "revision": e["corpus_revision"] = "other"
    elif mutation == "dimension": e["model"]["dimension"] = 3
    elif mutation == "nan": e["passages"][passages[0]["passage_id"]] = [float("nan"), 1]
    elif mutation == "zero": e["passages"][passages[0]["passage_id"]] = [0, 0]
    elif mutation == "ids": e["passages"].pop(passages[0]["passage_id"])
    if mutation != "hash":
        e["sha256"] = digest(canonical({k: v for k, v in e.items() if k != "sha256"}).encode())
    else: e["sha256"] = "corrupt"
    with pytest.raises(ValueError): QdrantRetriever(passages, ["p1", "p2"], "revision", e, "dense")


def test_paper_statistics_are_isolated(passages):
    one = QdrantRetriever(passages[:3], ["p1"], "revision")
    two = QdrantRetriever(passages, ["p1", "p2"], "revision")
    try:
        assert [h.model_dump() for h in one.search("p1", "zebra")] == [h.model_dump() for h in two.search("p1", "zebra")]
    finally: one.close(); two.close()


def test_local_retrieval_needs_no_network(passages, monkeypatch):
    def reject(*args, **kwargs): raise AssertionError("Unexpected network connection")
    monkeypatch.setattr(socket.socket, "connect", reject)
    r = QdrantRetriever(passages, ["p1", "p2"], "revision")
    try: assert r.search("p1", "zebra")
    finally: r.close()


def test_empty_paper_duplicate_id_and_missing_embeddings(passages):
    r = QdrantRetriever([], ["empty"], "revision")
    try: assert r.search("empty", "anything") == []
    finally: r.close()
    with pytest.raises(ValueError, match="Duplicate"): QdrantRetriever(passages + [copy.deepcopy(passages[0])], ["p1", "p2"], "revision")
    with pytest.raises(ValueError, match="require"): QdrantRetriever(passages, ["p1", "p2"], "revision", mode="dense")
