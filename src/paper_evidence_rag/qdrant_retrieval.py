"""Offline Qdrant local-mode retrieval with paper-local BM25 statistics."""
from collections import Counter
import math

import numpy as np
from qdrant_client import QdrantClient, models

from .ingestion import canonical, digest
from .retrieval import tokenize
from .schema import EvidenceHit, Passage


CONFIG = {"backend": "qdrant-local", "sparse": "BM25 TF with Qdrant IDF",
          "k1": 1.5, "b": 0.75, "tokenizer": "lowercase Unicode \\w+",
          "stemming": False, "stopwords": False, "statistics": "one collection per paper",
          "query_tf": "counts", "dense_distance": "cosine", "rrf_k": 60,
          "candidate_depth": "all passages in selected paper", "ties": "passage ID"}


def vector(values, dimension):
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (dimension,) or not np.isfinite(array).all() or np.linalg.norm(array) == 0:
        raise ValueError("Expected a finite, nonzero dense vector of the configured dimension")
    return array.tolist()


def validate_embeddings(embeddings, passages, revision):
    if embeddings is None:
        return
    if embeddings.get("corpus_revision") != revision:
        raise ValueError("Embedding corpus revision mismatch")
    unsigned = {k: v for k, v in embeddings.items() if k != "sha256"}
    if digest(canonical(unsigned).encode()) != embeddings.get("sha256"):
        raise ValueError("Embedding SHA-256 mismatch")
    meta = embeddings["model"]
    if not meta.get("id") or len(meta.get("revision", "")) != 40 or meta.get("dimension", 0) < 1:
        raise ValueError("Embedding model ID, pinned revision and dimension are required")
    if set(embeddings["passages"]) != {p.passage_id for p in passages}:
        raise ValueError("Embedding passage IDs do not match corpus")
    for values in [*embeddings["passages"].values(), *embeddings["queries"].values()]:
        vector(values, meta["dimension"])


def fuse(sparse, dense, rrf_k=60):
    scores = Counter()
    for ranking in (sparse, dense):
        for rank, (passage_id, _) in enumerate(ranking, 1):
            scores[passage_id] += 1 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class QdrantRetriever:
    def __init__(self, passages, paper_ids, corpus_revision, embeddings=None, mode="bm25"):
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        self.mode, self.revision, self.embeddings = mode, corpus_revision, embeddings
        self.by_id = {}
        self.papers = {paper_id: [] for paper_id in paper_ids}
        for value in passages:
            p = Passage.model_validate(value)
            if p.passage_id in self.by_id or p.paper_id not in self.papers:
                raise ValueError("Duplicate passage ID or unknown paper")
            self.by_id[p.passage_id] = p
            self.papers[p.paper_id].append(p)
        validate_embeddings(embeddings, list(self.by_id.values()), corpus_revision)
        if mode != "bm25" and embeddings is None:
            raise ValueError("Dense and hybrid retrieval require pinned embedding artifacts")
        self.client = QdrantClient(":memory:")
        self.collections, self.vocabularies, self.statistics = {}, {}, {}
        try:
            self._build()
        except Exception:
            self.close()
            raise

    def _build(self):
        for number, (paper_id, passages) in enumerate(sorted(self.papers.items())):
            # Collections, vocabulary IDs and length statistics are paper-local.
            # Filtering a shared collection would not make its IDF paper-local.
            tokens = [tokenize(p.text) for p in passages]
            vocabulary = {token: index for index, token in enumerate(sorted({t for row in tokens for t in row}))}
            self.vocabularies[paper_id] = vocabulary
            average = sum(map(len, tokens)) / len(tokens) if tokens else 0
            self.statistics[paper_id] = {"documents": len(tokens), "average_tokens": average,
                                         "vocabulary_size": len(vocabulary)}
            if not passages:
                continue
            collection = f"paper_{number}"
            self.collections[paper_id] = collection
            self.client.create_collection(collection, vectors_config={"dense": models.VectorParams(
                size=self.embeddings["model"]["dimension"], distance=models.Distance.COSINE)} if self.embeddings else {},
                sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)})
            points = []
            for point_id, (p, row) in enumerate(zip(passages, tokens)):
                counts = Counter(row)
                indices = sorted(vocabulary[t] for t in counts)
                weights = {vocabulary[t]: f * 2.5 / (f + 1.5 * (0.25 + 0.75 * len(row) / average))
                           for t, f in counts.items()} if average else {}
                # Store the BM25 term-frequency/length factor. Qdrant's IDF
                # modifier supplies the collection's inverse document frequency.
                values = {"bm25": models.SparseVector(indices=indices, values=[weights[i] for i in indices])}
                if self.embeddings:
                    values["dense"] = vector(self.embeddings["passages"][p.passage_id], self.embeddings["model"]["dimension"])
                points.append(models.PointStruct(id=point_id, vector=values,
                    payload={"passage_id": p.passage_id, "paper_id": paper_id}))
            self.client.upsert(collection, points=points, wait=True)

    def close(self):
        self.client.close()

    def inspect(self, passage_id):
        if passage_id not in self.by_id:
            raise ValueError(f"Unknown passage ID: {passage_id}")
        return self.by_id[passage_id]

    def _rank(self, paper_id, query, using):
        # Retrieve the full paper before sorting ties, so a backend cutoff cannot
        # choose an arbitrary member of a tie. Suitable for this bounded corpus.
        hits = self.client.query_points(self.collections[paper_id], query=query, using=using,
            query_filter=models.Filter(must=[models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))]),
            limit=len(self.papers[paper_id]), with_payload=True).points
        ranking = [(h.payload["passage_id"], float(h.score)) for h in hits]
        if any(self.inspect(pid).paper_id != paper_id or not math.isfinite(score) for pid, score in ranking):
            raise ValueError("Invalid Qdrant result")
        return sorted(ranking, key=lambda item: (-item[1], item[0]))

    def search(self, paper_id, question, k=5):
        if paper_id not in self.papers:
            raise ValueError(f"Unknown paper ID: {paper_id}")
        if k < 1:
            raise ValueError("k must be positive")
        tokens = tokenize(question)
        if not tokens or paper_id not in self.collections:
            return []
        vocabulary = self.vocabularies[paper_id]
        counts = Counter(t for t in tokens if t in vocabulary)
        sparse = []
        if counts and self.mode != "dense":
            indices = sorted(vocabulary[t] for t in counts)
            weights = {vocabulary[t]: count for t, count in counts.items()}
            sparse = self._rank(paper_id, models.SparseVector(indices=indices, values=[weights[i] for i in indices]), "bm25")
        dense = []
        if self.mode != "bm25":
            key = digest(question.encode("utf-8"))
            if key not in self.embeddings["queries"]:
                raise ValueError("Query embedding absent; encode this exact question offline first")
            dense = self._rank(paper_id, vector(self.embeddings["queries"][key], self.embeddings["model"]["dimension"]), "dense")
        ranked = sparse if self.mode == "bm25" else dense if self.mode == "dense" else fuse(sparse, dense)
        return [EvidenceHit(rank=i, score=score, corpus_revision=self.revision, passage=self.inspect(pid))
                for i, (pid, score) in enumerate(ranked[:k], 1)]
