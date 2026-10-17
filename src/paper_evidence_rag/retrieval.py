import re

from .schema import EvidenceHit, Passage


def tokenize(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class Retriever:
    def __init__(self, passages, paper_ids, corpus_revision):
        from rank_bm25 import BM25Okapi
        self.revision = corpus_revision
        self.by_id = {}
        self.papers = {paper_id: [] for paper_id in paper_ids}
        for value in passages:
            p = Passage.model_validate(value)
            if p.passage_id in self.by_id:
                raise ValueError("Duplicate passage ID")
            self.by_id[p.passage_id] = p
            self.papers[p.paper_id].append(p)
        self.indexes = {}
        self.vocabularies = {}
        for paper_id, items in self.papers.items():
            tokens = [tokenize(p.text) for p in items]
            self.vocabularies[paper_id] = {token for row in tokens for token in row}
            self.indexes[paper_id] = BM25Okapi(tokens, k1=1.5, b=0.75) if self.vocabularies[paper_id] else None

    def inspect(self, passage_id):
        if passage_id not in self.by_id:
            raise ValueError(f"Unknown passage ID: {passage_id}")
        return self.by_id[passage_id]

    def search(self, paper_id, question, k=5):
        if paper_id not in self.papers:
            raise ValueError(f"Unknown paper ID: {paper_id}")
        if k < 1:
            raise ValueError("k must be positive")
        tokens = tokenize(question)
        if not set(tokens) & self.vocabularies[paper_id]:
            return []
        scores = self.indexes[paper_id].get_scores(tokens)
        ranked = sorted(zip(self.papers[paper_id], scores), key=lambda pair: (-pair[1], pair[0].passage_id))[:k]
        return [EvidenceHit(rank=i, score=float(score), corpus_revision=self.revision, passage=p)
                for i, (p, score) in enumerate(ranked, 1)]
