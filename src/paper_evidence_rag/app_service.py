"""Model-independent application operations shared by the Gradio UI and tests."""
import copy
import json
from pathlib import Path
import threading
import time

from .guardrails import QueryGate
from .ingestion import canonical, digest
from .pdf_ingestion import UploadedPaper, extract_pdf
from .qdrant_retrieval import QdrantRetriever, validate_embeddings, vector
from .schema import Passage
from .workbench import Workbench
from .provider_status import LocalCallLimit, PipelineStageFailure, ProviderRateLimit, ProviderUnavailable


class DemoService:
    def __init__(self, artifacts, requests=None, generated=None, embeddings=None,
                 gate=None, encoder=None, answerer=None, cloud_answerer=None, public_pdf_hashes=()):
        self.viewer = Workbench(artifacts, requests, generated)
        self.gate = gate if gate is not None else QueryGate(None)
        self.encoder, self.answerer = encoder, answerer
        self.cloud_answerer = cloud_answerer
        self.public_pdf_hashes = frozenset(public_pdf_hashes)
        self.lock = threading.RLock()
        self.embeddings = json.loads(Path(embeddings).read_text(encoding="utf-8")) if embeddings else None
        passages = [Passage.model_validate(p) for p in self.viewer.contents["passages.json"]]
        try:
            validate_embeddings(self.embeddings, passages, self.viewer.manifest["corpus_revision"])
            self.dense = QdrantRetriever(self.viewer.contents["passages.json"],
                self.viewer.manifest["paper_ids"], self.viewer.manifest["corpus_revision"],
                self.embeddings, "dense") if self.embeddings else None
        except Exception:
            self.viewer.close()
            if self.cloud_answerer and hasattr(self.cloud_answerer, "close"):
                self.cloud_answerer.close()
            raise
        self.papers = {p["paper_id"]: p for p in self.viewer.contents["papers.json"]}

    def close(self):
        with self.lock:
            if self.dense:
                self.dense.close()
            self.viewer.close()
            if self.cloud_answerer and hasattr(self.cloud_answerer, "close"):
                self.cloud_answerer.close()

    def catalog(self):
        return self.viewer.catalog()

    def paper(self, paper_id, uploaded=None):
        if uploaded is not None:
            if not isinstance(uploaded, UploadedPaper) or paper_id != uploaded.paper_id:
                raise ValueError("Select a paper in this session.")
            return {"paper_id": uploaded.paper_id, "title": uploaded.title, "abstract": uploaded.abstract}
        if paper_id not in self.papers:
            raise ValueError("Select a known example paper.")
        return self.papers[paper_id]

    def browse(self, paper_id, uploaded=None):
        self.paper(paper_id, uploaded)
        sources = uploaded.passages if uploaded else [p for p in self.viewer.contents["passages.json"]
                                                      if p["paper_id"] == paper_id]
        return copy.deepcopy(sources)

    def example(self, paper_id, question, mode="dense", include_raw=False):
        """Fixed examples are an explicit allowlist, not a free-text gate bypass."""
        self.paper(paper_id)
        if (paper_id, question) not in self.viewer.cached:
            raise ValueError("Choose a saved example for this paper.")
        with self.lock:
            result = self.viewer.query(paper_id, question, include_raw)
            hits = self._search(paper_id, question, mode, 5)
            return {**result, "hits": hits, "method": mode, "origin": "saved",
                    "corpus_revision": self.viewer.manifest["corpus_revision"]}

    def _search(self, paper_id, question, mode, k):
        if mode not in ("dense", "bm25"):
            raise ValueError("Choose dense or BM25 search.")
        if mode == "bm25":
            return [h.model_dump() for h in self.viewer.retriever.search(paper_id, question, k)]
        if not self.dense:
            raise RuntimeError("Dense search requires the matching frozen vectors.")
        key = digest(question.encode("utf-8"))
        if key in self.embeddings["queries"]:
            return [h.model_dump() for h in self.dense.search(paper_id, question, k)]
        if self.encoder is None:
            raise RuntimeError("CPU question encoder is unavailable. Open a saved example or choose BM25.")
        values = vector(self.encoder.encode_question(question), self.embeddings["model"]["dimension"])
        # A request's vector is temporary. Keep the frozen artifact and its sealed
        # hash unchanged, and serialize shared Qdrant/model access with the lock.
        self.embeddings["queries"][key] = values
        try:
            return [h.model_dump() for h in self.dense.search(paper_id, question, k)]
        finally:
            del self.embeddings["queries"][key]

    def ask(self, paper_id, question, mode="dense", uploaded=None, generate=False, execution_mode=None):
        if type(generate) is not bool:
            raise ValueError("Generate must be a boolean.")
        paper = self.paper(paper_id, uploaded)
        if execution_mode not in (None, "gpu", "cpu"):
            raise ValueError("Choose GPU or CPU execution.")
        answerer = self.answerer
        if execution_mode == "cpu":
            answerer = self.cloud_answerer or (self.answerer if getattr(self.answerer, "device", "cpu") != "cuda" else None)
        elif execution_mode == "gpu" and getattr(answerer, "device", None) != "cuda":
            raise RuntimeError("GPU answer runtime is unavailable. Turn off Use HF GPU to use the CPU and hosted-model path.")
        started = time.perf_counter()
        with self.lock:
            decision = self.gate.screen(question, paper["title"], paper["abstract"])
            if decision.action != "allow":
                return {"state": "query_" + decision.action, "guardrail": decision.model_dump(),
                        "question": question, "paper_id": paper_id, "hits": [], "origin": "live"}
            if mode not in ("dense", "bm25"):
                raise ValueError("Choose dense or BM25 search.")
            k = 20 if generate and answerer else 5
            if uploaded:
                hits = self._uploaded_search(uploaded, question, mode, k)
                revision = (uploaded.dense_cache["revision"] if mode == "dense" and uploaded.dense_cache
                    else uploaded.revision)
            else:
                hits = self._search(paper_id, question, mode, k)
                revision = self.viewer.manifest["corpus_revision"]
            result = {"state": "retrieval_only" if hits else "no_evidence", "hits": hits[:5],
                "paper_id": paper_id, "question": question, "method": mode, "origin": "live",
                "corpus_revision": revision, "guardrail": decision.model_dump(),
                "notice": "Ranked passages only. No new answer was generated."}
            if generate and answerer and hits:
                try:
                    draft = answerer(paper_id, question, hits, revision, mode)
                    # Revalidate in the parent even if the inference worker says
                    # it succeeded. It may only quote prefixes of supplied hits.
                    accepted = self._checked_live(draft, paper_id, hits)
                    if "ranked_hits" in draft:
                        originals = {h["passage"]["passage_id"]: h["passage"] for h in hits}
                        ranked = draft["ranked_hits"]
                        if len(ranked) != len(hits) or {h["passage"]["passage_id"] for h in ranked} != set(originals):
                            raise ValueError("Reranker changed the evidence population")
                        if any(h["passage"] != originals[h["passage"]["passage_id"]] for h in ranked):
                            raise ValueError("Reranker changed original source text")
                        accepted["hits"] = ranked[:5]
                        accepted["score_kind"] = "reranker"
                    accepted["answer_origin"] = getattr(answerer, "origin", "Qwen3")
                    accepted["execution_mode"] = execution_mode
                    accepted["model"] = draft.get("model")
                    result.update(accepted)
                except LocalCallLimit as error:
                    result["notice"] = ("CPU search completed and sources are available. The local daily hosted-call limit is exhausted, so reranking or generation did not continue. No new answer was produced. Saved examples and browsing remain available."
                        if error.kind == "daily" else "CPU search completed and sources are available. Hosted calls are cooling down, so reranking or generation did not continue. No new answer was produced. Saved examples and browsing remain available.")
                except (ProviderRateLimit, ProviderUnavailable, RuntimeError, OSError, ValueError, TimeoutError, KeyError, TypeError) as error:
                    stage = getattr(error, "stage", None)
                    retry = getattr(error, "retry_after", None)
                    if getattr(error, "status_code", None) == 429 or isinstance(error, ProviderRateLimit):
                        wait = f" Try again in about {retry:g} seconds." if retry is not None else " Try again later."
                        provider = "OpenRouter " if getattr(error, "status_code", None) == 429 else "the provider "
                        failed = provider + "rate limit rejected " + (stage or "the hosted model") + "." + wait
                    elif isinstance(error, PipelineStageFailure):
                        failed = "the hosted " + (stage or "answer") + " stage failed."
                    elif isinstance(error, (ProviderUnavailable, RuntimeError, OSError)):
                        failed = "the hosted " + (stage or "answer") + " service was unavailable."
                    else:
                        failed = "the hosted answer pipeline could not complete."
                    if stage in ("reranking",) or "rerank" in str(stage):
                        lead = "CPU search completed, but "
                    elif stage == "generation":
                        lead = "CPU search and reranking completed, but "
                    else:
                        lead = "CPU search completed, but "
                    result["notice"] = lead + failed + " No fresh answer was produced. Retrieved sources remain available. Saved examples and browsing remain available."
            elif generate:
                result["notice"] = "CPU search completed and sources are available, but no hosted answer runtime is configured. No fresh answer was produced. Saved examples and browsing remain available."
            result["seconds"] = time.perf_counter() - started
            return result

    @staticmethod
    def _checked_live(draft, paper_id, hits):
        if draft.get("answer_format") == "hosted-grounded-v1":
            from .hosted_answers import validate_hosted_draft
            lookup = {h["passage"]["passage_id"]: h["passage"] for h in hits}
            presented = draft.get("presented")
            if (not isinstance(presented, list) or not 1 <= len(presented) <= 5
                    or any(not isinstance(p, dict) for p in presented)
                    or len({p.get("passage_id") for p in presented}) != len(presented)):
                raise ValueError("Invalid live evidence population")
            for p in presented:
                original = lookup.get(p.get("passage_id"))
                if original is None or p.get("paper_id") != paper_id or p != original:
                    raise ValueError("Live source changed from original evidence")
            validated = validate_hosted_draft(draft.get("raw_output"), paper_id, presented,
                output_limit_reached=draft.get("output_limit_reached", False))
            validated["prompt_version"] = draft.get("prompt_version")
            return {"state": "live_answer", "answer": validated, "presented": presented,
                "_raw": draft.get("raw_output", ""), "notice": "Quote checks establish source provenance, not factual correctness; review the answer and limitations."}
        from .answering import validate_draft
        lookup = {h["passage"]["passage_id"]: h["passage"] for h in hits}
        presented = draft["presented"]
        if len(presented) > 5 or len({p["passage_id"] for p in presented}) != len(presented):
            raise ValueError("Invalid live evidence population")
        for p in presented:
            original = lookup.get(p["passage_id"])
            if original is None or p["paper_id"] != paper_id:
                raise ValueError("Unexpected live citation")
            if {k: v for k, v in p.items() if k != "text"} != {k: v for k, v in original.items() if k != "text"}:
                raise ValueError("Live source metadata changed")
            if not p["text"] or not original["text"].startswith(p["text"]):
                raise ValueError("Live source is not an original prefix")
        validated = validate_draft(draft["raw_output"], paper_id, presented)
        if draft.get("output_limit_reached"):
            validated = {"status": "abstained", "reason": "output_limit", "claims": [], "answer": ""}
        return {"state": "live_answer", "answer": validated, "presented": presented,
                "_raw": draft["raw_output"], "notice": "Check relevance and completeness against the sources."}

    def _uploaded_search(self, paper, question, mode, k):
        embeddings = None
        if mode == "dense":
            if self.encoder is None:
                raise RuntimeError("Dense search on uploads needs the CPU encoder in the model runtime.")
            if paper.dense_cache is None or paper.dense_cache["model"] != self.encoder.metadata:
                if not hasattr(self.encoder, "split_documents"):
                    raise RuntimeError("The CPU encoder cannot split uploaded passages by its token budget.")
                splits = self.encoder.split_documents([p["text"] for p in paper.passages])
                from .isolated_encoder import MAX_TEXTS, _validate_splits
                splits = _validate_splits(splits, [p["text"] for p in paper.passages])
                passages, locations = [], {}
                for original, pieces in zip(paper.passages, splits, strict=True):
                    source_location = paper.locations.get(original["passage_id"], {})
                    is_split = len(pieces) > 1
                    for part_index, piece in enumerate(pieces):
                        passage = dict(original)
                        passage["passage_id"] = (original["passage_id"] + ":token:" + str(part_index)
                            if is_split else original["passage_id"])
                        passage["position"] = (original["position"] * 1000 + part_index
                            if is_split else original["position"])
                        passage["text"] = piece["text"]
                        passages.append(passage)
                        base = source_location.get("char_start", 0)
                        locations[passage["passage_id"]] = {
                            **source_location, "char_start": base + piece["start"],
                            "char_end": base + piece["end"],
                            "offsets": "zero-based, end-exclusive in extracted page text"}
                values = []
                texts = [p["text"] for p in passages]
                for offset in range(0, len(texts), MAX_TEXTS):
                    values.extend(self.encoder.encode_documents(texts[offset:offset + MAX_TEXTS]))
                if len(values) != len(passages):
                    raise ValueError("Encoder did not cover the uploaded paper")
                values = [vector(value, self.encoder.metadata["dimension"]) for value in values]
                dense_revision = digest(canonical({"source_revision": paper.revision,
                    "model": self.encoder.metadata,
                    "passages": [{"passage_id": p["passage_id"], "text": p["text"]}
                        for p in passages]}).encode("utf-8"))
                paper.dense_passages, paper.dense_locations = passages, locations
                paper.dense_cache = {"model": copy.deepcopy(self.encoder.metadata),
                    "revision": dense_revision,
                    "passages": {p["passage_id"]: v for p, v in zip(passages, values, strict=True)}}
            from .answering import sealed
            embeddings = sealed({"corpus_revision": paper.dense_cache["revision"], "model": self.encoder.metadata,
                "passages": paper.dense_cache["passages"],
                "queries": {digest(question.encode()): self.encoder.encode_question(question)}})
        # Each uploaded collection is request-local. It never enters the shared
        # example corpus or another visitor's state, including on failure.
        passages = paper.dense_passages if mode == "dense" and paper.dense_passages else paper.passages
        revision = paper.dense_cache["revision"] if mode == "dense" else paper.revision
        index = QdrantRetriever(passages, [paper.paper_id], revision, embeddings, mode)
        try:
            return [h.model_dump() for h in index.search(paper.paper_id, question, k)]
        finally:
            index.close()

    def passage(self, paper_id, passage_id, uploaded=None):
        self.paper(paper_id, uploaded)
        if uploaded:
            sources = uploaded.passages + (uploaded.dense_passages or [])
            found = next((p for p in sources if p["passage_id"] == passage_id), None)
            if found is None:
                raise ValueError("Source does not belong to this session's paper.")
            return copy.deepcopy(found)
        passage = self.viewer.passage(passage_id)["passage"]
        if passage["paper_id"] != paper_id:
            raise ValueError("Source does not belong to the selected paper.")
        return passage

    def upload(self, data):
        paper = extract_pdf(data)
        paper.public_source = digest(data) in self.public_pdf_hashes
        return paper
