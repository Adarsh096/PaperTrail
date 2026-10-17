"""Opt-in, zero-price hosted stages. No requests occur at import or setup time.

Requests send live questions and relevant document excerpts to hosted providers.
The application does not verify document origin or classify uploads as public.
"""
import copy
import math
import re
import sqlite3
import threading
import time

from .hosted_answers import ANSWER_FORMAT, PROMPT_VERSION, SYSTEM, schema
from .guardrails import Decision, QueryGate
from .hosted_guardrail import CallBudget
from .ingestion import canonical
from .provider_status import LocalCallLimit, PipelineStageFailure, ProviderRateLimit, ProviderUnavailable

SAFETY_MODEL = "nvidia/nemotron-3.5-content-safety:free"
RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
GENERATION_MODELS = ("nvidia/nemotron-3.5-super-vl:free", "nvidia/nemotron-3-super-120b-a12b:free")


class HostedRequestError(RuntimeError):
    def __init__(self, status_code=None, stage="safety", retry_after=None):
        super().__init__("Hosted model request unavailable")
        self.status_code = status_code if type(status_code) is int else None
        self.stage, self.retry_after = stage, retry_after


class PacedBudget:
    """One shared request counter for safety, reranking and generation.

    Reserve failed calls too, serialize the local cooldown, and never retry a
    provider request. SQLite persistence is only as durable as its host volume.
    """
    def __init__(self, path, daily_limit=30, interval=6.0, clock=time.time, sleep=time.sleep):
        self.budget = CallBudget(path, daily_limit, interval, clock)
        self.clock, self.sleep, self.lock = clock, sleep, threading.Lock()

    def reserve(self):
        with self.lock:
            with sqlite3.connect(self.budget.path) as db:
                latest = db.execute("SELECT MAX(at) FROM reservations").fetchone()[0]
            delay = 0 if latest is None else max(0, self.budget.interval - (self.clock() - latest))
            if delay:
                self.sleep(min(delay, self.budget.interval))
            self.budget.reserve()


class OpenRouterClient:
    """Fixed destination and models, explicit zero prices, no SDK retries."""
    def __init__(self, key, budget, client=None):
        import httpx
        if not isinstance(key, str) or not key.strip():
            raise ValueError("OpenRouter secret is missing")
        self.budget = budget
        self.client = client or httpx.Client(timeout=20.0, follow_redirects=False,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})

    def request(self, stage, payload):
        expected = {"safety": ("/chat/completions", (SAFETY_MODEL,)),
                    "rerank": ("/rerank", (RERANK_MODEL,)),
                    "generation": ("/chat/completions", GENERATION_MODELS)}
        if stage not in expected or payload.get("model") not in expected[stage][1]:
            raise ValueError("Unapproved hosted model or stage")
        body = copy.deepcopy(payload)
        if "provider" in body or "plugins" in body or "models" in body:
            raise ValueError("Routing overrides are not allowed")
        body["provider"] = {"only": ["nvidia"], "allow_fallbacks": False,
            "require_parameters": True, "max_price": {"prompt": 0, "completion": 0, "request": 0}}
        if len(canonical(body).encode()) > 256_000:
            raise ValueError("Hosted request exceeds the byte budget")
        try:
            self.budget.reserve()
        except LocalCallLimit:
            raise
        except Exception:
            raise ProviderUnavailable({"safety": "safety", "rerank": "reranking", "generation": "generation"}[stage]) from None
        response = None
        try:
            response = self.client.post("https://openrouter.ai/api/v1" + expected[stage][0], json=body)
            response.raise_for_status()
            result = response.json()
        except Exception:
            # Provider exceptions can include payloads or keys. Never expose
            # their text, retry automatically, or switch to a paid model.
            status = getattr(response, "status_code", None)
            retry = None
            if status == 429:
                try:
                    retry = float(response.headers.get("retry-after"))
                    if not 0 < retry <= 3600:
                        retry = None
                except (AttributeError, TypeError, ValueError):
                    retry = None
            raise HostedRequestError(status, {"safety": "safety", "rerank": "reranking", "generation": "generation"}[stage], retry) from None
        if not isinstance(result, dict) or "error" in result:
            raise ValueError("Invalid hosted response")
        return result

    def close(self):
        self.client.close()

    def limits(self):
        """Read counters only, never return the secret or the key's label."""
        try:
            response = self.client.get("https://openrouter.ai/api/v1/key")
            response.raise_for_status()
            data = response.json()["data"]
            return {k: data.get(k) for k in ("is_free_tier", "free_model_daily_requests", "usage_daily", "limit_remaining")}
        except Exception:
            raise RuntimeError("Hosted quota read unavailable") from None


def completed_text(response):
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Missing completion")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("Missing completion message")
    content = choice["message"].get("content")
    if choice.get("finish_reason") != "stop" or not isinstance(content, str) or not content.strip():
        raise ValueError("Incomplete completion")
    return content


def parse_safety(text):
    """Read NVIDIA's native labels, never match 'safe' inside 'unsafe'."""
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError("Invalid safety response")
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines or re.fullmatch(r"User Safety: (safe|unsafe)", lines[0]) is None:
        raise ValueError("Missing native safety decision")
    if len(lines) > 2 or (len(lines) == 2 and re.fullmatch(r"Safety Categories: [^\r\n]+", lines[1]) is None):
        raise ValueError("Unexpected safety output")
    return lines[0].split(": ", 1)[1]


class SafetyFirstGate:
    def __init__(self, client, relevance):
        self.client, self.relevance = client, relevance

    def screen(self, question, title, abstract):
        local = QueryGate(None).screen(question, title, abstract)
        if local.status == "invalid_input":
            return local
        try:
            response = self.client.request("safety", {"model": SAFETY_MODEL,
                "messages": [{"role": "user", "content": question}],
                "temperature": 0, "max_tokens": 256, "reasoning": {"enabled": False}})
            safety = parse_safety(completed_text(response))
        except LocalCallLimit as error:
            reason = "OpenRouter's daily app limit is exhausted; no safety or Groq request was sent, and CPU search and generation were not started." if error.kind == "daily" else "OpenRouter safety screening is cooling down; no safety or Groq request was sent, and CPU search and generation were not started."
            return Decision(action="clarify", category="local_limit", status="unavailable", reason=reason)
        except HostedRequestError as error:
            if error.status_code == 429:
                wait = f" Try again in about {error.retry_after:g} seconds." if error.retry_after is not None else " Try again later."
                reason = "OpenRouter's safety endpoint rejected the request with a rate limit; Groq was not called and CPU search was not started." + wait
                return Decision(action="clarify", category="provider_rate_limit", status="unavailable", reason=reason)
            return Decision(action="clarify", category="provider_unavailable", status="unavailable", reason="Safety screening failed; Groq was not called and evidence processing was not started.")
        except ProviderRateLimit as error:
            wait = f" Try again in about {error.retry_after:g} seconds." if error.retry_after is not None else " Try again later."
            return Decision(action="clarify", category="provider_rate_limit", status="unavailable", reason="OpenRouter's safety endpoint rejected the request with a rate limit; Groq was not called and CPU search was not started." + wait)
        except ProviderUnavailable:
            return Decision(action="clarify", category="provider_unavailable", status="unavailable", reason="OpenRouter safety screening failed; Groq was not called and CPU search was not started.")
        except (ValueError, RuntimeError, OSError):
            return Decision(action="clarify", category="safety_unavailable", status="unavailable",
                reason="Safety screening failed; Groq was not called and CPU search was not started.")
        if safety == "unsafe":
            return Decision(action="reject", category="unsafe", status="passed",
                reason="Safety screening rejected this request; Groq was not called and CPU search was not started.")
        # This is deliberately sequential. A blocked or failed safety check
        # must not spend Groq quota or reach embedding/search/model stages.
        return self.relevance.screen(question, title, abstract)


def checked_ranking(response, hits):
    """Resolve result indices to original sources; reject partial/duplicate IDs."""
    rows = response.get("results")
    if not isinstance(rows, list) or len(rows) != len(hits):
        raise ValueError("Incomplete reranker coverage")
    seen, ranked = set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid reranker row")
        index, score = row.get("index"), row.get("relevance_score")
        if type(index) is not int or not 0 <= index < len(hits) or index in seen:
            raise ValueError("Invalid reranker index")
        if type(score) not in (int, float) or not math.isfinite(score):
            raise ValueError("Invalid reranker score")
        seen.add(index)
        hit = copy.deepcopy(hits[index])
        hit["retrieval_score"], hit["score"] = hit["score"], float(score)
        ranked.append(hit)
    ranked.sort(key=lambda hit: (-hit["score"], hit["passage"]["passage_id"]))
    return [{**hit, "rank": i + 1} for i, hit in enumerate(ranked)]


class CloudAnswerer:
    """Batch rerank dense candidates, generate a bounded cited draft.

    The service still validates every quote and source in the parent. This
    adapter never substitutes a generated document for the original source.
    """
    origin = "OpenRouter"
    def __init__(self, client, model=GENERATION_MODELS[0]):
        if model not in GENERATION_MODELS:
            raise ValueError("Unapproved free generator")
        self.client, self.model = client, model

    def close(self):
        self.client.close()

    def rerank(self, paper_id, question, hits):
        if not 1 <= len(hits) <= 20:
            raise ValueError("Expected one to twenty evidence candidates")
        if len({h["passage"]["passage_id"] for h in hits}) != len(hits):
            raise ValueError("Duplicate evidence candidates")
        if any(h["passage"]["paper_id"] != paper_id for h in hits):
            raise ValueError("Cross-paper evidence")
        documents = [h["passage"]["text"] for h in hits]
        # A conservative byte bound leaves room for model framing within the
        # advertised 10,240-token query/document limit. Do not truncate or drop
        # a source to conceal a provider/model incompatibility.
        if any(len((question + text).encode()) > 9000 for text in documents):
            raise ValueError("Candidate exceeds the hosted reranker byte budget")
        ranked = checked_ranking(self.client.request("rerank", {"model": RERANK_MODEL,
            "query": question, "documents": documents, "top_n": len(documents)}), hits)
        return ranked

    def generate(self, question, presented, revision):
        if not 1 <= len(presented) <= 5:
            raise ValueError("Expected one to five supplied passages")
        response = self.client.request("generation", {"model": self.model, "temperature": 0,
            "max_tokens": 1024, "reasoning": {"enabled": False},
            "messages": [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": canonical({"question": question, "evidence": [
                    {"passage_id": p["passage_id"], "kind": p["kind"],
                     "section_name": p["section_name"], "text": p["text"]} for p in presented]})}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "paper_answer",
                "strict": True, "schema": schema()}}})
        choices = response.get("choices", [])
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message")
        raw = message.get("content") if isinstance(message, dict) else None
        return {"raw_output": raw if isinstance(raw, str) else "",
            "answer_format": ANSWER_FORMAT, "prompt_version": PROMPT_VERSION,
            "output_limit_reached": choice.get("finish_reason") != "stop",
            "presented": copy.deepcopy(presented), "model": self.model,
            "provider": response.get("provider"), "response_id": response.get("id"),
            "usage": response.get("usage", {}), "corpus_revision": revision}

    def __call__(self, paper_id, question, hits, revision, mode):
        try:
            ranked = self.rerank(paper_id, question, hits)
        except (LocalCallLimit, HostedRequestError, ProviderUnavailable):
            raise
        except Exception:
            raise PipelineStageFailure("reranking") from None
        presented = [copy.deepcopy(h["passage"]) for h in ranked[:5]]
        try:
            generated = self.generate(question, presented, revision)
        except (LocalCallLimit, HostedRequestError, ProviderUnavailable):
            raise
        except Exception:
            raise PipelineStageFailure("generation") from None
        return {**generated, "ranked_hits": ranked}
