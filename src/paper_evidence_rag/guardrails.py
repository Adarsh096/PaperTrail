"""Query screening before any evidence work; no model imports in this module."""
from collections import Counter
import math
import time
from typing import Literal

from pydantic import Field

from .schema import Record
from .ingestion import canonical, digest
from .provider_status import LocalCallLimit, ProviderRateLimit, ProviderUnavailable

POLICY_VERSION = "paper-query-v1"
LABELS = ("allow", "unsafe", "override", "off_topic", "ambiguous")
REASONS = {
    "allow": "Research question about the selected paper; evidence may still be absent.",
    "unsafe": "Request seeks actionable harmful assistance rather than research discussion.",
    "override": "Request attempts to override instructions or expose protected information.",
    "off_topic": "Request is unrelated to the selected paper.",
    "ambiguous": "Clarify the question or select the relevant paper.",
}
# The same descriptions are supplied to every candidate. Labels are mutually
# exclusive for scoring, not a claim that real prompts have only one risk.
DESCRIPTIONS = {
    "allow": "a legitimate research question about the selected paper, including sensitive academic analysis or questions whose answer may be missing",
    "unsafe": "a request for actionable assistance to commit harm, fraud, theft or abuse, rather than academic analysis",
    "override": "an instruction to ignore rules, reveal secrets or system instructions, or bypass the research assistant's restrictions",
    "off_topic": "an unrelated request such as shopping, entertainment, personal advice or a different research topic",
    "ambiguous": "a vague or underspecified question whose relationship to the selected paper needs clarification",
}


class Prediction(Record):
    # Require a complete distribution so confidence and alternate labels remain
    # inspectable. Confidence is a model score, not a calibrated probability.
    scores: dict[str, float]

    def checked(self):
        if set(self.scores) != set(LABELS):
            raise ValueError("Expected the complete guardrail label set")
        if any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.scores.values()):
            raise ValueError("Guardrail scores must be finite values in [0, 1]")
        if not math.isclose(sum(self.scores.values()), 1, abs_tol=1e-4):
            raise ValueError("Guardrail scores must sum to one")
        return self


class Decision(Record):
    action: Literal["allow", "reject", "clarify"]
    category: str
    reason: str
    status: Literal["passed", "invalid_input", "unavailable", "timeout", "malformed_output"]
    policy_version: str = POLICY_VERSION
    scores: dict[str, float] = Field(default_factory=dict)
    elapsed_seconds: float = Field(default=0.0, ge=0, allow_inf_nan=False)


def query_state(question, title, abstract):
    """Place the query first; never silently shorten an attack-bearing query."""
    return {"query": question, "paper_title": title, "paper_abstract": abstract}


class QueryGate:
    def __init__(self, predictor, min_confidence=0.0, response_budget_seconds=5.0):
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("Invalid guardrail confidence threshold")
        if not math.isfinite(response_budget_seconds) or response_budget_seconds <= 0:
            raise ValueError("Invalid guardrail response budget")
        self.predictor = predictor
        self.min_confidence = min_confidence
        self.response_budget_seconds = response_budget_seconds

    def screen(self, question, title, abstract):
        started = time.perf_counter()
        if not isinstance(question, str) or not question.strip() or len(question) > 2000 or any(ord(c) < 32 and c not in "\n\r\t" for c in question):
            return Decision(action="reject", category="invalid_input", reason="Enter a nonblank question of at most 2000 characters without control bytes.", status="invalid_input")
        if not isinstance(title, str) or not isinstance(abstract, str):
            raise ValueError("Paper context must be text")
        if self.predictor is None:
            return Decision(action="clarify", category="unavailable", reason="Query screening is not configured; CPU search and generation were not started.", status="unavailable")
        try:
            prediction = Prediction.model_validate(self.predictor(query_state(question, title, abstract))).checked()
            elapsed = time.perf_counter() - started
            if elapsed > self.response_budget_seconds:
                raise TimeoutError("Guardrail response budget exceeded")
            # A tie is uncertainty, not a reason to admit the lexical first
            # label. Keep the winner deterministic, then clarify tied maxima.
            category = min(prediction.scores, key=lambda k: (-prediction.scores[k], k))
            maximum = prediction.scores[category]
            if sum(value == maximum for value in prediction.scores.values()) > 1 or maximum < self.min_confidence:
                category = "ambiguous"
            return Decision(action="allow" if category == "allow" else "clarify" if category == "ambiguous" else "reject",
                category=category, reason=REASONS[category], status="passed", scores=prediction.scores, elapsed_seconds=elapsed)
        except LocalCallLimit as error:
            reason = ("The local daily query-screening limit is exhausted; no Groq screening request was sent, and CPU search and generation were not started."
                      if error.kind == "daily" else
                      "Query screening is cooling down; no Groq screening request was sent, and CPU search and generation were not started.")
            return Decision(action="clarify", category="local_limit", reason=reason, status="unavailable", elapsed_seconds=time.perf_counter()-started)
        except ProviderRateLimit as error:
            wait = f" Try again in about {error.retry_after:g} seconds." if error.retry_after is not None else " Try again later."
            return Decision(action="clarify", category="provider_rate_limit", reason="Groq rejected the screening request with a rate limit; CPU search and generation were not started." + wait, status="unavailable", elapsed_seconds=time.perf_counter()-started)
        except ProviderUnavailable:
            return Decision(action="clarify", category="provider_unavailable", reason="Groq screening failed; CPU search and generation were not started.", status="unavailable", elapsed_seconds=time.perf_counter()-started)
        except TimeoutError:
            status = "timeout"
        except (ValueError, TypeError, KeyError):
            status = "malformed_output"
        except (OSError, RuntimeError):
            status = "unavailable"
        # Do not expose exception messages, credentials or a malformed draft in
        # the public result. The caller retains private per-run diagnostics.
        return Decision(action="clarify", category=status, reason="Query screening failed; CPU search and generation were not started.", status=status, elapsed_seconds=time.perf_counter()-started)


def summarize(rows):
    """Keep denominators visible and model failures separate from category errors."""
    if not rows or len({r["case_id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty unique guardrail cases")
    for row in rows:
        if row["expected"] not in LABELS:
            raise ValueError("Unknown expected guardrail category")
        decision = Decision.model_validate(row["decision"])
        if decision.status == "passed":
            Prediction(scores=decision.scores).checked()
            if decision.category not in LABELS:
                raise ValueError("Unknown predicted guardrail category")
            action = "allow" if decision.category == "allow" else "clarify" if decision.category == "ambiguous" else "reject"
            if decision.action != action:
                raise ValueError("Guardrail category and action disagree")
        elif decision.action == "allow":
            raise ValueError("Failed screening cannot allow a query")
    categories = {}
    for category in LABELS:
        subset = [r for r in rows if r["expected"] == category]
        categories[category] = {"count": len(subset), "correct": sum(r["decision"]["status"] == "passed" and r["decision"]["category"] == category for r in subset)}
    valid = [r for r in rows if r["expected"] == "allow"]
    blocked = [r for r in rows if r["expected"] != "allow"]
    times = sorted(r["decision"]["elapsed_seconds"] for r in rows)
    return {"count": len(rows), "category_results": categories,
        "correct": sum(v["correct"] for v in categories.values()),
        "false_acceptance": {"count": sum(r["decision"]["action"] == "allow" for r in blocked), "denominator": len(blocked)},
        "false_rejection": {"count": sum(r["decision"]["action"] == "reject" for r in valid), "denominator": len(valid)},
        "valid_not_allowed": {"count": sum(r["decision"]["action"] != "allow" for r in valid), "denominator": len(valid)},
        "statuses": dict(Counter(r["decision"]["status"] for r in rows)),
        "downstream_queries_blocked": sum(r["decision"]["action"] != "allow" for r in rows),
        "latency_seconds": {"p50": times[math.ceil(len(times)*.5)-1], "p95": times[math.ceil(len(times)*.95)-1]},
        "warning": "Small labelled debugging set; model scores are uncalibrated and latency is hardware-specific."}


def evaluate_cases(cases, predictor, model, code_revision, min_confidence=0.0):
    """Run labelled cases without exposing labels or answers to the predictor."""
    source = cases.get("rows", [])
    if not source or len({r["case_id"] for r in source}) != len(source):
        raise ValueError("Expected nonempty unique guardrail cases")
    for row in source:
        if row["expected"] not in LABELS or row["split"] not in ("development", "check"):
            raise ValueError("Invalid guardrail category or evaluation split")
        if set(row["state"]) != {"query", "paper_title", "paper_abstract"} or any(not isinstance(v, str) for v in row["state"].values()):
            raise ValueError("Invalid paper query state")
    gate = QueryGate(predictor, min_confidence=min_confidence)
    rows = []
    for row in source:
        state = row["state"]
        decision = gate.screen(state["query"], state["paper_title"], state["paper_abstract"])
        rows.append({**row, "decision": decision.model_dump()})
    return {"policy_version": POLICY_VERSION, "model": model, "code_revision": code_revision,
        "cases_sha256": digest(canonical(cases).encode()), "min_confidence": min_confidence,
        "rows": rows, "aggregate": {s: summarize([r for r in rows if r["split"] == s]) for s in ("development", "check") if any(r["split"] == s for r in rows)}}
