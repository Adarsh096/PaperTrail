"""Read-only loopback workbench. No model imports, tokens or inference.

    Cached answers are validated on load, then keyed by exact paper/question
    plus complete run lineage. New questions retrieve sources immediately and
    explicitly require a separate offline generation run.
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files
from pathlib import Path
import time
from urllib.parse import parse_qs, urlsplit

from .answer_evaluation import evaluate_answers
from .answering import DEFAULT_MODEL_ID, prompt_hash, sealed
from .ingestion import canonical, load_artifacts
from .qdrant_retrieval import CONFIG, QdrantRetriever


class Workbench:
    def __init__(self, artifacts, requests=None, generated=None, query_gate=None):
        self.query_gate = query_gate
        self.manifest, self.contents = load_artifacts(Path(artifacts))
        if self.manifest["paper_count"] > 20 or self.manifest["passage_count"] > 2000:
            raise ValueError("Workbench supports at most 20 papers and 2000 passages")
        if bool(requests) != bool(generated):
            raise ValueError("Provide both requests and generated output, or neither")
        self.cached, self.lineage = {}, None
        if requests:
            read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
            request, output = read(requests), read(generated)
            if output["model"]["id"] != DEFAULT_MODEL_ID:
                raise ValueError("Workbench answer cache requires the selected Qwen3 model")
            evaluate_answers(self.contents, self.manifest, request, output, "cache-replay")
            self.lineage = sealed({"corpus_revision": self.manifest["corpus_revision"],
                "input_sha256": self.manifest["input_sha256"], "model": output["model"],
                "prompt_version": request["prompt_version"], "prompt_sha256": prompt_hash(request["prompt_version"]),
                "mode": request["mode"], "candidate_sha256": request["candidate_sha256"],
                "scores_sha256": request["scores_sha256"], "request_sha256": request["sha256"],
                "generation_sha256": output["sha256"]})
            for row in output["rows"]:
                key = (row["paper_id"], row["question"])
                if key in self.cached:
                    raise ValueError("Ambiguous cached paper/question")
                self.cached[key] = row
        self.retriever = QdrantRetriever(self.contents["passages.json"], self.manifest["paper_ids"],
                                         self.manifest["corpus_revision"])

    def close(self):
        self.retriever.close()

    def catalog(self):
        return {"corpus_revision": self.manifest["corpus_revision"], "split": self.manifest["split"],
            "paper_count": self.manifest["paper_count"], "passage_count": self.manifest["passage_count"],
            "model": DEFAULT_MODEL_ID, "cached_question_count": len(self.cached),
            "papers": [{"paper_id": p["paper_id"], "title": p["title"],
                "questions": [{"question_id": r["question_id"], "question": r["question"]}
                    for r in self.cached.values() if r["paper_id"] == p["paper_id"]]}
                for p in self.contents["papers.json"]]}

    def query(self, paper_id, question, include_raw=False):
        if not isinstance(paper_id, str) or paper_id not in self.manifest["paper_ids"]:
            raise LookupError("Unknown paper ID")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("Enter a question between 1 and 2000 characters")
        if type(include_raw) is not bool:
            raise ValueError("include_raw must be a boolean")
        if self.query_gate is not None:
            paper = next(p for p in self.contents["papers.json"] if p["paper_id"] == paper_id)
            # Screening also precedes cached answers. A rejected or uncertain
            # query must not escape the gate through a cache hit.
            decision = self.query_gate.screen(question, paper["title"], paper["abstract"])
            if decision.action != "allow":
                return {"state": "query_" + decision.action, "paper_id": paper_id,
                    "question": question, "guardrail": decision.model_dump()}
        row = self.cached.get((paper_id, question))
        if row:
            result = {"state": "saved_answer", "question": question, "paper_id": paper_id,
                "answer": row["result"], "presented": row["presented"], "lineage": self.lineage,
                "notice": "Quotation checks passed only for accepted claims. Review relevance and completeness."}
            if include_raw:
                result["diagnostic"] = {"label": "Unvalidated model output", "raw_output": row["raw_output"],
                    "validation_reason": row["result"]["reason"], "output_limit_reached": row["output_limit_reached"]}
            return result
        started = time.monotonic()
        hits = self.retriever.search(paper_id, question, 5)
        if time.monotonic() - started > 5:
            raise TimeoutError("Retrieval exceeded the five-second response budget")
        return {"state": "needs_generation" if hits else "no_evidence", "paper_id": paper_id,
            "question": question, "hits": [h.model_dump() for h in hits],
            "index": {"configuration": CONFIG, "corpus_revision": self.manifest["corpus_revision"],
                      "passages_sha256": self.manifest["artifact_sha256"]["passages.json"]},
            "notice": "No saved Qwen3 answer for this exact question. These are keyword results. Encode and generate offline to add an answer."}

    def passage(self, passage_id):
        try:
            return {"corpus_revision": self.manifest["corpus_revision"],
                    "passage": self.retriever.inspect(passage_id).model_dump()}
        except ValueError as exc:
            raise LookupError("Unknown passage ID") from exc


def make_server(workbench, port=8765):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid local port")

    class Handler(BaseHTTPRequestHandler):
        # Single request worker serializes Qdrant access. Read and header waits
        # are bounded; this is a local inspection tool, not a public web host.
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass  # Do not write user questions or source text to server logs.

        def respond(self, value, status=200, content_type="application/json; charset=utf-8"):
            body = canonical(value).encode() if content_type.startswith("application/json") else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in hosts:
                self.respond({"error": "Use the loopback workbench URL"}, 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + host for host in hosts}:
                self.respond({"error": "Cross-origin requests are disabled"}, 403)
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            url = urlsplit(self.path)
            if url.path == "/":
                self.respond(b'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper evidence workbench</title><link rel="stylesheet" href="/viewer.css"><script src="/viewer.js" defer></script></head><body><main id="app"></main></body></html>', content_type="text/html; charset=utf-8")
            elif url.path in {"/viewer.js", "/viewer.css"}:
                self.respond(files("paper_evidence_rag").joinpath("assets", url.path[1:]).read_bytes(),
                    content_type="text/javascript; charset=utf-8" if url.path.endswith("js") else "text/css; charset=utf-8")
            elif url.path == "/api/catalog":
                self.respond(workbench.catalog())
            elif url.path == "/api/passage":
                try:
                    self.respond(workbench.passage(parse_qs(url.query).get("id", [""])[0]))
                except LookupError as exc:
                    self.respond({"error": str(exc)}, 404)
            else:
                self.respond({"error": "Unknown route"}, 404)

        def do_POST(self):
            if not self.allowed():
                return
            if self.path != "/api/query":
                self.respond({"error": "Unknown route"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 8192 or self.headers.get("Transfer-Encoding"):
                    self.respond({"error": "Request body must be at most 8192 bytes"}, 413)
                    return
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self.respond({"error": "Expected application/json"}, 415)
                    return
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) - {"paper_id", "question", "include_raw"}:
                    raise ValueError("Invalid query fields")
                self.respond(workbench.query(body.get("paper_id"), body.get("question"), body.get("include_raw", False)))
            except LookupError as exc:
                self.respond({"error": str(exc)}, 404)
            except (ValueError, UnicodeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except TimeoutError as exc:
                self.respond({"error": str(exc)}, 504)

    return HTTPServer(("127.0.0.1", port), Handler)


def serve(artifacts, requests=None, generated=None, port=8765, query_gate=None):
    workbench = Workbench(artifacts, requests, generated, query_gate)
    try:
        with make_server(workbench, port) as server:
            print(f"Evidence workbench: http://127.0.0.1:{server.server_port}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        workbench.close()
