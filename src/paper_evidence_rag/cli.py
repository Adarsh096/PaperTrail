import argparse
import json
import sys
from pathlib import Path

from .evaluation import evaluate
from .ingestion import build, canonical, load_artifacts
from .retrieval import Retriever
from .qdrant_retrieval import QdrantRetriever


def main():
    parser = argparse.ArgumentParser(description="Inspect paper-scoped QASPER evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    guard = commands.add_parser("screen-query", help="Screen one query using a local pinned compact classifier")
    guard.add_argument("--artifacts", type=Path, default=Path("data/build"))
    guard.add_argument("--paper", required=True)
    guard.add_argument("--question", required=True)
    guard.add_argument("--model", required=True)
    guard.add_argument("--model-directory", type=Path, required=True)
    guard.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    guard_eval = commands.add_parser("evaluate-guardrails", help="Evaluate labelled queries with the optional local classifier")
    guard_eval.add_argument("--cases", type=Path, required=True)
    guard_eval.add_argument("--model", required=True)
    guard_eval.add_argument("--model-directory", type=Path, required=True)
    guard_eval.add_argument("--code-revision", required=True)
    guard_eval.add_argument("--output", type=Path, required=True)
    guard_eval.add_argument("--min-confidence", type=float, default=0.0)
    semantic = commands.add_parser("prepare-semantic", help="Freeze validated saved answers for semantic evaluation")
    semantic.add_argument("--artifacts", type=Path, required=True)
    semantic.add_argument("--requests", type=Path, required=True)
    semantic.add_argument("--generated", type=Path, required=True)
    semantic.add_argument("--code-revision", required=True)
    semantic.add_argument("--output", type=Path, required=True)
    trace = commands.add_parser("trace-semantic", help="Trace frozen gold evidence through candidates, selection and packing")
    for name in ("inputs", "candidates", "requests", "output"):
        trace.add_argument("--" + name, type=Path, required=True)
    judge = commands.add_parser("score-semantic", help="Run or replay the bounded optional Ragas judge")
    judge.add_argument("--inputs", type=Path, required=True)
    judge.add_argument("--ledger", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--code-revision", required=True)
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--max-output-tokens", type=int, default=2048)
    judge.add_argument("--request-timeout", type=int, default=60, help="Hosted request timeout in seconds, 1 through 300")
    judge.add_argument("--structured-output", action="store_true", help="Request native JSON schema if supported by the proxy/model")
    judge.add_argument("--offline", action="store_true")
    judge.add_argument("--question-aware", action="store_true", help="Use the separate question-aware v1 rubric instead of standard Ragas metrics")
    judge.add_argument("--vectors", type=Path, help="Verified vectors for Ragas reverse questions")
    judge.add_argument("--max-calls", type=int, default=320)
    judge.add_argument("--max-token-bound", type=int, default=2_000_000)
    semantic_encoder = commands.add_parser("encode-semantic", help="Encode frozen Ragas questions with pinned EmbeddingGemma")
    semantic_encoder.add_argument("--inputs", type=Path, required=True)
    semantic_encoder.add_argument("--model-directory", type=Path, required=True)
    semantic_encoder.add_argument("--output", type=Path, required=True)
    semantic_encoder.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    b = commands.add_parser("build")
    b.add_argument("--source", type=Path, required=True)
    b.add_argument("--sha256", required=True)
    b.add_argument("--output", type=Path, default=Path("data/build"))
    b.add_argument("--paper-limit", type=int, default=5)
    server = commands.add_parser("serve", help="Open a local evidence workbench without model inference")
    server.add_argument("--artifacts", type=Path, required=True)
    server.add_argument("--requests", type=Path)
    server.add_argument("--generated", type=Path)
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--guardrail-model", help="Optional pinned local guardrail; model inference runs in this server")
    server.add_argument("--guardrail-directory", type=Path)
    e = commands.add_parser("encode")
    e.add_argument("--artifacts", type=Path, default=Path("data/build"))
    e.add_argument("--model-directory", type=Path, required=True)
    e.add_argument("--question", help="Also encode one exact question for later offline dense search")
    e.add_argument("--output", type=Path, default=Path("data/embeddings.json"))
    e.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    c = commands.add_parser("compare")
    c.add_argument("--artifacts", type=Path, default=Path("data/build"))
    c.add_argument("--embeddings", type=Path)
    c.add_argument("--code-revision", required=True)
    c.add_argument("--output", type=Path, default=Path("outputs/comparison.json"))
    for name in ("prepare-answers", "generate-answers", "evaluate-answers"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare-answers":
            command.add_argument("--candidates", type=Path, required=True)
            command.add_argument("--scores", type=Path, required=True)
            command.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="dense")
            from .answering import PROMPTS, PROMPT_VERSION
            command.add_argument("--prompt-version", choices=sorted(PROMPTS), default=PROMPT_VERSION)
        else:
            command.add_argument("--requests", type=Path, required=True)
        if name == "generate-answers":
            command.add_argument("--model-directory", type=Path, required=True)
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if name == "evaluate-answers":
            command.add_argument("--artifacts", type=Path, default=Path("data/build"))
            command.add_argument("--generated", type=Path, required=True)
            command.add_argument("--code-revision", required=True)
    for name in ("freeze-candidates", "score-candidates", "evaluate-reranker"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name != "score-candidates":
            command.add_argument("--artifacts", type=Path, default=Path("data/build"))
            command.add_argument("--embeddings", type=Path, required=True)
        if name == "freeze-candidates":
            command.add_argument("--candidate-count", type=int, default=50)
            command.add_argument("--modes", nargs="+", choices=("bm25", "dense", "hybrid"), default=["bm25", "dense", "hybrid"])
        else:
            command.add_argument("--candidates", type=Path, required=True)
        if name == "score-candidates":
            command.add_argument("--model-directory", type=Path, required=True)
            command.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        if name == "evaluate-reranker":
            command.add_argument("--scores", type=Path, required=True)
            command.add_argument("--code-revision", required=True)
    for name in ("search", "inspect", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--artifacts", type=Path, default=Path("data/build"))
        command.add_argument("--backend", choices=("qdrant", "legacy"), default="qdrant")
        command.add_argument("--mode", choices=("bm25", "dense", "hybrid"), default="bm25")
        command.add_argument("--embeddings", type=Path)
        if name == "search":
            command.add_argument("--paper", required=True)
            command.add_argument("--question", required=True)
            command.add_argument("--k", type=int, default=5)
        elif name == "inspect":
            command.add_argument("--passage", required=True)
        else:
            command.add_argument("--code-revision", required=True)
            command.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = parser.parse_args()
    try:
        if args.command == "evaluate-guardrails":
            from .guardrails import evaluate_cases
            from .guardrail_models import LocalClassifier, MODEL_PINS
            cases = json.loads(args.cases.read_text(encoding="utf-8"))
            result = evaluate_cases(cases, LocalClassifier(args.model, args.model_directory),
                {"id": args.model, "revision": MODEL_PINS[args.model]}, args.code_revision, args.min_confidence)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(canonical(result), encoding="utf-8")
            result = {"output": str(args.output), "cases_sha256": result["cases_sha256"], "count": len(result["rows"])}
        elif args.command == "screen-query":
            from .guardrails import QueryGate
            from .guardrail_models import LocalClassifier
            _, contents = load_artifacts(args.artifacts)
            paper = next((p for p in contents["papers.json"] if p["paper_id"] == args.paper), None)
            if paper is None:
                raise ValueError("Unknown paper ID")
            result = QueryGate(LocalClassifier(args.model, args.model_directory, args.device)).screen(
                args.question, paper["title"], paper["abstract"]).model_dump()
        elif args.command == "build":
            result = build(args.source, args.output, args.sha256, args.paper_limit)
        elif args.command == "serve":
            from .workbench import serve
            gate = None
            if bool(args.guardrail_model) != bool(args.guardrail_directory):
                raise ValueError("Provide both guardrail model and directory")
            if args.guardrail_model:
                from .guardrails import QueryGate
                from .guardrail_models import LocalClassifier
                gate = QueryGate(LocalClassifier(args.guardrail_model, args.guardrail_directory))
            serve(args.artifacts, args.requests, args.generated, args.port, gate)
            return
        elif args.command == "prepare-semantic":
            from .semantic_evaluation import prepare_semantic
            manifest, contents = load_artifacts(args.artifacts)
            result = prepare_semantic(contents, manifest,
                json.loads(args.requests.read_text(encoding="utf-8")),
                json.loads(args.generated.read_text(encoding="utf-8")), args.code_revision)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(canonical(result), encoding="utf-8")
            result = {"output": str(args.output), "sha256": result["sha256"], "questions": len(result["rows"])}
        elif args.command == "encode-semantic":
            from .semantic_embeddings import encode_texts
            result = encode_texts(json.loads(args.inputs.read_text(encoding="utf-8")), args.model_directory, args.device)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(canonical(result), encoding="utf-8")
            result = {"sha256": result["sha256"], "text_count": len(result["vectors"])}
        elif args.command == "trace-semantic":
            from .evaluation_hardening import trace_frozen
            read = lambda path: json.loads(path.read_text(encoding="utf-8"))
            result = trace_frozen(read(args.inputs), read(args.candidates), read(args.requests))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(canonical(result), encoding="utf-8")
            result = {"output": str(args.output), "questions": len(result["rows"]), "sha256": result["sha256"]}
        elif args.command == "score-semantic":
            import asyncio
            import os
            from .semantic_judge import JudgeLedger, judge_config, score_inputs
            if args.max_calls < 1 or args.max_token_bound < 1:
                raise ValueError("Judge limits must be positive")
            if not 1 <= args.request_timeout <= 300:
                raise ValueError("Judge request timeout must be between 1 and 300 seconds")
            os.environ["RAGAS_DO_NOT_TRACK"] = "true"
            client = None
            if not args.offline:
                from openai import AsyncOpenAI
                base, token = os.environ.get("MODEL_PROXY_URL"), os.environ.get("MODEL_PROXY_API_KEY")
                if not base or not token:
                    raise ValueError("Live semantic scoring requires caller-supplied Model Proxy credentials")
                from urllib.parse import urlparse
                url = urlparse(base)
                if url.scheme != "https" or not url.hostname or not url.hostname.endswith(".kaggle.net"):
                    raise ValueError("Expected an HTTPS Kaggle Model Proxy endpoint")
                client = AsyncOpenAI(base_url=base.rstrip("/") + ("" if base.rstrip("/").endswith("/openapi") else "/openapi"),
                    api_key=token, max_retries=0, timeout=args.request_timeout)
            config = judge_config(args.judge_model, args.max_output_tokens, args.structured_output)
            if args.request_timeout != 60:
                config["request_timeout_seconds"] = args.request_timeout
            ledger = JudgeLedger(args.ledger, config, args.max_calls,
                args.max_token_bound, offline=args.offline)
            if args.question_aware:
                from .evaluation_hardening import evaluate_frozen
                scored = asyncio.run(evaluate_frozen(json.loads(args.inputs.read_text(encoding="utf-8")),
                    ledger, client, args.code_revision))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(canonical(scored), encoding="utf-8")
                print(canonical({"output": str(args.output), "aggregate": scored["aggregate"]}), end="")
                return
            scored = asyncio.run(score_inputs(json.loads(args.inputs.read_text(encoding="utf-8")),
                ledger, client, args.output, args.code_revision,
                json.loads(args.vectors.read_text(encoding="utf-8")) if args.vectors else None))
            result = {"questions": scored["question_count"], "aggregate": scored["aggregate"], "usage": scored["usage"]}
        elif args.command == "encode":
            from .embeddings import encode
            result, timing = encode(args.artifacts, args.model_directory, extra_question=args.question, device=args.device)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(canonical(result), encoding="utf-8")
            result = {"sha256": result["sha256"], **timing}
        elif args.command == "compare":
            from .comparison import compare
            manifest, artifacts = load_artifacts(args.artifacts)
            embeddings = json.loads(args.embeddings.read_text(encoding="utf-8")) if args.embeddings else None
            result, timings = compare(artifacts, manifest, args.code_revision, embeddings)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(canonical(result), encoding="utf-8")
            args.output.with_suffix(".timings.json").write_text(canonical(timings), encoding="utf-8")
            result = {"output": str(args.output), "aggregate": {mode: report["aggregate"]
                for mode, report in result["results"].items()}, "embedding_sha256": result["embedding_sha256"]}
        elif args.command in ("prepare-answers", "generate-answers", "evaluate-answers"):
            read = lambda path: json.loads(path.read_text(encoding="utf-8"))
            timings = None
            if args.command == "prepare-answers":
                from .answering import prepare_requests
                result = prepare_requests(read(args.candidates), read(args.scores), args.mode, args.prompt_version)
            elif args.command == "generate-answers":
                from .generation import generate
                result, timings = generate(read(args.requests), args.model_directory, device=args.device)
            else:
                from .answer_evaluation import evaluate_answers
                manifest, contents = load_artifacts(args.artifacts)
                result = evaluate_answers(contents, manifest, read(args.requests), read(args.generated), args.code_revision)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(canonical(result), encoding="utf-8")
            if timings is not None:
                args.output.with_suffix(".timings.json").write_text(canonical(timings), encoding="utf-8")
            result = {"output": str(args.output), "sha256": result["sha256"]}
        elif args.command in ("freeze-candidates", "score-candidates", "evaluate-reranker"):
            from .reranking import freeze_candidates, assess
            timings = None
            if args.command == "score-candidates":
                from .cross_encoder import score_candidates
                result, timings = score_candidates(json.loads(args.candidates.read_text(encoding="utf-8")),
                                                  args.model_directory, device=args.device)
            else:
                manifest, artifacts = load_artifacts(args.artifacts)
                embeddings = json.loads(args.embeddings.read_text(encoding="utf-8"))
                if args.command == "freeze-candidates":
                    result, timings = freeze_candidates(artifacts, manifest, embeddings, args.candidate_count, tuple(args.modes))
                else:
                    result = assess(artifacts, manifest, embeddings,
                        json.loads(args.candidates.read_text(encoding="utf-8")),
                        json.loads(args.scores.read_text(encoding="utf-8")), args.code_revision)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(canonical(result), encoding="utf-8")
            if timings is not None:
                args.output.with_suffix(".timings.json").write_text(canonical(timings), encoding="utf-8")
            result = {"output": str(args.output), "sha256": result.get("sha256")}
        else:
            manifest, artifacts = load_artifacts(args.artifacts)
            if args.backend == "legacy" and (args.mode != "bm25" or args.embeddings):
                raise ValueError("Legacy backend supports only BM25 without embeddings")
            embeddings = json.loads(args.embeddings.read_text(encoding="utf-8")) if args.embeddings else None
            retriever = (Retriever(artifacts["passages.json"], manifest["paper_ids"], manifest["corpus_revision"])
                         if args.backend == "legacy" else QdrantRetriever(artifacts["passages.json"], manifest["paper_ids"],
                             manifest["corpus_revision"], embeddings, args.mode))
            if args.command == "search":
                result = [h.model_dump() for h in retriever.search(args.paper, args.question, args.k)]
            elif args.command == "inspect":
                result = retriever.inspect(args.passage).model_dump()
            else:
                result = evaluate(retriever, artifacts["questions.json"], manifest, args.code_revision)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes(canonical(result).encode("utf-8"))
            if hasattr(retriever, "close"):
                retriever.close()
        print(canonical(result), end="")
    except (ValueError, KeyError, OSError, TypeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
