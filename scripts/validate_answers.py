"""Replay citations and metrics; run the identical lightweight fixture suite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from paper_evidence_rag.answer_evaluation import evaluate_answers
from paper_evidence_rag.ingestion import canonical, digest, load_artifacts

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--requests", type=Path, required=True)
parser.add_argument("--generated", type=Path, required=True)
parser.add_argument("--tests", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--code-revision", required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
manifest, contents = load_artifacts(args.artifacts)
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
requests, generated = read(args.requests), read(args.generated)
report = evaluate_answers(contents, manifest, requests, generated, args.code_revision)
test = subprocess.run([sys.executable, "-m", "pytest", str(args.tests), "-q"],
    capture_output=True, timeout=120)
log = test.stdout + test.stderr
(args.output / "tests.log").write_bytes(log)
if test.returncode:
    raise SystemExit("Fixture suite failed; inspect tests.log")
(args.output / "evaluation.json").write_text(canonical(report), encoding="utf-8")
result = {"status": "passed", "code_revision": args.code_revision,
    "paper_count": manifest["paper_count"], "passage_count": manifest["passage_count"],
    "question_count": report["question_count"], "request_sha256": requests["sha256"],
    "generation_sha256": generated["sha256"], "report_sha256": report["sha256"],
    "tests_sha256": digest(log)}
(args.output / "result.json").write_text(canonical(result), encoding="utf-8")
print(canonical(result), end="")
