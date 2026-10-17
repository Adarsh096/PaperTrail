"""Save bounded, inspectable acceptance artifacts without rerunning inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from paper_evidence_rag.comparison import compare
from paper_evidence_rag.ingestion import canonical, digest, load_artifacts

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--artifacts", type=Path, required=True)
parser.add_argument("--embeddings", type=Path, required=True)
parser.add_argument("--tests", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--code-revision", required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
test = subprocess.run([sys.executable, "-m", "pytest", "-q", str(args.tests),
    "--basetemp", str(args.output / "pytest-temp")], capture_output=True, text=True, timeout=120)
(args.output / "tests.log").write_text(test.stdout + test.stderr, encoding="utf-8")
if test.returncode:
    raise RuntimeError("Fixture suite failed; inspect tests.log")
manifest, contents = load_artifacts(args.artifacts)
assert manifest["paper_count"] <= 20
embeddings = json.loads(args.embeddings.read_text(encoding="utf-8"))
report, timings = compare(contents, manifest, args.code_revision, embeddings)
second, _ = compare(contents, manifest, args.code_revision, embeddings)
assert canonical(report) == canonical(second), "Repeated rankings or metrics changed"
assert all(r["question_count"] <= 20 for r in report["results"].values())
assert len({canonical({k: r[k] for k in ("eligible_count", "excluded_counts", "evidence_mapping_counts")})
    for r in report["results"].values()}) == 1, "Evaluation population changed between backends"
(args.output / "comparison.json").write_text(canonical(report), encoding="utf-8")
(args.output / "timings.json").write_text(canonical(timings), encoding="utf-8")
result = {"status": "passed", "code_revision": args.code_revision, "corpus_revision": manifest["corpus_revision"],
    "paper_count": manifest["paper_count"], "passage_count": manifest["passage_count"],
    "comparison_sha256": digest(canonical(report).encode()), "embedding_sha256": embeddings["sha256"],
    "fixture_log_sha256": digest((args.output / "tests.log").read_bytes()),
    "aggregate": {mode: r["aggregate"] for mode, r in report["results"].items()}}
(args.output / "result.json").write_text(canonical(result), encoding="utf-8")
print(canonical(result))
