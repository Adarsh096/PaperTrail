"""Run fixtures and verify repeated paired reranking from cached model scores."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from paper_evidence_rag.ingestion import canonical, digest, load_artifacts
from paper_evidence_rag.reranking import assess
from paper_evidence_rag.schema import Passage

parser = argparse.ArgumentParser(description=__doc__)
for name in ("artifacts", "embeddings", "candidates", "scores", "tests", "output"):
    parser.add_argument("--" + name, type=Path, required=True)
parser.add_argument("--code-revision", required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
test = subprocess.run([sys.executable, "-m", "pytest", "-q", str(args.tests),
    "--basetemp", str(args.output / "pytest-temp")], capture_output=True, text=True, timeout=120)
(args.output / "tests.log").write_text(test.stdout + test.stderr, encoding="utf-8")
if test.returncode:
    raise RuntimeError("Fixture suite failed; inspect tests.log")
manifest, contents = load_artifacts(args.artifacts)
embeddings, candidates, scores = [json.loads(p.read_text(encoding="utf-8"))
                                for p in (args.embeddings, args.candidates, args.scores)]
report = assess(contents, manifest, embeddings, candidates, scores, args.code_revision)
assert report == assess(contents, manifest, embeddings, candidates, scores, args.code_revision)
source = {p["passage_id"]: p for p in contents["passages.json"]}
for mode in report["results"].values():
    for row in mode["reranked"]["questions"]:
        for hit in row["retrieved"]:
            assert hit["passage"] == Passage.model_validate(source[hit["passage"]["passage_id"]]).model_dump()
        if row["metrics"]:
            assert row["metrics"]["recall_at_5"] <= row["candidate_recall"]
            assert row["metrics"]["recall_at_10"] <= row["candidate_recall"]
(args.output / "reranking.json").write_text(canonical(report), encoding="utf-8")
result = {"status": "passed", "code_revision": args.code_revision,
    "paper_count": manifest["paper_count"], "passage_count": manifest["passage_count"],
    "question_count": len(candidates["questions"]), "candidate_sha256": candidates["sha256"],
    "scores_sha256": scores["sha256"], "report_sha256": digest(canonical(report).encode()),
    "tests_sha256": digest((args.output / "tests.log").read_bytes())}
(args.output / "result.json").write_text(canonical(result), encoding="utf-8")
print(canonical(result))
