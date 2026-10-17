"""Run the fixture suite and real-data reproducibility checks in a bounded batch.

Dependencies and the package must already be installed. This script uses no network.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from paper_evidence_rag.evaluation import evaluate
from paper_evidence_rag.ingestion import build, canonical, digest, load_artifacts
from paper_evidence_rag.retrieval import Retriever

TRAIN_SHA256 = "9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-revision", required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([sys.executable, "-m", "pytest", str(args.tests), "-q", "-p", "no:cacheprovider",
                             "--basetemp", str(args.output / "fixture-tmp")],
                            capture_output=True, text=True, timeout=180)
    (args.output / "tests.log").write_bytes((result.stdout + result.stderr).encode())
    if result.returncode:
        raise RuntimeError("Fixture suite failed; see tests.log")
    first, second = args.output / "build", args.output / "rebuild"
    manifest = build(args.source, first, TRAIN_SHA256)
    build(args.source, second, TRAIN_SHA256)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {p.name: p.read_bytes() for p in second.iterdir()}
    assert manifest["paper_count"] == 5
    _, artifacts = load_artifacts(first)
    r = Retriever(artifacts["passages.json"], manifest["paper_ids"], manifest["corpus_revision"])
    report = evaluate(r, artifacts["questions.json"], manifest, args.code_revision)
    for row in report["questions"]:
        for hit in row["retrieved"]:
            assert r.inspect(hit["passage"]["passage_id"]).model_dump() == hit["passage"]
    deterministic = {k: v for k, v in report.items() if k != "elapsed_seconds"}
    (args.output / "evaluation.json").write_bytes(canonical(report).encode())
    summary = {"status": "passed", "manifest": manifest, "evaluation_sha256": digest(canonical(deterministic).encode()),
               "code_revision": args.code_revision, "python": platform.python_version(),
               "platform": platform.system(), "test_output": result.stdout.strip(),
               "elapsed_seconds": time.perf_counter() - start}
    (args.output / "result.json").write_bytes(canonical(summary).encode())
    print(json.dumps({"status": "passed", "papers": 5, "passages": manifest["passage_count"],
                      "questions": report["question_count"]}), flush=True)


if __name__ == "__main__":
    main()
