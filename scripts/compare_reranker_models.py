"""Compare completed reranker reports and costs from one paired experiment."""
import argparse
import json
from pathlib import Path

from paper_evidence_rag.ingestion import canonical
from paper_evidence_rag.model_comparison import compare_models

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run", type=Path, action="append", required=True,
    help="Directory containing reranking.json and scores.timings.json; repeat for each model")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
reports, timings = {}, {}
for directory in args.run:
    report = json.loads((directory / "reranking.json").read_text(encoding="utf-8"))
    name = report["model"]["id"]
    if name in reports:
        raise ValueError("Duplicate model run")
    reports[name] = report
    timings[name] = json.loads((directory / "scores.timings.json").read_text(encoding="utf-8"))
result = compare_models(reports, timings)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(canonical(result), encoding="utf-8")
print(canonical(result))
