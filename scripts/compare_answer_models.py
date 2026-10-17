import argparse
import json
from pathlib import Path

from paper_evidence_rag.answer_comparison import compare_answers
from paper_evidence_rag.ingestion import canonical

parser = argparse.ArgumentParser(description="Compare two pinned generators on identical evidence")
parser.add_argument("--run", type=Path, action="append", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
runs = [[json.loads((root / name).read_text(encoding="utf-8"))
         for name in ("generated.json", "evaluation.json", "generated.timings.json")] for root in args.run]
result = compare_answers(runs)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(canonical(result), encoding="utf-8")
print("Paired structure and answer comparison saved")
