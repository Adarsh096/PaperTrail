"""Split CPU generation requests or join verified output partitions."""
import argparse
import json
from pathlib import Path

from paper_evidence_rag.answer_batches import merge_generated, split_requests
from paper_evidence_rag.ingestion import canonical

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("operation", choices=("split", "merge"))
parser.add_argument("--requests", type=Path, required=True)
parser.add_argument("--directory", type=Path, required=True)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
read = lambda p: json.loads(p.read_text(encoding="utf-8"))
requests = read(args.requests)
batches = split_requests(requests)
if args.operation == "split":
    args.directory.mkdir(parents=True, exist_ok=True)
    for i, batch in enumerate(batches):
        (args.directory / f"requests-{i}.json").write_text(canonical(batch), encoding="utf-8")
else:
    if args.output is None:
        parser.error("merge requires --output")
    outputs = [read(args.directory / f"generated-{i}.json") for i in range(len(batches))]
    result = merge_generated(requests, batches, outputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical(result), encoding="utf-8")
print(f"{args.operation}: {len(batches)} bounded batches")
