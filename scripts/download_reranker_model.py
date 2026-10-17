"""Acquire only the pinned public cross-encoder files, anonymously."""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download
from paper_evidence_rag.ingestion import canonical
from paper_evidence_rag.reranking import DEFAULT_MODEL_ID, MODEL_PINS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", choices=sorted(MODEL_PINS), default=DEFAULT_MODEL_ID)
args = parser.parse_args()
snapshot_download(args.model, revision=MODEL_PINS[args.model], local_dir=args.output, token=False,
    allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "vocab.txt", "sentencepiece.bpe.model"])
hashes = {p.relative_to(args.output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(args.output.rglob("*")) if p.is_file() and ".cache" not in p.parts
          and p.name != "snapshot-manifest.json"}
(args.output / "snapshot-manifest.json").write_text(canonical({"id": args.model,
    "revision": MODEL_PINS[args.model], "files": hashes}), encoding="utf-8")
print("Pinned public reranker acquired and hashed without credentials")
