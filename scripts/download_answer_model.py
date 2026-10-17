"""Acquire the pinned public generator anonymously; inference remains offline."""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download
from paper_evidence_rag.answering import MODEL_PINS, DEFAULT_MODEL_ID
from paper_evidence_rag.ingestion import canonical

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--model", choices=sorted(MODEL_PINS), default=DEFAULT_MODEL_ID)
args = parser.parse_args()
snapshot_download(args.model, revision=MODEL_PINS[args.model], local_dir=args.output, token=False,
    allow_patterns=["config.json", "generation_config.json", "*.safetensors", "model.safetensors.index.json", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"])
def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


hashes = {p.relative_to(args.output).as_posix(): file_hash(p)
          for p in sorted(args.output.rglob("*")) if p.is_file() and ".cache" not in p.parts
          and p.name != "snapshot-manifest.json"}
(args.output / "snapshot-manifest.json").write_text(canonical({"id": args.model,
    "revision": MODEL_PINS[args.model], "files": hashes}), encoding="utf-8")
print("Pinned public generator acquired and hashed without credentials")
