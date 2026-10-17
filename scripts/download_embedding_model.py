"""Download the pinned public model anonymously, separately from inference."""
import argparse
import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download
from paper_evidence_rag.embeddings import MODEL_ID, MODEL_REVISION
from paper_evidence_rag.ingestion import canonical

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("models/embeddinggemma-2"))
args = parser.parse_args()
snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=args.output, token=False)
hashes = {}
for path in sorted(args.output.rglob("*")):
    if path.is_file() and ".cache" not in path.parts and path.name != "snapshot-manifest.json":
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        hashes[path.relative_to(args.output).as_posix()] = h.hexdigest()
(args.output / "snapshot-manifest.json").write_text(canonical({"id": MODEL_ID,
    "revision": MODEL_REVISION, "files": hashes}), encoding="utf-8")
print("Pinned public model downloaded and hashed; no token used")
