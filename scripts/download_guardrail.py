"""Explicit online preparation; query screening itself loads local files only."""
import argparse
import hashlib
import json
from pathlib import Path

from paper_evidence_rag.guardrail_models import MODEL_PINS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_PINS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(args.model, revision=MODEL_PINS[args.model], local_dir=args.output,
        allow_patterns=["config.json", "rl_agent_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "model.safetensors", "tokenizer/*", "encoder/config.json", "spm.model", "vocab.txt"],
        ignore_patterns=["multilingual/*", "typed-decisions/*", "onnx/*"], token=False)
    files = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*") if p.is_file() and ".cache" not in p.parts and p.name not in ("guardrail-snapshot.json", "guardrail-file-hashes.json")}
    (args.output/"guardrail-snapshot.json").write_text(json.dumps({"id": args.model, "revision": MODEL_PINS[args.model]}), encoding="utf-8")
    (args.output/"guardrail-file-hashes.json").write_text(json.dumps(files, sort_keys=True), encoding="utf-8")
    print("Prepared pinned guardrail", args.model, len(files), "hashed files")


if __name__ == "__main__":
    main()
