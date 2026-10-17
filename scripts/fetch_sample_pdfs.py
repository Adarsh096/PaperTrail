"""Download two attributed CC BY 4.0 sample papers; never run during a UI request."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

SAMPLES = [
    {"name": "qasper", "title": "A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers",
     "authors": "Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, Matt Gardner", "year": 2021,
     "source": "https://aclanthology.org/2021.naacl-main.365/", "url": "https://aclanthology.org/2021.naacl-main.365.pdf",
     "sha256": "b15b4378988ed4e553dd312eaf334411abcf1fb28c53b846d360435eb55cd193", "license": "CC-BY-4.0"},
    {"name": "sentence-bert", "title": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
     "authors": "Nils Reimers, Iryna Gurevych", "year": 2019,
     "source": "https://aclanthology.org/D19-1410/", "url": "https://aclanthology.org/D19-1410.pdf",
     "sha256": "e2703128039850ba7e098bfc6c7d9f3f3e75e6318815c5dd90af71085b1d384e", "license": "CC-BY-4.0"},
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/sample-pdfs"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        target = args.output / (sample["name"] + ".pdf")
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == sample["sha256"]:
            continue
        with urllib.request.urlopen(sample["url"], timeout=30) as response:
            data = response.read(5 * 1024 * 1024 + 1)
        if not data.startswith(b"%PDF-") or len(data) > 5 * 1024 * 1024:
            raise ValueError("Sample download is not a bounded PDF")
        if hashlib.sha256(data).hexdigest() != sample["sha256"]:
            raise ValueError("Sample changed. Review source and hash before using it.")
        target.write_bytes(data)
    (args.output / "sources.json").write_text(json.dumps(SAMPLES, indent=2) + "\n", encoding="utf-8")
    print("Two verified CC BY 4.0 sample PDFs ready.")


if __name__ == "__main__":
    main()
