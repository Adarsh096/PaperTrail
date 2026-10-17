"""Bounded text-PDF extraction in a disposable process, without model imports."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from .schema import Passage

MAX_BYTES = 5 * 1024 * 1024
MAX_PAGES = 20
MAX_CHARS = 160_000
MAX_PASSAGES = 300
CHUNK_CHARS = 1000
CHUNK_WORDS = 180
OVERLAP_WORDS = 35


@dataclass
class UploadedPaper:
    paper_id: str
    title: str
    abstract: str
    revision: str
    passages: list[dict]
    pages: int
    locations: dict = field(default_factory=dict)
    dense_cache: dict | None = None
    public_source: bool = False
    dense_passages: list[dict] | None = None
    dense_locations: dict = field(default_factory=dict)


def chunk_page(text):
    """Prefer paragraph/sentence boundaries; preserve exact source slices.

    Character and Unicode-word bounds are model-free estimates, not token counts.
    Dense indexing further splits chunks using the actual encoder tokenizer.
    Each page is independent so citations never obscure a page transition.
    """
    start = 0
    while start < len(text):
        if not text[start:].strip():
            break
        words = list(re.finditer(r"\S+", text[start:start + CHUNK_CHARS]))
        end = min(len(text), start + CHUNK_CHARS)
        if len(words) > CHUNK_WORDS:
            end = start + words[CHUNK_WORDS - 1].end()
        if end < len(text):
            lower = start + (end - start) // 2
            paragraphs = [m.end() for m in re.finditer(r"\n\s*\n", text[start:end]) if start + m.end() >= lower]
            sentences = [m.end() for m in re.finditer(r"[.!?](?:[\"')\]]*)\s+", text[start:end]) if start + m.end() >= lower]
            boundaries = paragraphs or sentences
            if boundaries:
                end = start + boundaries[-1]
            elif text[end - 1:end].strip() and text[end:end + 1].strip():
                space = text.rfind(" ", lower, end)
                if space > lower:
                    end = space + 1
        yield start, end, text[start:end]
        if end == len(text):
            break
        tail = list(re.finditer(r"\S+", text[start:end]))
        overlap = [start + word.start() for word in tail[-OVERLAP_WORDS:]
                   if start + word.start() >= end - 240]
        start = max(start + 1, overlap[0] if overlap else end)


def extract_pdf(data: bytes, timeout=15):
    """Accept bytes, never a client-supplied path to a server file.

    Linux workers also impose CPU and address-space limits. The timeout works
    on Windows too. Temporary copies disappear on success, failure and timeout.
    """
    if not isinstance(data, bytes) or not data or len(data) > MAX_BYTES:
        raise ValueError("Upload one PDF of at most 5 MB.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("This file is not a supported PDF.")
    with tempfile.TemporaryDirectory(prefix="paper-upload-") as directory:
        root = Path(directory)
        (root / "input.pdf").write_bytes(data)
        try:
            completed = subprocess.run([sys.executable, "-m", __name__, str(root)],
                capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            raise ValueError("PDF extraction timed out. Try a smaller text PDF.") from None
        if completed.returncode != 0 or not (root / "result.json").is_file():
            raise ValueError("PDF could not be read within the extraction limits.")
        result = json.loads((root / "result.json").read_text(encoding="utf-8"))
        if "error" in result:
            raise ValueError(result["error"])
        return UploadedPaper(**result)


def _worker(root):
    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    from pypdf import PdfReader

    data = (root / "input.pdf").read_bytes()
    file_hash = hashlib.sha256(data).hexdigest()
    revision = hashlib.sha256((file_hash + f":page-boundaries-v1:{CHUNK_CHARS}:{CHUNK_WORDS}:{OVERLAP_WORDS}:240").encode()).hexdigest()
    reader = PdfReader(root / "input.pdf", strict=True)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are unsupported. Upload an unlocked text PDF.")
    if not 1 <= len(reader.pages) <= MAX_PAGES:
        raise ValueError("Upload a PDF with between 1 and 20 pages.")
    paper_id, passages, chars, locations = "upload-" + revision[:20], [], 0, {}
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        chars += len(text)
        if chars > MAX_CHARS:
            raise ValueError("PDF contains too much text. Use a shorter paper.")
        for block_index, (start, end, block) in enumerate(chunk_page(text)):
            if not block.strip():
                continue
            if len(block) > 20_000 or len(passages) >= MAX_PASSAGES:
                raise ValueError("PDF exceeds the text-block limit. Use a shorter paper.")
            passages.append(Passage(passage_id=f"{paper_id}:paragraph:{page_index}:{block_index}",
                paper_id=paper_id, kind="paragraph", section_index=page_index,
                position=block_index, section_name=f"Page {page_index + 1}", text=block,
                source_file="uploaded.pdf").model_dump())
            locations[passages[-1]["passage_id"]] = {"page": page_index + 1, "char_start": start,
                "char_end": end, "offsets": "zero-based, end-exclusive in extracted page text"}
    if not passages:
        raise ValueError("No readable text found. Scanned PDFs need OCR and are unsupported.")
    # Metadata is untrusted text too; do not interpret it as HTML or a filename.
    title = str((reader.metadata or {}).get("/Title", "") or "Uploaded paper")[:200]
    return {"paper_id": paper_id, "title": title, "abstract": passages[0]["text"][:3000],
        "revision": revision, "passages": passages, "pages": len(reader.pages), "locations": locations}


if __name__ == "__main__":
    root = Path(sys.argv[1])
    try:
        result = _worker(root)
    except ValueError as error:
        result = {"error": str(error)}
    except Exception:
        result = {"error": "Malformed or unsupported PDF. Try a different text PDF."}
    (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
