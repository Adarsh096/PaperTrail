import hashlib
import json
from pathlib import Path

from .schema import Annotation, PaperDocument, Passage, Question

SOURCE_REVISION = "fdc9d8214fbab5dd782958601db4d678e6934a54"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def normalize(raw: dict, paper_limit=5):
    if type(paper_limit) is not int or not 1 <= paper_limit <= 20:
        raise ValueError("Paper limit must be between 1 and 20")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Expected a nonempty QASPER training object")
    papers, passages, questions = [], [], []
    selected = sorted(raw)[:paper_limit]
    for paper_id in selected:
        source = raw[paper_id]
        document = PaperDocument.model_validate({
            "paper_id": paper_id,
            **{key: source[key] for key in ("title", "abstract", "full_text", "figures_and_tables")},
        })
        papers.append(document.model_dump())

        def add(kind, position, text, section_name, section_index=None, source_file=None):
            if not text.strip():
                return
            suffix = f"{section_index}:{position}" if kind == "paragraph" else str(position)
            passages.append(Passage(
                passage_id=f"{paper_id}:{kind}:{suffix}", paper_id=paper_id,
                kind=kind, position=position, text=text, section_name=section_name,
                section_index=section_index, source_file=source_file,
            ).model_dump())

        add("abstract", 0, document.abstract, "Abstract")
        for section_index, section in enumerate(document.full_text):
            for position, paragraph in enumerate(section.paragraphs):
                add("paragraph", position, paragraph, section.section_name, section_index)
        for position, figure in enumerate(document.figures_and_tables):
            add("caption", position, figure.caption, "Figures and tables", source_file=figure.file)
        seen = set()
        for qa in source["qas"]:
            if qa["question_id"] in seen:
                raise ValueError("Duplicate question ID within paper")
            seen.add(qa["question_id"])
            annotations = []
            for answer in qa["answers"]:
                a = answer["answer"]
                kind = ("unanswerable" if a["unanswerable"] else "yes_no" if a["yes_no"] is not None
                        else "extractive" if a["extractive_spans"] else "free_form")
                annotations.append(Annotation(
                    annotation_id=answer["annotation_id"], answer_type=kind,
                    **{key: a[key] for key in ("unanswerable", "evidence", "extractive_spans", "free_form_answer", "yes_no")},
                ))
            questions.append(Question(paper_id=paper_id, question_id=qa["question_id"],
                                      question=qa["question"], annotations=annotations).model_dump())
    questions.sort(key=lambda q: (q["paper_id"], q["question_id"]))
    return papers, passages, questions


def build(source: Path, output: Path, expected_sha256: str, paper_limit=5):
    data = source.read_bytes()
    if digest(data) != expected_sha256:
        raise ValueError("Training input SHA-256 mismatch")
    papers, passages, questions = normalize(json.loads(data), paper_limit)
    artifacts = {"papers.json": papers, "passages.json": passages, "questions.json": questions}
    hashes = {name: digest(canonical(value).encode()) for name, value in artifacts.items()}
    manifest = {
        "schema_version": 1, "dataset": "allenai/qasper", "source_revision": SOURCE_REVISION,
        "dataset_version": "0.3", "split": "train", "input_sha256": expected_sha256,
        "selection": ("first five training paper IDs in ascending lexical order" if paper_limit == 5
                      else f"first {paper_limit} training paper IDs in ascending lexical order"),
        "paper_ids": [p["paper_id"] for p in papers], "artifact_sha256": hashes,
        "paper_count": len(papers), "passage_count": len(passages), "question_count": len(questions),
        "retrieval": {"algorithm": "BM25Okapi", "k1": 1.5, "b": 0.75,
                      "tokenizer": "lowercase Unicode \\w+", "scope": "selected paper"},
    }
    manifest["corpus_revision"] = digest(canonical(manifest).encode())
    output.mkdir(parents=True, exist_ok=True)
    for name, value in {**artifacts, "manifest.json": manifest}.items():
        (output / name).write_bytes(canonical(value).encode("utf-8"))
    return manifest


def load_artifacts(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in manifest.items() if k != "corpus_revision"}
    if digest(canonical(unsigned).encode()) != manifest["corpus_revision"]:
        raise ValueError("Manifest corpus revision mismatch")
    if set(manifest["artifact_sha256"]) != {"papers.json", "passages.json", "questions.json"}:
        raise ValueError("Unexpected artifact filenames")
    contents = {}
    for name, expected in manifest["artifact_sha256"].items():
        data = (root / name).read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Artifact SHA-256 mismatch: {name}")
        contents[name] = json.loads(data)
    return manifest, contents
