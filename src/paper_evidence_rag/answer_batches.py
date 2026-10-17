"""Partition offline generation without changing evidence or prompt lineage."""
from .answering import sealed
from .reranking import checked_hash


def split_requests(requests, size=6):
    checked_hash(requests)
    if type(size) is not int or not 1 <= size <= 6:
        raise ValueError("Generation batch size must be between 1 and 6")
    if not 1 <= len(requests["rows"]) <= 20:
        raise ValueError("Expected between 1 and 20 questions")
    keys = [(r["paper_id"], r["question_id"]) for r in requests["rows"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate request question")
    base = {k: v for k, v in requests.items() if k not in {"sha256", "rows"}}
    return [sealed({**base, "rows": requests["rows"][i:i + size]})
            for i in range(0, len(keys), size)]


def merge_generated(requests, batches, outputs):
    """Require exact partition coverage and identical model/runtime metadata.

    Each result keeps its original row. Full source/citation validation still
    runs through evaluate_answers after merging; no text repair occurs here.
    """
    checked_hash(requests)
    if not batches or len(batches) != len(outputs):
        raise ValueError("Missing generation batches")
    expected = requests["rows"]
    flattened, rows, template = [], [], None
    base = {k: v for k, v in requests.items() if k not in {"sha256", "rows"}}
    for batch, output in zip(batches, outputs):
        checked_hash(batch)
        checked_hash(output)
        if {k: v for k, v in batch.items() if k not in {"sha256", "rows"}} != base:
            raise ValueError("Batch request lineage mismatch")
        if output["request_sha256"] != batch["sha256"] or output["prompt_version"] != requests["prompt_version"]:
            raise ValueError("Batch output request mismatch")
        question = lambda r: (r["paper_id"], r["question_id"], r["question"])
        if list(map(question, output["rows"])) != list(map(question, batch["rows"])):
            raise ValueError("Batch output question mismatch")
        meta = {k: v for k, v in output.items() if k not in {"sha256", "rows", "request_sha256"}}
        if template is not None and template != meta:
            raise ValueError("Batch model or runtime mismatch")
        template = meta
        flattened.extend(batch["rows"])
        rows.extend(output["rows"])
    if flattened != expected:
        raise ValueError("Batches must cover the full request exactly in order")
    return sealed({**template, "request_sha256": requests["sha256"], "rows": rows})
