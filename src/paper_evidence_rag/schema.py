from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Section(Record):
    section_name: str
    paragraphs: list[str]


class Figure(Record):
    caption: str
    file: str


class PaperDocument(Record):
    paper_id: str
    title: str
    abstract: str
    full_text: list[Section]
    figures_and_tables: list[Figure]


class Passage(Record):
    passage_id: str
    paper_id: str
    kind: Literal["abstract", "paragraph", "caption"]
    section_index: int | None = None
    position: int
    section_name: str
    text: str
    source_file: str | None = None


class Annotation(Record):
    annotation_id: str
    unanswerable: bool
    answer_type: Literal["unanswerable", "yes_no", "extractive", "free_form"]
    evidence: list[str]
    extractive_spans: list[str]
    free_form_answer: str
    yes_no: bool | None


class Question(Record):
    paper_id: str
    question_id: str
    question: str
    annotations: list[Annotation] = Field(min_length=1)


class EvidenceHit(Record):
    rank: int
    score: float
    corpus_revision: str
    passage: Passage
