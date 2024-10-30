from typing import TypedDict


class EnrichedPdfFile(TypedDict):
    filename: str
    source: str
    source_type: str
    text: str


class EnrichedPdfChunk(TypedDict):
    filename: str
    source: str
    source_type: str
    chunk: str


class EnrichedPdfChunkWithEntities(TypedDict):
    filename: str
    source: str
    source_type: str
    chunk: str
    entities: list[str]


class EnrichedPdfChunkWithQuestion(TypedDict):
    filename: str
    source: str
    source_type: str
    chunk: str
    question: str


class EnrichedPdfChunkWithQAPair(TypedDict):
    filename: str
    source: str
    source_type: str
    chunk: str
    question: str
    answer: str


class EnrichedPdfChunkWithRefinedQAPair(TypedDict):
    filename: str
    source: str
    source_type: str
    chunk: str
    question: str
    answer: str
    original_question: str

class FinetuneEntry(TypedDict):
    question: str
    answer: str
    source: str
    chunk: str
    chunk_index: int
    chunk_size: int
    chunk_overlap: int
