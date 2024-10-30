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


class FinetuneEntry(TypedDict):
    filename: str
    source: str
    source_type: str
    chunk: str
    question: str
    answer: str
