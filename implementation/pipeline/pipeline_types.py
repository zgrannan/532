from typing import TypedDict


class HasQuestion(TypedDict):
    question: str

class HasSource(TypedDict):
    source: str

class HasSourceType(TypedDict):
    source_type: str


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


class EnrichedPdfChunkWithQuestion(HasQuestion, HasSource, HasSourceType):
    filename: str
    chunk: str

class FinetuneEntry(HasQuestion, HasSource, HasSourceType):
    filename: str
    chunk: str
    answer: str
    pass_through: bool

class FinetuneEntryRefined(TypedDict):
    filename: str
    source: str
    source_type: str
    chunk: str
    question: str
    answer: str
    original_question: str

