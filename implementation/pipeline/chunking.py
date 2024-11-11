from agent import StatelessAgent, OpenAIAgent
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from pipeline_types import (
    FinetuneEntry,
    EnrichedPdfChunk,
    EnrichedPdfFile,
    EnrichedPdfChunkWithEntities,
    EnrichedPdfChunkWithQuestion,
)
from datetime import datetime
from uuid import uuid4
from typing import AsyncIterator, List
from helpers import split_text
import os 

# Chunk and Embed text into ChromaDB VectorDB 

class EmbedChunksAgent(StatelessAgent[EnrichedPdfFile, List[EnrichedPdfChunk]]):
    def __init__(
        self,
        embeddings_func: OpenAIEmbeddings,
        vector_store: Chroma,
        chunk_size: int,
        chunk_overlap: int,
    ):
        super().__init__("Embed Chunks Agent")
        self.embeddings_func = embeddings_func
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def process_element(
        self, input: EnrichedPdfFile
    ) -> AsyncIterator[List[EnrichedPdfChunk]]:
        chunks = split_text(input["text"], self.chunk_size, self.chunk_overlap)
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunking text into {len(chunks)} chunks"
        )
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "filename": os.path.splitext(os.path.basename(input["filename"]))[0]
                },
            )
            for chunk in chunks
        ]
        uuids = [str(uuid4()) for _ in chunks]
        await self.vector_store.aadd_documents(docs, ids=uuids)

        for chunk in chunks:
            yield EnrichedPdfChunk(
                filename=input["filename"],
                source=input["source"],
                source_type=input["source_type"],
                chunk=chunk,
            )

class ChunkTextAgent(StatelessAgent[EnrichedPdfFile, EnrichedPdfChunk]):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(f"Chunk Text Agent ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def process_element(
        self, input: EnrichedPdfFile
    ) -> AsyncIterator[EnrichedPdfChunk]:
        chunks = split_text(input["text"], self.chunk_size, self.chunk_overlap)
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Chunking text into {len(chunks)} chunks"
        )
        for chunk in chunks:
            yield EnrichedPdfChunk(
                filename=input["filename"],
                source=input["source"],
                source_type=input["source_type"],
                chunk=chunk,
            )
