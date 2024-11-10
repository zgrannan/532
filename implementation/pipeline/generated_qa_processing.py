from agent import Agent, MapAgent
from pipeline_types import (
    FinetuneEntry,
    EnrichedPdfChunkWithQuestion,
)
from typing import AsyncIterator
from langchain_openai import OpenAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from helpers import slurp_iterator
from datetime import datetime

class RemoveDuplicateQuestionsAgent(
    Agent[EnrichedPdfChunkWithQuestion, EnrichedPdfChunkWithQuestion]
):
    async def _process(
        self, inputs: AsyncIterator[EnrichedPdfChunkWithQuestion]
    ) -> AsyncIterator[EnrichedPdfChunkWithQuestion]:
        seen = set()
        async for input in inputs:
            if input["question"] not in seen:
                seen.add(input["question"])
                yield input

class EmbeddingsAgent(MapAgent[str, list[float]]):
    def __init__(self, embeddings_func: OpenAIEmbeddings):
        super().__init__("Embeddings Agent")
        self.embeddings_func = embeddings_func

    async def handle(self, input: str) -> list[float]:
        return await self.embeddings_func.aembed_query(input)
    

class RemoveSimilarQuestionsAgent(Agent[FinetuneEntry, FinetuneEntry]):
    def __init__(self, embeddings_func: OpenAIEmbeddings, threshold: float):
        super().__init__("Remove Similar Questions Agent")
        self.embed_agent = EmbeddingsAgent(embeddings_func)
        self.threshold = threshold

    async def _process(
        self, inputs: AsyncIterator[FinetuneEntry]
    ) -> AsyncIterator[FinetuneEntry]:
        input_list = await slurp_iterator(inputs)
        embeddings = [
            await self.embed_agent.handle(input["question"]) for input in input_list
        ]
        similarity_matrix = cosine_similarity(embeddings)

        # Set diagonal to 0 to avoid self-matches
        np.fill_diagonal(similarity_matrix, 0)

        # Find pairs above threshold
        similar_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > self.threshold:
                    similar_pairs.append((i, j))

        indices_to_remove = set()
        # For each similar pair, remove the second question
        for pair in similar_pairs:
            indices_to_remove.add(pair[1])

        filtered_data = [
            item for i, item in enumerate(input_list) if i not in indices_to_remove
        ]

        # Print the number of questions removed
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Removed {len(indices_to_remove)} similar questions from {len(input_list)} questions"
        )

        for item in filtered_data:
            yield item