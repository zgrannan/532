from agent import Agent, MapAgent, StatelessAgent, OpenAIAgent
from agent import ModelProvider
from agent import LLMClientSettings
from helpers import get_response_format
from pipeline_types import (
    HasQuestion,
    FinetuneEntry,
    EnrichedPdfChunkWithQuestion,
)
from typing import AsyncIterator, TypeVar
from langchain_openai import OpenAIEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from helpers import slurp_iterator, get_json_response_async
from datetime import datetime
from pydantic import BaseModel
import logging

T = TypeVar("T")

class RemoveDuplicatesAgent(Agent[T, T]):
    def __init__(self):
        super().__init__("Remove Duplicates Agent")

    async def _process(self, input_stream: AsyncIterator[T]) -> AsyncIterator[T]:
        seen = set()
        async for input in input_stream:
            if input not in seen:
                seen.add(input)
                yield input

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


HasQuestionT = TypeVar("HasQuestionT", bound=HasQuestion)


class RemoveSimilarQuestionsAgent(Agent[HasQuestionT, HasQuestionT]):
    def __init__(self, embeddings_func: OpenAIEmbeddings, threshold: float):
        super().__init__("Remove Similar Questions Agent")
        self.embed_agent = EmbeddingsAgent(embeddings_func)
        self.threshold = threshold

    async def _process(
        self, inputs: AsyncIterator[HasQuestionT]
    ) -> AsyncIterator[HasQuestionT]:
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
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Removed {len(indices_to_remove)} similar questions from {len(input_list)} questions"
        )

        for item in filtered_data:
            yield item


SOURCE_IN_QUESTION_PROMPT = """
Analyze questions and ensure proper source attribution by following these rules:

# Inputs

Question
Source
Source Type (paper, book, article, webpage)

# Instructions

If source exists in question: Return original
If source missing: Add source using appropriate format:

Paper/Article: "...in 'Title'..."
Book: "...in Book Title..."
Website: "...on Website Name..."

# Source Placement

Insert where most natural
Avoid disrupting technical terms
Maintain grammar and readability
Use abbreviated forms for repeated mentions

Return the original or modified question.
Remember to integrate the source into the question in a natural way and to return a question.

Question:
{question}

Source:
{source}

Source Type:
{source_type}
"""


class SourceInQuestion(BaseModel):
    question: str


AddSourceToQuestionT = TypeVar(
    "AddSourceToQuestionT", EnrichedPdfChunkWithQuestion, FinetuneEntry
)


class AddSourceToQuestionAgent(
    OpenAIAgent, StatelessAgent[AddSourceToQuestionT, AddSourceToQuestionT]
):
    def __init__(
        self, settings: LLMClientSettings
    ):
        super().__init__(settings)
        StatelessAgent.__init__(self, name="Add Source to Question Agent")

    async def process_element(
        self, input: AddSourceToQuestionT
    ) -> AsyncIterator[AddSourceToQuestionT]:
        response_format = get_response_format(self.model_provider, SourceInQuestion)
        resp = await get_json_response_async(
            client=self.client,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SOURCE_IN_QUESTION_PROMPT.format(
                        question=input["question"],
                        source=input["source"],
                        source_type=input["source_type"],
                    ),
                },
            ],
            response_format=response_format,
            agent_name=self.name,
        )
        logging.debug(f"Original Question: {input['question']}")
        logging.debug(f"Modified Question: {resp.question}")
        result: AddSourceToQuestionT = input.copy()
        result["question"] = resp.question
        yield result
