from asyncio import Future
from pydantic import BaseModel, Field
from typing import AsyncGenerator, AsyncIterator, List, Generator, Tuple
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging

from helpers import (
    get_json_response_async,
    get_messages_response_async,
    get_model,
    get_messages_response_async,
)
from agent import OpenAIAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTION_EXTRACTION_SYSTEM_PROMPT = """
Given a piece of text, generate enough diverse questions to thoroughly explore the key entities mentioned.
For each key entity, generate a set of 'what, why, summarize, where' questions, ensuring the questions cover different aspects of the entity.
The questions should follow these instructions:

# Instructions

1. Generate a mix of 'what, why, summarize, where' questions.
2. Ensure the questions are diverse and cover the entities identified in the text.
3. Include the source information in the questions.
4. Keep the number of questions open-ended to ensure full coverage of the topic.
5. Ensure clarity and relevance in each question.
6. Extract the questions as a list in JSON format

Below is an example

# Example

Entities: Visual Prompt Tuning, Vision Transformers, Prompt Tuning
Text: "Learning generative image models from various domains efficiently needs transferring knowledge from an image synthesis model trained on a large dataset. We present a recipe for learning vision transformers by generative knowledge transfer. We base our framework on generative vision transformers representing an image as a sequence of visual tokens with the autoregressive or non-autoregressive transformers. To adapt to a new domain, we employ prompt tuning, which prepends learnable tokens called prompts to the image token sequence and introduces a new prompt design for our task. We study on a variety of visual domains with varying amounts of training images. We show the effectiveness of knowledge transfer and a significantly better image generation quality."
Source: "Visual prompt tuning"
SourceType: "Paper"

# Output (JSON):
```json
{{
    "questions": [
        "What is the purpose of using prompt tuning in the framework described in the paper 'Visual prompt tuning'?",
        "Why does learning generative image models from various domains require transferring knowledge from a model trained on a large dataset, according to the paper 'Visual prompt tuning'?",
        "Summarize the method of generative knowledge transfer for vision transformers as presented in the paper 'Visual prompt tuning.'",
        "Where are the learnable tokens, or prompts, added in the process of adapting vision transformers to a new domain, as explained in the paper 'Visual prompt tuning'?",
        "What are the two types of transformers mentioned in the framework for representing an image as a sequence of visual tokens in the paper 'Visual prompt tuning'?",
    ]
}}

# Entities:
{entities}

# Text:
{text}

# Source: {source}
# SourceType: {source_type}
"""


class QuestionAnswerModel(BaseModel):
    questions: List[str]


ANSWER_EXTRACTION_SYSTEM_PROMPT = """
You are tasked with answering questions based on a provided text.
Your goal is to generate high-quality, detailed answers by following these instructions:

# Instructions:
1. Reference the Text: Answer directly using relevant details from the text. Avoid introducing unsupported claims.
2. Comprehensive Response: Address all parts of the question thoroughly, covering multiple aspects if needed.
3. Detail-Oriented: Highlight key elements like techniques, processes, models, or challenges, expanding on them for clarity.
4. Organized Structure: Use clear paragraphs or points for complex answers.
5. Clarity and Examples: Ensure the answer is precise and easy to follow. Include examples or quotes from the text when applicable.
6. Include Sources: Clearly reference the source information at the end of the answer.

# Text:
{text}
"""

from typing import TypedDict


class GenerateQuestionsArgs(TypedDict):
    chunk: str
    source: str
    source_type: str
    entities: List[str]


class QuestionGenerator(OpenAIAgent[GenerateQuestionsArgs, str]):
    def __init__(self, model: str, batch_size: int = 10):
        super().__init__(model)
        self.batch_size = batch_size

    async def _process(
        self, inputs: AsyncIterator[GenerateQuestionsArgs]
    ) -> AsyncIterator[str]:
        async for input in inputs:
            for i in range(0, len(input["entities"]), self.batch_size):
                entities = input["entities"][i : i + self.batch_size]
                for question in await self._generate_questions(
                    input["chunk"], input["source"], input["source_type"], entities
                ):
                    yield question

    async def _generate_questions(
        self, chunk: str, source: str, source_type: str, entities: List[str]
    ) -> List[str]:
        qa_prompt = QUESTION_EXTRACTION_SYSTEM_PROMPT.format(
            text=chunk,
            entities=", ".join(entities),
            source=source,
            source_type=source_type,
        )
        print(f"Length of qa_prompt: {len(qa_prompt)}")
        return (
            await get_json_response_async(
                client=self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": qa_prompt},
                ],
                response_format=QuestionAnswerModel,
            )
        ).questions


DEFAULT_QUESTION_ANSWER_MODEL = "meta-llama-3.1-8b-instruct-q6_k"


class QAPair(TypedDict):
    question: str
    answer: str
    chunk: str
    chunk_index: int


class GetAnswerAgent(OpenAIAgent[Tuple[int, str], QAPair]):
    def __init__(self, chunk_index: int, chunk: str ):
        super().__init__(DEFAULT_QUESTION_ANSWER_MODEL)
        self.text_chunk = chunk
        self.chunk_index = chunk_index

    async def _process(
        self, inputs: AsyncIterator[str]
    ) -> AsyncIterator[QAPair]:
        async for question in inputs:
            print(f"Generating answer for Question: {question}")
            answer = await get_messages_response_async(
                client=self.client,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": ANSWER_EXTRACTION_SYSTEM_PROMPT.format(
                            text=self.text_chunk
                        ),
                    },
                    {"role": "user", "content": question},
                ],
            )
            yield QAPair(question=question, answer=answer, chunk=self.text_chunk, chunk_index=self.chunk_index)
