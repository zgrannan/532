from asyncio import Future
from pydantic import BaseModel, Field
from typing import AsyncGenerator, List, Generator
from openai import OpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging

from helpers import (
    get_json_response_async,
    get_messages_response_async,
    get_model,
    get_messages_response_async,
)

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
    client: AsyncOpenAI
    model: str
    chunk: str
    source: str
    source_type: str


async def _generate_questions(
    args: GenerateQuestionsArgs, entities: List[str]
) -> List[str]:
    qa_prompt = QUESTION_EXTRACTION_SYSTEM_PROMPT.format(
        text=args["chunk"],
        entities=", ".join(entities),
        source=args["source"],
        source_type=args["source_type"],
    )
    print(f"Length of qa_prompt: {len(qa_prompt)}")
    return (await get_json_response_async(
        client=args["client"],
        model=get_model(args["model"]),
        messages=[
            {"role": "system", "content": qa_prompt},
        ],
        response_format=QuestionAnswerModel,
    )).questions


async def generate_questions(
    args: GenerateQuestionsArgs, entities: List[str], batch_size: int = 10
) -> AsyncGenerator[str, None]:
    """
    Generate questions based on provided entities and text chunk. Returns a
    generator of questions instead of a list to facilitate streaming.

    Args:
        args (GenerateQuestionsArgs): Arguments including client, model, chunk,
        source, and source_type. entities (List[str]): List of entities to
        generate questions for. batch_size (int, optional): Number of entities
        to process in each batch. Defaults to 10.

    Yields:
        str: Generated questions one by one.
    """
    for i in range(0, len(entities), batch_size):  # iterate in batches of 10's
        questions_batch = await _generate_questions(args, entities[i : i + batch_size])
        for question in questions_batch:
            yield question


DEFAULT_QUESTION_ANSWER_MODEL = "meta-llama-3.1-8b-instruct-q6_k"


async def get_answer(client: AsyncOpenAI, chunk: str, question: str) -> str:
    return await get_messages_response_async(
        client=client,
        model=get_model(DEFAULT_QUESTION_ANSWER_MODEL),
        messages=[
            {
                "role": "system",
                "content": ANSWER_EXTRACTION_SYSTEM_PROMPT.format(text=chunk),
            },
            {"role": "user", "content": question},
        ],
    )
