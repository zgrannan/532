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
from agent import MapAgent
from agent import StatelessAgent
from pipeline_types import FinetuneEntry
from pipeline_types import (
    EnrichedPdfChunkWithEntities,
    EnrichedPdfChunkWithQuestion,
    EnrichedPdfChunk
)
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTION_EXTRACTION_SYSTEM_PROMPT = """
Given a piece of text, generate enough diverse questions to thoroughly explore the key topics mentioned.
Generate a set of 'what, why, summarize, where, how questions, ensuring the questions cover different aspects of the text.
The questions should follow these instructions:

# Instructions

1. Generate a mix of 'what, why, summarize, where, how questions.
2. Ensure the questions are diverse and cover the topics identified in the text.
3. ALWAYS Include the source information in the questions.
4. Connect main ideas and themes in the text to the questions.
5. DO NOT INCLUDE anything other than the questions in the output.
6. Generate questions that can be accurately answered based on the text provided.
7. Extract the questions as a list in JSON format


Below is an example

# Example

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

DO NOT GENERATE QUESTIONS FROM THIS EXAMPLE. USE THE TEXT PROVIDED IN THE QUESTION PROMPT.

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

FURTHER_QUESTIONS_PROMPT = """
Given a list of example questions about a document, create a new set of diverse questions that maintain alignment with the themes, styles, and types of inquiries shown in the examples. 

Ensure that the new questions:

- Cover a variety of topics and perspectives introduced by the examples.
- Vary in structure (e.g., open-ended and factual).
- Introduce new angles or aspects of inquiry that remain relevant to the document's subject.
- Focus on generating questions that promotes deeper analysis, provide context, and encourage a comprehensive understanding of the document content

Return at most 10 new questions. 

Questions:
{question}


Return a list of new questions in JSON format.
"""

from typing import TypedDict


class QuestionGenerator(
    OpenAIAgent,
    StatelessAgent[EnrichedPdfChunk, EnrichedPdfChunkWithQuestion],
):
    def __init__(self, model: str, batch_size: int = 10, model_provider: str = "LMStudio"):
        super().__init__(model=model, model_provider=model_provider)
        StatelessAgent.__init__(self, name="Question Generator")
        self.batch_size = batch_size

    async def process_element(
        self, input: EnrichedPdfChunk
    ) -> AsyncIterator[EnrichedPdfChunkWithQuestion]:

        for question in await self._generate_questions(
            input["chunk"], input["source"], input["source_type"]
        ):
            yield EnrichedPdfChunkWithQuestion(
                filename=input["filename"],
                source=input["source"],
                source_type=input["source_type"],
                chunk=input["chunk"],
                question=question,
            )

    async def _generate_questions(
        self, chunk: str, source: str, source_type: str,
    ) -> List[str]:
        qa_prompt = QUESTION_EXTRACTION_SYSTEM_PROMPT.format(
            text=chunk,
            source=source,
            source_type=source_type,
        )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Length of qa_prompt: {len(qa_prompt)}")
        if self.model_provider == "LMStudio":
            response_format = QuestionAnswerModel
        else:
            response_format = {
                "type": "json_object", 
                "schema": QuestionAnswerModel.model_json_schema(),
            }
        init_questions =  await get_json_response_async(
                                                        client=self.client,
                                                        model=self.model,
                                                        messages=[
                                                            {"role": "system", "content": qa_prompt},
                                                        ],
                                                        response_format=response_format,
                                                        agent_name=f"Initial {self.name}",
                                                    )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Length of init_questions: {len(init_questions.questions)}")
        additional_question_prompt = FURTHER_QUESTIONS_PROMPT.format(question="\n".join([q for q in init_questions.questions]))
        additional_questions = await get_json_response_async(
                                                        client=self.client,
                                                        model=self.model,
                                                        messages=[
                                                            {"role": "system", "content": additional_question_prompt},
                                                        ],
                                                        response_format=response_format,
                                                        agent_name=f"Secondary {self.name}",
                                                    )
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Length of additional_questions: {len(additional_questions.questions)}")
        return init_questions.questions + additional_questions.questions

class QAPair(TypedDict):
    question: str
    answer: str
    chunk: str
    chunk_index: int


class QuestionWithChunk(TypedDict):
    question: str
    chunk: str


class GetAnswerAgent(OpenAIAgent, MapAgent[QuestionWithChunk, str]):
    def __init__(self, model, model_provider: str = "LMStudio"):
        super().__init__(model=model, model_provider=model_provider)
        MapAgent.__init__(self, name="Get Answer Agent")

    async def handle(self, input: QuestionWithChunk) -> str:
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating answer for Question: {input['question']}")
            return await get_messages_response_async(
                client=self.client,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": ANSWER_EXTRACTION_SYSTEM_PROMPT.format(
                            text=input["chunk"]
                        ),
                    },
                    {"role": "user", "content": input["question"]},
                ],
                agent_name=self.name,
            )
        except Exception as e:
            # Return default answer, ideally code should not reach here
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in GetAnswerAgent handle(), {str(e)}")
            return "ERROR ANSWER NOT FOUND"
