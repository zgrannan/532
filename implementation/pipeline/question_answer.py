from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging

from pipeline.helpers import get_json_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUESTION_EXTRACTION_SYSTEM_PROMPT = \
"""
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


ANSWER_EXTRACTION_SYSTEM_PROMPT = \
"""
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

def generate_questions(
    client: OpenAI,
    model: str,
    chunk: str,
    entities: List[str],
    source: str,
    source_type: str
) -> List[str]:
    qa_prompt = QUESTION_EXTRACTION_SYSTEM_PROMPT.format(
        text=chunk,
        entities=", ".join(entities),
        source=source,
        source_type=source_type
    )
    print(f"Length of qa_prompt: {len(qa_prompt)}")
    return get_json_response(
        client=client,
        model=model,
        messages=[
            {
                "role": "system",
                "content":qa_prompt
            },
        ],
        response_format=QuestionAnswerModel,
    ).questions
