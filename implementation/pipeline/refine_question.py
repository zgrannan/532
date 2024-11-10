from typing import List, AsyncIterator
from pydantic import BaseModel
from datetime import datetime
import random 
from agent import StatelessAgent, OpenAIAgent
from pipeline_types import FinetuneEntry, EnrichedPdfChunkWithQuestion
from helpers import get_json_response_async

DEFAULT_REFINE_QUESTIONS_MODEL = "meta-llama-3.1-8b-instruct"

REFINED_QUESTIONS_PROMPT = """
    You are provided with a question and an answer.
    Your job is to generate a set of new questions that can be answered with the given answer but is diverse and approaches
    the original question from different perspectives.

    Ensure that the generated questions are clear, purposeful, specific, and invoke critical thinking.
    Return at most 10 new questions.
    Question:
    {question}

    Answer:
    {answer}

    Return a list of new questions in JSON format.
"""

class RefinedQuestionsModel(BaseModel):
    questions: List[str]




class RefineQuestionsAgent(
    OpenAIAgent,
    StatelessAgent[FinetuneEntry, FinetuneEntry],
):
    def __init__(self, sampling_percent: float = 0.1):
        super().__init__(DEFAULT_REFINE_QUESTIONS_MODEL)
        StatelessAgent.__init__(self, name="Refine Questions Agent")
        self.sampling_percent = sampling_percent

    async def process_element(
        self, input: FinetuneEntry
    ) -> AsyncIterator[FinetuneEntry]:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating refined question for Question: {input['question']}"
        )
        resp = await get_json_response_async(
            client=self.client,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": REFINED_QUESTIONS_PROMPT.format(
                        question=input["question"],
                        answer=input["answer"],
                    ),
                },
            ],
            response_format=RefinedQuestionsModel,
            agent_name=self.name,
        )
        yield { # Also return the original question, since we are trying to 'pass through' original q/a pair, we need to pass through this as well in GetRAGAnswerAgent
            **input,
            "pass_through": True
          }  
        for question in resp.questions:
            if random.random() > self.sampling_percent:
                continue
            else:
                yield {
                    **input, # The answer 
                    "question": question,
                }


FURTHER_QUESTIONS_PROMPT = """
Given a list of example questions about a document, create a new set of diverse questions that maintain alignment with the themes, styles, and types of inquiries shown in the examples. Ensure that the new questions:

Cover a variety of topics and perspectives introduced by the examples.
Vary in structure (e.g., open-ended and factual).
Introduce new angles or aspects of inquiry that remain relevant to the document's subject.
Focus on generating questions that invite deeper analysis, provide context, and encourage a comprehensive understanding of the document content
    
Return at most 10 new questions. 

Question:
{question}


Return a list of new questions in JSON format.
"""


class RefineQuestionsAgentInit(
    OpenAIAgent,
    StatelessAgent[EnrichedPdfChunkWithQuestion, EnrichedPdfChunkWithQuestion],
):
    def __init__(self, sampling_percent: float = 0.1):
        super().__init__(DEFAULT_REFINE_QUESTIONS_MODEL)
        StatelessAgent.__init__(self, name="Refine Questions Agent")
        self.sampling_percent = sampling_percent

    async def process_element(
        self, input: EnrichedPdfChunkWithQuestion
    ) -> AsyncIterator[EnrichedPdfChunkWithQuestion]:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating refined question for Question: {input['question']}"
        )
        resp = await get_json_response_async(
            client=self.client,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": FURTHER_QUESTIONS_PROMPT.format(
                        question=input["question"],
                    ),
                },
            ],
            response_format=RefinedQuestionsModel,
            agent_name=self.name,
        )
        yield input  # Also return the original question
        for question in resp.questions:
            yield {
                **input,
                "question": question,
                "refined": True,
            }

# Add check that source is in question 