from openai import AsyncOpenAI
import pandas as pd
import asyncio
import os
from pydantic import SecretStr
from helpers import (
    get_simple_response,
    get_openai_client,
    get_model,
    get_async_client,
    get_default_client,
)
from enum import Enum
from typing import AsyncIterator, Awaitable, TypedDict, List, Callable

from agent import Cache
from helpers import LM_STUDIO_BASE_URL


class BetterAnswer(Enum):
    A = "A"
    B = "B"


class EvalWinner(Enum):
    BASE = "BASE"
    FINE_TUNED = "FINE_TUNED"


class EvalQuestion(TypedDict):
    context: str
    question: str
    true_answer: str


JUDGE_SYSTEM_PROMPT = """
    You are an expert judge that determines which one of two answers is better.
    The context is {context}.
    The question is {question}.
    The true answer is {true_answer}.
    The first answer is {answer_a}.
    The second answer is {answer_b}.

    First, think step by step about which answer is better.
    If an answer contains factually incorrect information or contradicts the true answer
    it should be strongly penalized.
    If the question refers to a specific source, and the answer indicates that
    it is not aware about the specific source, it should be penalized even if it
    gives good general information.
    An answer that provides specific and correct information should be preferred to a
    more vague answer that provides general information.

    When you have made your decision, complete your response with the string
    "ANSWER_A" if A is better, and "ANSWER_B" if B is better (exclude the
    quotes).
"""

JUDGE_MODEL = "gpt-4o"  # We want to use a good model here.


async def judge(
    context: str, question: str, true_answer: str, answer_a: str, answer_b: str
) -> BetterAnswer:

    prompt = JUDGE_SYSTEM_PROMPT.format(
        context=context,
        question=question,
        true_answer=true_answer,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    # Use default OpenAI client for gpt4
    client = AsyncOpenAI()

    response = await get_simple_response(client, get_model(JUDGE_MODEL), prompt)

    if response.strip().endswith("ANSWER_A"):
        return BetterAnswer.A
    elif response.strip().endswith("ANSWER_B"):
        return BetterAnswer.B

    # LLM didn't follow instructions, lets try some basic heuristics

    if "ANSWER_A" in response.strip() and "ANSWER_B" in response.strip():
        raise ValueError(
            f"Shouldn't have both ANSWER_A and ANSWER_B in response: {response}"
        )
    elif "ANSWER_A" in response.strip():
        return BetterAnswer.A
    elif "ANSWER_B" in response:
        return BetterAnswer.B
    else:
        raise ValueError(
            f"Unexpected response format (should end with ANSWER_A or ANSWER_B): {response}"
        )


ResponseGenerator = Callable[[str], Awaitable[str]]


class JudgeResult:
    def __init__(
        self,
        question: str,
        context: str,
        true_answer: str,
        base_answer: str,
        finetuned_answer: str,
        better_answer: EvalWinner,
    ):
        self.question = question
        self.context = context
        self.true_answer = true_answer
        self.base_answer = base_answer
        self.finetuned_answer = finetuned_answer
        self.better_answer = better_answer

    def to_json(self) -> dict:
        return {
            "question": self.question,
            "context": self.context,
            "true_answer": self.true_answer,
            "base_answer": self.base_answer,
            "finetuned_answer": self.finetuned_answer,
            "better_answer": self.better_answer.value,
        }

    def save_json(self, filename: str) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def __str__(self):
        separator = "=" * 80
        dash = "-" * 80
        return (
            f"Question: {self.question}\n"
            f"Context: {self.context}\n"
            f"\n{separator}\n"
            f"TRUE ANSWER:\n"
            f"{dash}\n"
            f"{self.true_answer}\n"
            f"\n{separator}\n"
            f"BASE MODEL ANSWER:\n"
            f"{dash}\n"
            f"{self.base_answer}\n"
            f"\n{separator}\n"
            f"FINETUNED MODEL ANSWER:\n"
            f"{dash}\n"
            f"{self.finetuned_answer}\n"
            f"{separator}\n"
            f"Better Answer: {self.better_answer}"
        )


async def judge_questions(
    eval_questions: List[EvalQuestion],
    get_base_response: ResponseGenerator,
    get_finetuned_response: ResponseGenerator,
) -> AsyncIterator[JudgeResult]:

    judge_cache = Cache(f"cache/{JUDGE_MODEL}_judge_cache")
    max_concurrency = 16
    semaphore = asyncio.Semaphore(max_concurrency)

    async def judge_question(eval_question: EvalQuestion) -> JudgeResult:
        context = eval_question["context"]
        question = eval_question["question"]
        true_answer = eval_question["true_answer"]

        base_answer, finetuned_answer = await asyncio.gather(
            get_base_response(question), get_finetuned_response(question)
        )

        better_answer = await judge_cache.apply(
            lambda: judge(
                context, question, true_answer, base_answer, finetuned_answer
            ),
            (
                JUDGE_SYSTEM_PROMPT,
                context,
                question,
                true_answer,
                base_answer,
                finetuned_answer,
            ),
        )

        if better_answer == BetterAnswer.A:
            model_type = EvalWinner.BASE
        elif better_answer == BetterAnswer.B:
            model_type = EvalWinner.FINE_TUNED
        else:
            raise ValueError(f"Unexpected better answer: {better_answer}")

        return JudgeResult(
            question, context, true_answer, base_answer, finetuned_answer, model_type
        )

    async def judge_question_wrapper(eval_question: EvalQuestion) -> JudgeResult:
        async with semaphore:
            return await judge_question(eval_question)

    tasks = [
        asyncio.create_task(judge_question_wrapper(eval_question))
        for eval_question in eval_questions
    ]

    for task in tasks:
        yield await task


TEST_DATASET = "outputs/zack6_test_output.csv"
FINETUNED_MODEL = "2024nov14_arxiv_qa_data_3"
BASE_MODEL = "llama-3.2-3b-instruct"
FINETUNED_MODEL_ENDPOINT = (
    "https://lwylxx9n5zky48k6.us-east-1.aws.endpoints.huggingface.cloud"
)
FINETUNED_MODEL_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

BASE_MODEL_ENDPOINT = (
    "https://kndmrt9vgkei36qm.us-east-1.aws.endpoints.huggingface.cloud"
)
BASE_MODEL_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


async def main():
    df = pd.read_csv(TEST_DATASET)
    finetune_entries = df.to_dict("records")
    print(f"Loaded {len(finetune_entries)} finetune entries from {TEST_DATASET}")

    eval_questions = [
        EvalQuestion(
            context=entry["source"],
            question=entry["question"],
            true_answer=entry["answer"],
        )
        for entry in finetune_entries
    ]

    base_client = AsyncOpenAI(
        base_url=BASE_MODEL_ENDPOINT,
        api_key=BASE_MODEL_API_KEY,
    )
    finetuned_client = AsyncOpenAI(
        base_url=FINETUNED_MODEL_ENDPOINT,
        api_key=FINETUNED_MODEL_API_KEY,
    )

    base_cache = Cache(f"cache/{BASE_MODEL}_judge_cache")
    finetuned_cache = Cache(f"cache/{FINETUNED_MODEL}_judge_cache")

    async def get_base_response(question: str) -> str:
        return await base_cache.apply(
            lambda: get_simple_response(base_client, BASE_MODEL, question), question
        )

    async def get_finetuned_response(question: str) -> str:
        return await finetuned_cache.apply(
            lambda: get_simple_response(finetuned_client, FINETUNED_MODEL, question),
            question,
        )

    results = []
    i = 0
    async for result in judge_questions(
        eval_questions, get_base_response, get_finetuned_response
    ):
        results.append(result)
        print(f"Winner: {result.better_answer}")
        if result.better_answer == EvalWinner.BASE:
            print(f"Saving base model win {i}")
            print(result)
            result.save_json(
                f"judge_failures/{FINETUNED_MODEL}/base_model_win_{i}.json"
            )
        i += 1
    # Count winners
    base_wins = sum(1 for r in results if r.better_answer == EvalWinner.BASE)
    finetuned_wins = sum(1 for r in results if r.better_answer == EvalWinner.FINE_TUNED)

    print("\nFinal Results:")
    print(f"Base Model Wins: {base_wins}")
    print(f"Fine-tuned Model Wins: {finetuned_wins}")
    print(f"Total Comparisons: {len(results)}")
    if len(results) > 0:
        print(f"Fine-tuned Win Rate: {(finetuned_wins/len(results))*100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
