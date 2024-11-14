from openai import AsyncOpenAI
import pandas as pd
import asyncio
from helpers import (
    get_simple_response,
    get_openai_client,
    get_model,
    get_async_client,
    get_default_client,
)
from enum import Enum
from typing import Awaitable, TypedDict, List, Callable


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
    You are an expert judge that determines which answer is better between two answers.
    The context is {context}.
    The question is {question}.
    The true answer is {true_answer}.
    The first answer is {answer_a}.
    The second answer is {answer_b}.

    First, think step by step about which answer is better. When you have made
    your decision, complete your response with the string "ANSWER_A" if A is
    better, and "ANSWER_B" if B is better (exclude the quotes).
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


async def judge_questions(
    eval_questions: List[EvalQuestion],
    get_base_response: ResponseGenerator,
    get_finetuned_response: ResponseGenerator,
) -> List[EvalWinner]:
    results: List[EvalWinner] = []

    for eval_question in eval_questions:
        context = eval_question["context"]
        question = eval_question["question"]
        true_answer = eval_question["true_answer"]

        base_answer = await get_base_response(question)
        finetuned_answer = await get_finetuned_response(question)

        better_answer = await judge(
            context, question, true_answer, base_answer, finetuned_answer
        )

        if better_answer == BetterAnswer.A:
            model_type = EvalWinner.BASE
        elif better_answer == BetterAnswer.B:
            model_type = EvalWinner.FINE_TUNED
        else:
            raise ValueError(f"Unexpected better answer: {better_answer}")

        # print(f"Question: {question}")
        # print(f"Context: {context}")
        # print(f"True Answer: {true_answer}")
        # print(f"Base Answer: {base_answer}")
        # print(f"Finetuned Answer: {finetuned_answer}")
        print(f"Better Answer: {model_type}")

        results.append(model_type)

    return results

TEST_DATASET = "outputs/zack5_test_output.csv"
FINETUNED_MODEL = "2024nov14_arxiv_qa_data_2"
BASE_MODEL = "llama-3.2-3b-instruct"

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

    client = get_async_client()

    get_base_response = lambda question: get_simple_response(
        client, BASE_MODEL, question
    )

    get_finetuned_response = lambda question: get_simple_response(
        client, FINETUNED_MODEL, question
    )

    results = await judge_questions(
        eval_questions, get_base_response, get_finetuned_response
    )

    print(results)


if __name__ == "__main__":
    asyncio.run(main())
