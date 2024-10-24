from helpers import get_simple_response, get_lm_studio_client
from enum import Enum
from typing import TypedDict, List, Callable

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

JUDGE_MODEL = "meta-llama-3.1-70b-instruct"

def judge(
        context: str,
        question: str,
        true_answer: str,
        answer_a: str,
        answer_b: str
        ) -> BetterAnswer:

    prompt = JUDGE_SYSTEM_PROMPT.format(
        context=context,
        question=question,
        true_answer=true_answer,
        answer_a=answer_a,
        answer_b=answer_b
    )

    client = get_lm_studio_client()

    response = get_simple_response(client, JUDGE_MODEL, prompt)

    if response.strip().endswith("ANSWER_A"):
        return BetterAnswer.A
    elif response.strip().endswith("ANSWER_B"):
        return BetterAnswer.B
    else:
        raise ValueError(f"Unexpected response format (should end with ANSWER_A or ANSWER_B): {response}")

ResponseGenerator = Callable[[str], str]

def judge_questions(
        eval_questions: List[EvalQuestion],
        get_base_response: ResponseGenerator,
        get_finetuned_response: ResponseGenerator
        ) -> List[EvalWinner]:
    results = []

    for eval_question in eval_questions:
        context = eval_question["context"]
        question = eval_question["question"]
        true_answer = eval_question["true_answer"]

        base_answer = get_base_response(question)
        finetuned_answer = get_finetuned_response(question)

        better_answer = judge(context, question, true_answer, base_answer, finetuned_answer)

        if better_answer == BetterAnswer.A:
            model_type = EvalWinner.BASE
        elif better_answer == BetterAnswer.B:
            model_type = EvalWinner.FINE_TUNED

        results.append(model_type)

    return results
