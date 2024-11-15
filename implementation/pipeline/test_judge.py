from typing import List
import unittest
from judge import judge, BetterAnswer, judge_questions, EvalWinner
from helpers import slurp_iterator
from judge import EvalQuestion


class TestJudge(unittest.IsolatedAsyncioTestCase):
    async def test_judge(self):
        self.assertEqual(
            (await judge("", "What is 2 + 2", "4", "3", "4")),
            BetterAnswer.B,
        )

    async def test_cases(self):
        import os
        import json

        test_cases_dir = "judge_test_cases"
        for filename in os.listdir(test_cases_dir):
            if filename.endswith(".json"):
                with open(os.path.join(test_cases_dir, filename)) as f:
                    test_case = json.load(f)

                result = await judge(
                    context=test_case["context"],
                    question=test_case["question"],
                    true_answer=test_case["true_answer"],
                    answer_a=test_case["base_answer"],
                    answer_b=test_case["finetuned_answer"],
                )

                if test_case["better_answer"] == "BASE":
                    expected = BetterAnswer.A
                elif test_case["better_answer"] == "FINETUNED":
                    expected = BetterAnswer.B
                else:
                    raise ValueError(
                        f"Unexpected better answer in test case {filename}: {test_case['better_answer']}"
                    )

                self.assertEqual(
                    result,
                    expected,
                    f"Test case {filename} failed - expected {test_case['better_answer']} to win",
                )

    async def test_judge_questions(self):
        async def base_response(question: str) -> str:
            return "4"

        async def fine_tuned_response(question: str) -> str:
            return "5"

        eval_questions: List[EvalQuestion] = [
            {"context": "", "question": "What is 2 + 2", "true_answer": "4"},
            {"context": "", "question": "What is 2 + 3", "true_answer": "5"},
        ]
        self.assertEqual(
            [
                result.better_answer
                for result in await slurp_iterator(
                    judge_questions(eval_questions, base_response, fine_tuned_response)
                )
            ],
            [EvalWinner.BASE, EvalWinner.FINE_TUNED],
        )
