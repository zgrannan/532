import unittest
from judge import judge, BetterAnswer, judge_questions, EvalWinner

class TestJudge(unittest.TestCase):
    def test_judge(self):
        self.assertEqual(judge("", "What is 2 + 2", "4", "3", "4"), BetterAnswer.B)

    def test_judge_questions(self):
        def base_response(question: str) -> str:
            return "4"
        def fine_tuned_response(question: str) -> str:
            return "5"
        eval_questions = [
            {"context": "", "question": "What is 2 + 2", "true_answer": "4"},
            {"context": "", "question": "What is 2 + 3", "true_answer": "5"}
        ]
        self.assertEqual(
            judge_questions(
                eval_questions,
                base_response,
                fine_tuned_response
            ), [EvalWinner.BASE, EvalWinner.FINE_TUNED]
        )
