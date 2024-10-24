import unittest
from helpers import get_default_client
from question_answer import get_answer

class TestQuestionAnswer(unittest.TestCase):
    def test_get_answer(self):
        client = get_default_client()
        answer = get_answer(
            client,
            chunk="My favorite food is sushi",
            question="What is my favorite food?"
        )
        self.assertTrue("sushi" in answer.lower())
