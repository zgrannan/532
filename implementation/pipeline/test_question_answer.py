import unittest
from helpers import get_async_client
from question_answer import get_answer

class TestQuestionAnswer(unittest.IsolatedAsyncioTestCase):
    async def test_get_answer(self):
        client = get_async_client()
        answer = await get_answer(
            client,
            chunk="My favorite food is sushi",
            question="What is my favorite food?"
        )
        self.assertTrue("sushi" in answer.lower())
