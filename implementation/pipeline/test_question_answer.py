import unittest
from helpers import get_async_client
from question_answer import QuestionWithChunk
from question_answer import GetAnswerAgent


class TestQuestionAnswer(unittest.IsolatedAsyncioTestCase):
    async def test_get_answer(self):
        agent = GetAnswerAgent()
        response = await agent.handle(
            QuestionWithChunk(
                question="What is my favorite food?", chunk="My favorite food is sushi"
            )
        )
        self.assertTrue("sushi" in response.lower())
