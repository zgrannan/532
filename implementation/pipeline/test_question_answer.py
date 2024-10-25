import unittest
from helpers import get_async_client
from question_answer import GetAnswerAgent


class TestQuestionAnswer(unittest.IsolatedAsyncioTestCase):
    async def test_get_answer(self):
        agent = GetAnswerAgent(chunk="My favorite food is sushi")
        async for response in agent.process_once("What is my favorite food?"):
            self.assertTrue("sushi" in response["answer"].lower())
