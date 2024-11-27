from agent import StatelessAgent, Agent, OpenAIMessagesAgent
from langchain_chroma import Chroma
from typing import List, AsyncIterator, Dict, Tuple
from agent import LLMClientSettings
from pipeline_types import FinetuneEntry
from datetime import datetime
import logging

REFINED_RAG_ANSWER_PROMPT = """
You are tasked with answering questions based on a provided text.
You are provided with a question and an initial answer.
You are also provided with some supporting documentation to help create a new response

Your goal is to generate high-quality, detailed answers by following these instructions:
If the answer is not found in the text, respond with "NO ANSWER FOUND"

# Instructions:
1. Reference the Text: Answer directly using relevant details from the text. Avoid introducing unsupported claims.
2. Comprehensive Response: Address all parts of the question thoroughly, covering multiple aspects if needed.
3. Detail-Oriented: Highlight key elements like techniques, processes, models, or challenges, expanding on them for clarity.
4. Organized Structure: Use clear paragraphs or points for complex answers.
5. Clarity and Examples: Ensure the answer is precise and easy to follow. Include examples or quotes from the text when applicable.
6. Include Sources: Clearly reference the source information at the end of the answer.
7. Include only the answer in your response

Question:
{question}

Initial Answer:
{answer}

Supporting Documentation:
{docs}
"""


class GetRAGAnswerAgent(Agent[FinetuneEntry, FinetuneEntry]):

    def __init__(self, settings: LLMClientSettings, vector_store: Chroma, k: int = 5):
        super().__init__("Get RAG Answer Agent")
        self.messages_agent = OpenAIMessagesAgent(settings)
        self.vector_store = vector_store
        self.k = k

    def edges(self) -> List[Tuple[int, int]]:
        return [
            *(self.messages_agent.edges()),
            (self.id, self.messages_agent.id),
            (self.messages_agent.id, self.id),
        ]

    def nodes(self) -> Dict[int, str]:
        return {**self.messages_agent.nodes(), self.id: self.name}

    async def _process(
        self, inputs: AsyncIterator[FinetuneEntry]
    ) -> AsyncIterator[FinetuneEntry]:
        # Test rag

        # Get docs from vector store
        async for input in inputs:
            try:
                if isinstance(input, dict) and "pass_through" in input and input["pass_through"]:
                    logging.debug(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Pass through: {input['question']}")
                    yield input # Pass through the original question answer pair
                    continue
                else:
                    logging.debug(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Searching filter=filename: {input['source']}"
                    )
                    docs = await self.vector_store.asimilarity_search(
                        query=input["question"],
                        k=self.k,
                        filter={"filename": input["source"]},
                    )
                    docs_str = "\n".join([r.page_content for r in docs])
                    logging.debug(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating RAG answer for Question: {input['question']}"
                    )
                    resp = await self.messages_agent.handle(
                        [
                            {
                                "role": "system",
                                "content": REFINED_RAG_ANSWER_PROMPT.format(
                                    question=input["question"],
                                    answer=input["answer"],
                                    docs=docs_str,
                                ),
                            },
                        ],
                    )
                    if resp != "NO ANSWER FOUND":
                        yield {**input, "answer": resp}
            except Exception as e:
                logging.error(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error generating RAG answer: {e}"
                )
