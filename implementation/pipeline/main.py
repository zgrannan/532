from uuid import uuid4
import PyPDF2
import asyncio
from chromadb import Documents
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI, AsyncOpenAI
import os
import time
from helpers import get_default_client
from agent import Agent, OpenAIAgent, Compose
from entity_extraction import EntityExtractionAgent
from helpers import once
from helpers import get_json_response_async
from agent import EnrichAgent, UnchunkingAgent
from helpers import get_messages_response_async
from agent import TakeOnly
from question_answer import QuestionWithChunk
from pipeline_types import FinetuneEntry
from pipeline_types import EnrichedPdfChunk
from pipeline_types import EnrichedPdfFile
from pipeline_types import (
    EnrichedPdfChunkWithEntities,
    EnrichedPdfChunkWithQuestion,
)
from question_answer import QAPair
from question_answer import GetAnswerAgent
from question_answer import QuestionGenerator
from utils.pdf import read_pdf
from langchain_core.documents import Document
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Coroutine,
    Generator,
    List,
    TypeVar,
    TypedDict,
    Tuple,
    cast,
)
from pydantic import BaseModel
from helpers import (
    get_async_client,
    split_text,
    list_of_dicts_to_dict_of_lists,
    upload_to_hf,
    remove_duplicates,
)
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import argparse
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv(find_dotenv())


REFINED_QUESTIONS_PROMPT = """
    You are provided with a question and an answer.
    Your job is to generate a set of new questions that can be answered with the given answer but is diverse and approaches
    the original question from different perspectives.

    Ensure that the generated questions are clear, purposeful, specific, and invoke critical thinking
    Question:
    {question}

    Answer:
    {answer}

    Return a list of new questions in JSON format.
"""

DEFAULT_REFINE_QUESTIONS_MODEL = "meta-llama-3.1-8b-instruct"


class RefinedQuestionsModel(BaseModel):
    questions: List[str]


class RefineQuestionsAgent(OpenAIAgent[FinetuneEntry, FinetuneEntry]):
    def __init__(self):
        super().__init__(DEFAULT_REFINE_QUESTIONS_MODEL)
        self.name = "Refine Questions Agent"

    async def _process(
        self, inputs: AsyncIterator[FinetuneEntry]
    ) -> AsyncIterator[FinetuneEntry]:
        async for input in inputs:
            print(f"Generating refined question for Question: {input['question']}")
            resp = await get_json_response_async(
                client=self.client,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": REFINED_QUESTIONS_PROMPT.format(
                            question=input["question"],
                            answer=input["answer"],
                        ),
                    },
                ],
                response_format=RefinedQuestionsModel,
            )
            yield input  # Also return the original question
            for question in resp.questions:
                print("Refined Question: ", question)
                yield {
                    **input,
                    "question": question,
                }


class EmbedChunksAgent(Agent[List[EnrichedPdfChunk], List[EnrichedPdfChunk]]):
    def __init__(self, embeddings_func: OpenAIEmbeddings, vector_store: Chroma):
        super().__init__("Embed Chunks Agent")
        self.embeddings_func = embeddings_func
        self.vector_store = vector_store

    async def _process(
        self, inputs: AsyncIterator[List[EnrichedPdfChunk]]
    ) -> AsyncIterator[List[EnrichedPdfChunk]]:
        async for input_list in inputs:
            docs = [
                Document(
                    page_content=input["chunk"],
                    metadata={
                        "filename": os.path.splitext(
                            os.path.basename(input["filename"])
                        )[0]
                    },
                )
                for input in input_list
            ]
            uuids = [str(uuid4()) for _ in input_list]

            await self.vector_store.aadd_documents(docs, ids=uuids)
            yield input_list


class RemoveDuplicateQuestionsAgent(
    Agent[EnrichedPdfChunkWithQuestion, EnrichedPdfChunkWithQuestion]
):
    async def _process(
        self, inputs: AsyncIterator[EnrichedPdfChunkWithQuestion]
    ) -> AsyncIterator[EnrichedPdfChunkWithQuestion]:
        seen = set()
        async for input in inputs:
            if input["question"] not in seen:
                seen.add(input["question"])
                yield input


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


async def slurp_iterator(generator: AsyncIterator[T]) -> List[T]:
    return [item async for item in generator]


class ChunkTextAgent(Agent[EnrichedPdfFile, EnrichedPdfChunk]):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        super().__init__(f"Chunk Text Agent ({chunk_size})")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def _process(
        self, inputs: AsyncIterator[EnrichedPdfFile]
    ) -> AsyncIterator[EnrichedPdfChunk]:
        async for input in inputs:
            chunks = split_text(input["text"], self.chunk_size, self.chunk_overlap)
            for index, chunk in enumerate(chunks):
                yield EnrichedPdfChunk(
                    filename=input["filename"],
                    source=input["source"],
                    source_type=input["source_type"],
                    chunk=chunk,
                )


# class RefineQuestionAgent(Agent[FinetuneEntry, FinetuneEntry]):
#     async def _process(
#         self, inputs: AsyncIterator[FinetuneEntry]
#     ) -> AsyncIterator[FinetuneEntry]:
#         print("Refining questions...")
#         async for input in inputs:
#             print(f"Refining question: {input['question']}")
#             yield input


def extract_title(pdf_path) -> str:
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)

        # Attempt to extract title from metadata
        if reader.metadata is not None:
            if "/Title" in reader.metadata and reader.metadata["/Title"]:
                return cast(str, reader.metadata["/Title"])

        print("Using filename as title")
        return os.path.splitext(os.path.basename(pdf_path))[0]


class EnrichPdfFileAgent(Agent[str, EnrichedPdfFile]):
    async def _process(
        self, inputs: AsyncIterator[str]
    ) -> AsyncIterator[EnrichedPdfFile]:
        async for filename in inputs:
            source = extract_title(filename)
            text = read_pdf(filename)
            yield EnrichedPdfFile(
                filename=filename, source=source, source_type="paper", text=text
            )


async def generate_finetune_entries_for_files_in_directory(
    directory: str,
) -> List[FinetuneEntry]:
    pdf_files = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".pdf")
    ]

    for file in pdf_files:
        print(f"Processing file: {file}")

    RUN_NAME = "20241028165628"
    EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5@f32"  # on LM Studio
    chunk_size = 5000
    chunk_overlap = 100
    entity_batch_size = 10
    question_batch_size = 10
    model = "meta-llama-3.1-8b-instruct-q6_k"

    embeddings_func = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:1234/v1",
        api_key="test",
        check_embedding_ctx_length=False,  # https://github.com/langchain-ai/langchain/issues/21318
    )

    vector_store = Chroma(
        collection_name=RUN_NAME,  # Config name,
        embedding_function=embeddings_func,
        persist_directory="./chroma_langchain_db",
    )

    pipeline = (
        EnrichPdfFileAgent()
        .and_then(ChunkTextAgent(chunk_size, chunk_overlap))
        .chunk(10)  # We embed 10 text chunks at a time
        .and_then(EmbedChunksAgent(embeddings_func, vector_store))
        .and_then(UnchunkingAgent())  # Undo the previous chunking
        .fan_out(
            10,
            EnrichAgent(
                EntityExtractionAgent(entity_batch_size),
                lambda e: e["chunk"],
                lambda e, entities: EnrichedPdfChunkWithEntities(
                    filename=e["filename"],
                    source=e["source"],
                    source_type=e["source_type"],
                    chunk=e["chunk"],
                    entities=entities,
                ),
            ),
        )
        .fan_out(10, QuestionGenerator(model, question_batch_size))
        .and_then(RemoveDuplicateQuestionsAgent())
        .fan_out(
            10,
            EnrichAgent(
                GetAnswerAgent(),
                lambda e: QuestionWithChunk(question=e["question"], chunk=e["chunk"]),
                lambda e, answer: FinetuneEntry(
                    filename=e["filename"],
                    source=e["source"],
                    source_type=e["source_type"],
                    chunk=e["chunk"],
                    question=e["question"],
                    answer=answer,
                ),
            ),
        )
        .fan_out(10, RefineQuestionsAgent())
        .and_then(RemoveSimilarQuestionsAgent(embeddings_func, 0.9))
        .fan_out(10, GetRAGAnswerAgent(vector_store))
    )
    print(pipeline.to_dot())

    return await slurp_iterator(pipeline.process_list(pdf_files))


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


class GetRAGAnswerAgent(OpenAIAgent[FinetuneEntry, FinetuneEntry]):

    def __init__(self, vector_store: Chroma, k: int = 3):
        super().__init__(
            model=DEFAULT_REFINE_QUESTIONS_MODEL,
            embedding_model="text-embedding-nomic-embed-text-v1.5@f32",
        )
        self.name = "Get RAG Answer Agent"
        self.vector_store = vector_store
        self.k = k

    async def _process(
        self, inputs: AsyncIterator[FinetuneEntry]
    ) -> AsyncIterator[FinetuneEntry]:
        # Test rag

        # Get docs from vector store
        async for input in inputs:
            print("similarity search")
            docs = await self.vector_store.asimilarity_search(
                query=input["question"], k=self.k, filter={"filename": input["source"]}
            )
            docs_str = "\n".join([r.page_content for r in docs])
            print(f"Generating answer for Question: {input['question']}")
            resp = await get_messages_response_async(
                client=self.client,
                model=self.model,
                messages=[
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
            yield input  # Also return original entry
            yield {**input, "answer": resp}


class RemoveSimilarQuestionsAgent(Agent[FinetuneEntry, FinetuneEntry]):
    def __init__(self, embeddings_func: OpenAIEmbeddings, threshold: float):
        super().__init__("Remove Similar Questions Agent")
        self.embeddings_func = embeddings_func
        self.threshold = threshold

    async def _process(
        self, inputs: AsyncIterator[FinetuneEntry]
    ) -> AsyncIterator[FinetuneEntry]:
        input_list = await slurp_iterator(inputs)
        embeddings = [
            self.embeddings_func.embed_query(input["question"]) for input in input_list
        ]
        similarity_matrix = cosine_similarity(embeddings)

        # Set diagonal to 0 to avoid self-matches
        np.fill_diagonal(similarity_matrix, 0)

        # Find pairs above threshold
        similar_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i][j] > self.threshold:
                    similar_pairs.append((i, j))

        indices_to_remove = set()
        # For each similar pair, remove the second question
        for pair in similar_pairs:
            indices_to_remove.add(pair[1])

        filtered_data = [
            item for i, item in enumerate(input_list) if i not in indices_to_remove
        ]
        for item in filtered_data:
            yield item


async def main():
    start_time = time.time()

    # Inputs
    ollama_base_url_embeddings = "http://localhost:11434/api/embeddings"
    embedding_model = "nomic-embed-text"

    # Hugging Face
    repo_id = "CPSC532/arxiv_qa_data"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default=datetime.now().strftime("%Y%m%d%H%M%S"),
        help="Name of the config to use (default: timestamp)",
    )
    args = parser.parse_args()
    config_name = args.config_name

    # First stage getting initial Q/A Pairs
    finetune_entries = await generate_finetune_entries_for_files_in_directory("../data")
    df = pd.DataFrame(finetune_entries)

    # if outputs folder doesn't exist, create it
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    df.to_csv(f"outputs/{config_name}_output.csv", index=False)
    # Seconds stage refining Q/A Pairs and getting answers from chunk + RAG

    # Save to csv using Pandas

    # Upload to Huggingface
    qa_pairs_dict = list_of_dicts_to_dict_of_lists(finetune_entries)
    upload_to_hf(
        data=qa_pairs_dict,
        repo_id=repo_id,
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        config_name=config_name,
    )

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")


if __name__ == "__main__":
    asyncio.run(main())
