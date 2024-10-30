import PyPDF2
import asyncio
from openai import OpenAI, AsyncOpenAI
import os
import time
from helpers import get_default_client
from agent import Agent, OpenAIAgent, ComposedAgent
from entity_extraction import EntityExtractionAgent
from helpers import once
from helpers import get_json_response_async
from agent import EnrichAgent
from pipeline_types import FinetuneEntry
from pipeline_types import EnrichedPdfChunk
from pipeline_types import EnrichedPdfFile
from pipeline_types import (
    EnrichedPdfChunkWithEntities,
    EnrichedPdfChunkWithQAPair,
    EnrichedPdfChunkWithQuestion,
    EnrichedPdfChunkWithRefinedQAPair,
)
from question_answer import QAPair
from question_answer import GetAnswerAgent
from question_answer import QuestionGenerator
from utils.pdf import read_pdf
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


class RefineQuestionsAgent(
    OpenAIAgent[EnrichedPdfChunkWithQAPair, EnrichedPdfChunkWithRefinedQAPair]
):
    def __init__(self):
        super().__init__(DEFAULT_REFINE_QUESTIONS_MODEL)

    async def _process(
        self, inputs: AsyncIterator[EnrichedPdfChunkWithQAPair]
    ) -> AsyncIterator[EnrichedPdfChunkWithQAPair]:
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
            for question in resp.questions:
                print("Refined Question: ", question)
                yield {
                    **input,
                    "question": question,
                    "original_question": input["question"],
                }


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

    chunk_size = 5000
    chunk_overlap = 100
    entity_batch_size = 10
    question_batch_size = 10
    model = "meta-llama-3.1-8b-instruct-q6_k"

    pipeline = (
        EnrichPdfFileAgent()
        .and_then(ChunkTextAgent(chunk_size, chunk_overlap))
        .fan_out(
            10,
            EnrichAgent(
                EntityExtractionAgent(entity_batch_size),
                lambda e: e["chunk"],
                lambda e, entities: cast(
                    EnrichedPdfChunkWithEntities, {**e, "entities": entities}
                ),
            ),
        )
        .fan_out(10, QuestionGenerator(model, question_batch_size))
        .and_then(RemoveDuplicateQuestionsAgent())
        .fan_out(10, GetAnswerAgent())
    )

    return await slurp_iterator(pipeline.process_list(pdf_files))


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
