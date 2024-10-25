import PyPDF2
import asyncio
from openai import OpenAI, AsyncOpenAI
import os
import time
from helpers import get_default_client
from agent import Agent, OpenAIAgent, ComposedAgent
from entity_extraction import EntityExtractionAgent
from helpers import once
from question_answer import QAPair
from question_answer import GetAnswerAgent
from question_answer import QuestionGenerator
from utils.pdf import read_pdf
from typing import Any, AsyncGenerator, Coroutine, Generator, List, TypeVar, TypedDict
from helpers import (
    get_async_client,
    split_text,
    list_of_dicts_to_dict_of_lists,
    upload_to_hf,
    remove_duplicates,
)
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class QuestionsFromChunkArgs(TypedDict):
    chunk: str
    source: str
    source_type: str


class QuestionsFromChunkAgent(OpenAIAgent[QuestionsFromChunkArgs, str]):
    def __init__(self, model: str, entity_batch_size: int, question_batch_size: int):
        super().__init__(model)
        self.entity_batch_size = entity_batch_size
        self.question_batch_size = question_batch_size
        self.entity_extraction_agent = EntityExtractionAgent(
            max_entities=self.entity_batch_size
        )
        self.question_generator_agent = QuestionGenerator(
            model=self.model, batch_size=self.question_batch_size
        )

    async def _process(
        self, inputs: AsyncGenerator[QuestionsFromChunkArgs, None]
    ) -> AsyncGenerator[str, None]:
        assert isinstance(inputs, AsyncGenerator), "inputs must be an AsyncGenerator"
        async for input in inputs:
            entities = (
                await slurp_generator(
                    self.entity_extraction_agent.process(once(input["chunk"]))
                )
            )[0]
            print(f"Number of entities extracted: {len(entities)}")
            async for question in self.question_generator_agent.process(
                once(
                    {
                        "chunk": input["chunk"],
                        "source": input["source"],
                        "source_type": input["source_type"],
                        "entities": entities,
                    }
                )
            ):
                yield question


class RemoveDuplicateQuestionsAgent(Agent[str, str]):
    async def _process(
        self, inputs: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        assert isinstance(inputs, AsyncGenerator), "inputs must be an AsyncGenerator"
        seen = set()
        async for question in inputs:
            if question not in seen:
                seen.add(question)
                yield question


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


async def slurp_generator(generator: AsyncGenerator[T, None]) -> List[T]:
    return [item async for item in generator]


class QAPairAgent(Agent[str, QAPair]):
    def __init__(self, model: str, source: str, source_type: str):
        super().__init__()
        self.source = source
        self.source_type = source_type
        self.questions_from_chunk_agent = (
            QuestionsFromChunkAgent(
                model=model, entity_batch_size=10, question_batch_size=10
            )
            .and_then(RemoveDuplicateQuestionsAgent())
            .chunk(10)
        )

    async def _process(
        self, inputs: AsyncGenerator[str, None]
    ) -> AsyncGenerator[QAPair, None]:
        async for chunk in inputs:
            answers_agent = self.questions_from_chunk_agent.parallel(
                lambda: GetAnswerAgent(chunk)
            )
            async for pair in answers_agent.process_once(
                {
                    "chunk": chunk,
                    "source": self.source,
                    "source_type": self.source_type,
                }
            ):
                yield pair


class FinetuneEntry(TypedDict):
    question: str
    answer: str
    source: str


class ChunkTextAgent(Agent[str, str]):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def _process(
        self, inputs: AsyncGenerator[str, None]
    ) -> AsyncGenerator[str, None]:
        async for input in inputs:
            chunks = split_text(input, self.chunk_size, self.chunk_overlap)
            for chunk in chunks:
                yield chunk


async def generate_qa_pairs(text: str, source: str, source_type: str) -> List[QAPair]:
    model = "meta-llama-3.1-8b-instruct-q6_k"
    chunk_agent = (
        ChunkTextAgent(chunk_size=5000, chunk_overlap=100)
        .chunk(10)
        .parallel(lambda: QAPairAgent(model, source, source_type))
    )
    return await slurp_generator(chunk_agent.process(once(text)))


async def generate_finetune_entries(
    filename: str, source: str, source_type: str
) -> List[FinetuneEntry]:
    text = read_pdf(filename)
    qa_pairs = await generate_qa_pairs(text, source, source_type)
    return [
        {
            "question": qa["question"],
            "answer": qa["answer"],
            "source": source,
        }
        for qa in qa_pairs
    ]


def extract_title(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        # Attempt to extract title from metadata
        if "/Title" in reader.metadata:
            return reader.metadata["/Title"]
        raise ValueError("No title found in metadata")


async def generate_finetune_entries_from_file(filename: str) -> List[FinetuneEntry]:
    source = extract_title(filename)
    return await generate_finetune_entries(filename, source, "paper")


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

    tasks = [generate_finetune_entries_from_file(file_path) for file_path in pdf_files]
    results = await asyncio.gather(*tasks)

    finetune_entries = [entry for result in results for entry in result]
    return finetune_entries


async def main():
    start_time = time.time()

    # Inputs
    ollama_base_url_embeddings = "http://localhost:11434/api/embeddings"
    embedding_model = "nomic-embed-text"

    # Hugging Face
    repo_id = "CPSC532/arxiv_qa_data"
    config_name = "test_dataset_2024OCT25"

    finetune_entries = await generate_finetune_entries_for_files_in_directory("../data")

    # Save to csv using Pandas
    df = pd.DataFrame(finetune_entries)
    df.to_csv(f"{config_name}_output.csv", index=False)

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
