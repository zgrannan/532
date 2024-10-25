import PyPDF2
import asyncio
from openai import OpenAI, AsyncOpenAI
import os
import time
from helpers import get_default_client
from utils.pdf import read_pdf
from typing import Any, AsyncGenerator, Coroutine, Generator, List, TypeVar, TypedDict
from entity_extraction import (
    get_entities,
)
from question_answer import (
    generate_questions,
    get_answer,
)
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


async def generate_questions_for_chunk(
    client: AsyncOpenAI, chunk: str, model: str, source: str, source_type: str
) -> AsyncGenerator[str, None]:
    entities = await get_entities(client=client, text=chunk, max_entities=10)
    print(f"Number of entities extracted: {len(entities)}")
    return generate_questions(
        {
            "client": client,
            "model": model,
            "chunk": chunk,
            "source": source,
            "source_type": source_type,
        },
        entities,
    )


class QAPair(TypedDict):
    question: str
    answer: str


async def remove_duplicate_questions(
    questions: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    seen = set()
    async for question in questions:
        if question not in seen:
            seen.add(question)
            yield question


T = TypeVar("T")


async def slurp_generator(generator: AsyncGenerator[T, None]) -> List[T]:
    return [item async for item in generator]


async def generate_qa_pairs_for_chunk(
    client: AsyncOpenAI, chunk: str, model: str, source: str, source_type: str
) -> AsyncGenerator[QAPair, None]:
    questions = await generate_questions_for_chunk(
        client=client, chunk=chunk, model=model, source=source, source_type=source_type
    )

    questions = remove_duplicate_questions(questions)

    # Question Answering -> Should check for hallucination
    # We drain the questions generator here to enable the answers to be obtained
    # in parallel
    questions_list = await slurp_generator(questions)

    answers = await asyncio.gather(
        *[get_answer(client, chunk, question) for question in questions_list]
    )

    for question, answer in zip(questions_list, answers):
        yield {
            "question": question,
            "answer": answer,
        }


class FinetuneEntry(TypedDict):
    question: str
    answer: str
    source: str


async def generate_qa_pairs(text: str, source: str, source_type: str) -> List[QAPair]:
    model = "meta-llama-3.1-8b-instruct-q6_k"
    client = get_async_client()
    chunks = split_text(text, chunk_size=5000, chunk_overlap=100)
    print(f"{len(chunks)} chunks created")
    chunked_results = await asyncio.gather(
        *[
            slurp_generator(
                generate_qa_pairs_for_chunk(client, chunk, model, source, source_type)
            )
            for chunk in chunks
        ]
    )
    return [qa_pair for qa_pair_list in chunked_results for qa_pair in qa_pair_list]


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
    config_name = "test_dataset_2024OCT24"

    finetune_entries = await generate_finetune_entries_for_files_in_directory(
        "../data"
    )

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
