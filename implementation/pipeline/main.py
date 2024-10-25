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


async def remove_duplicate_questions(questions: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    seen = set()
    async for question in questions:
        if question not in seen:
            seen.add(question)
            yield question


T = TypeVar('T')

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

    answers = await asyncio.gather(*[get_answer(client, chunk, question) for question in questions_list])

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
        *[slurp_generator(generate_qa_pairs_for_chunk(client, chunk, model, source, source_type)) for chunk in chunks]
    )
    return [qa_pair for qa_pair_list in chunked_results for qa_pair in qa_pair_list]

async def generate_finetune_entries(filename: str, source: str, source_type: str) -> List[FinetuneEntry]:
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


async def main():
    start_time = time.time()

    # Inputs
    ollama_base_url_embeddings = "http://localhost:11434/api/embeddings"
    embedding_model = "nomic-embed-text"
    filename = "../data/Sohn_Visual_Prompt_Tuning_for_Generative_Transfer_Learning_CVPR_2023_paper.pdf"
    source = "Visual Prompt Tuning for Generative Transfer Learning"
    source_type = "paper"

    # Hugging Face
    repo_id = "CPSC532/arxiv_qa_data"
    config_name = "test_dataset_2024OCT23"

    qa_pairs = await generate_finetune_entries(filename, source, source_type)

    # Save to csv using Pandas
    df = pd.DataFrame(qa_pairs)
    df.to_csv(f"{config_name}_output.csv", index=False)

    # Upload to Huggingface
    qa_pairs_dict = list_of_dicts_to_dict_of_lists(qa_pairs)
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
