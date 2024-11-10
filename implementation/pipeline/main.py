import os
import time
import random
import asyncio
import argparse
import pandas as pd
import numpy as np
from uuid import uuid4
from datetime import datetime
from typing import TypeVar, List
from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from agent import (
    EnrichAgent,
    UnchunkingAgent,
)
from entity_extraction import EntityExtractionAgent
from helpers import (
    list_of_dicts_to_dict_of_lists,
    upload_to_hf,
    slurp_iterator
)
from question_answer import (
    QuestionWithChunk,
    GetAnswerAgent,
    QuestionGenerator,
)
from pipeline_types import FinetuneEntry
from token_tracking import tracker
from utils import EnrichPdfFileAgent
from chunking import EmbedChunksAgent, ChunkTextAgent
from generated_qa_processing import (
    RemoveSimilarQuestionsAgent,
    GetRAGAnswerAgent,
    RefineQuestionsAgent,
)

load_dotenv(find_dotenv())

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


async def generate_finetune_entries_for_files_in_directory(
    directory: str,
    config_name: str,
) -> List[FinetuneEntry]:
    pdf_files = [
        os.path.join(directory, filename)
        for filename in os.listdir(directory)
        if filename.endswith(".pdf")
    ]

    for file in pdf_files:
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing file: {file}"
        )

    EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5@f32"  # on LM Studio
    chunk_size = 5000
    chunk_overlap = 100
    entity_batch_size = 10
    question_batch_size = 10
    # model = "meta-llama-3.1-8b-instruct-q6_k"
    model = "lmstudio-community/qwen2.5-7b-instruct"
    embeddings_func = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:1234/v1",
        api_key="test",
        check_embedding_ctx_length=False,  # https://github.com/langchain-ai/langchain/issues/21318
    )

    vector_store = Chroma(
        collection_name=config_name,  # Config name,
        embedding_function=embeddings_func,
        persist_directory="./chroma_langchain_db",
    )

    enrich_pdf_agent = EnrichPdfFileAgent().with_cache(
        f"cache/{config_name}_enrich_pdf_cache.json", batch_size=1
    )

    # Chunking and embedding into vectordb are using different chunking parameters than first stage
    granular_chunking_pipeline = (
        enrich_pdf_agent
        .and_then(EmbedChunksAgent(embeddings_func, vector_store, 500, 150))
        .chunk(10)
    )
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting to chunk and embed into vectordb"
    )
    res = await slurp_iterator(granular_chunking_pipeline.process_list(pdf_files))
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generating finetune entries"
    )

    pipeline = (
        enrich_pdf_agent
        .and_then(ChunkTextAgent(chunk_size, chunk_overlap))
        .chunk(10)  # We embed 10 text chunks at a time
        .and_then(UnchunkingAgent())  # Undo the previous chunking
        .fan_out(
            10,
            QuestionGenerator(model, question_batch_size).with_cache(
                filename=f"cache/{config_name}_question_generator_cache.json",
                batch_size=10,
            ),
        )
        .and_then(RemoveSimilarQuestionsAgent(embeddings_func, 0.9))
        .fan_out(
            10,
            EnrichAgent(
                GetAnswerAgent(model).with_cache(
                    filename=f"cache/{config_name}_get_answer_cache.json", batch_size=10
                ),
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
        .fan_out(
            10,
            RefineQuestionsAgent(sampling_percent=0.5).with_cache(
                f"cache/{config_name}_refine_question_cache.json", batch_size=10
            ),
        )
        .and_then(RemoveSimilarQuestionsAgent(embeddings_func, 0.8))
        .fan_out(10, GetRAGAnswerAgent(model, vector_store))
    )
    print(pipeline.to_dot())

    return await slurp_iterator(pipeline.process_list(pdf_files))

async def main():
    start_time = time.time()

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

    # asyncio.create_task(tracker.periodic_save("test_periodic_save", interval=5))

    # First stage getting initial Q/A Pairs
    finetune_entries = await generate_finetune_entries_for_files_in_directory(
        "../data", config_name
    )
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Generated {len(finetune_entries)} finetune entries"
    )
    finetune_entries_filtered = (
        []
    )  # In case of any errors, we can filter out the None entries
    for entry in finetune_entries:
        if entry is not None:
            try:
                finetune_entries_filtered.append(entry)
            except Exception as e:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error filtering entry: {e}"
                )
                continue
    df = pd.DataFrame(finetune_entries_filtered)
    # if outputs folder doesn't exist, create it
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    df.to_csv(f"outputs/{config_name}_output.csv", index=False)

    # Upload to Huggingface
    qa_pairs_dict = list_of_dicts_to_dict_of_lists(finetune_entries)
    upload_to_hf(
        data=qa_pairs_dict,
        repo_id=repo_id,
        api_key=os.getenv("HUGGINGFACE_API_KEY"),
        config_name=config_name,
    )

    end_time = time.time()
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total time: {end_time - start_time} seconds"
    )
    tracker.save_to_file(config_name)


if __name__ == "__main__":
    asyncio.run(main())
