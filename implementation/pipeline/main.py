import json
import os
import time
import random
import asyncio
import argparse
import pandas as pd
import numpy as np
from uuid import uuid4
from datetime import datetime
from typing import TypeVar, List, Any
from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from agent import (
    EnrichAgent,
    UnchunkingAgent,
)
from entity_extraction import EntityExtractionAgent
from helpers import list_of_dicts_to_dict_of_lists, upload_to_hf, slurp_iterator
from agent import Agent, Cache, StatelessAgent
from helpers import get_embedding_func
from agent import Pipeline
from helpers import LM_STUDIO_BASE_URL
from agent import LLMClientSettings
from helpers import get_pipeline_config_name
from pipeline_types import EnrichedPdfChunkWithQuestion
from rag import EMBEDDING_MODEL, get_vector_store
from pipeline_types import EnrichedPdfFile
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
    AddSourceToQuestionAgent,
)
from refine_question import RefineQuestionsAgent
from rag_answer import GetRAGAnswerAgent
from pydantic import BaseModel, SecretStr
from pathlib import Path
from typing import Literal
import logging

load_dotenv(find_dotenv())


class ConfigSettings(BaseModel):
    include_source: bool = True
    max_documents: int = 50  # Maximum number of documents to process
    llm_parallelism: int = 20  # Number of LLM calls to run in parallel
    max_base_questions_per_chunk: int = (
        10  # Maximum number of initial questions to generate per document chunk before expansion
    )
    document_chunk_size: int = 5000
    document_chunk_overlap: int = 100
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 100
    batch_size: int = 10
    test_ratio: float  # Number of entries that should be used for testing only
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5@f32"
    config_name: str
    llm_settings: LLMClientSettings


class PipelineConfig(ConfigSettings):
    vector_store: Chroma
    embedding_function: OpenAIEmbeddings

    class Config:
        arbitrary_types_allowed = True


def enrich_pdf_file_agent(
    config: PipelineConfig,
) -> StatelessAgent[str, EnrichedPdfFile]:
    return EnrichPdfFileAgent().with_cache(
        f"cache/{config.config_name}_enrich_pdf_cache"
    )


def create_embedding_pipeline(config: PipelineConfig) -> Any:
    """Pipeline for embedding documents"""
    return (
        enrich_pdf_file_agent(config)
        .and_then_sl(
            EmbedChunksAgent(
                config.embedding_function,
                config.vector_store,
                config.rag_chunk_size,
                config.rag_chunk_overlap,
            )
        )
        .with_cache(f"cache/{config.config_name}_embed_chunks_cache.json")
    )


def create_qa_pipeline(config: PipelineConfig) -> Pipeline[str, FinetuneEntry]:
    """Main Pipeline for generating Q/A pairs"""

    # For some reason extracting this to a variable is necessary to satisfy mypy
    add_source_stage1: Agent[
        EnrichedPdfChunkWithQuestion, EnrichedPdfChunkWithQuestion
    ] = AddSourceToQuestionAgent(config.llm_settings)

    return (
        enrich_pdf_file_agent(config)
        .and_then(
            ChunkTextAgent(config.document_chunk_size, config.document_chunk_overlap)
        )
        .fan_out(
            config.llm_parallelism,
            QuestionGenerator(
                config.llm_settings,
                max_questions=config.max_base_questions_per_chunk,
            ).with_cache(
                filename=f"cache/{config.config_name}_question_generator_cache"
            ),
        )
        .and_then(RemoveSimilarQuestionsAgent(config.embedding_function, 0.9))
        .and_then_if(
            config.include_source,
            add_source_stage1.parallelize(config.llm_parallelism),
        )
        .fan_out(
            config.llm_parallelism,
            EnrichAgent(
                GetAnswerAgent(config.llm_settings).with_cache(
                    filename=f"cache/{config.config_name}_get_answer_cache",
                ),
                lambda e: QuestionWithChunk(question=e["question"], chunk=e["chunk"]),
                lambda e, answer: FinetuneEntry(
                    filename=e["filename"],
                    source=e["source"],
                    source_type=e["source_type"],
                    chunk=e["chunk"],
                    question=e["question"],
                    answer=answer,
                    pass_through=False,
                ),
            ),
        )
        .fan_out(
            config.llm_parallelism,
            RefineQuestionsAgent(
                settings=config.llm_settings, sampling_percent=0.5
            ).with_cache(f"cache/{config.config_name}_refine_question_cache"),
        )
        .and_then(RemoveSimilarQuestionsAgent(config.embedding_function, 0.9))
        .and_then_if(
            config.include_source,
            AddSourceToQuestionAgent(config.llm_settings).parallelize(
                config.llm_parallelism
            ),
        )
        .fan_out(
            config.llm_parallelism,
            GetRAGAnswerAgent(config.llm_settings, config.vector_store),
        )
    )


async def generate_finetune_entries_for_files_in_directory(
    config: PipelineConfig,
    directory: str,
) -> List[FinetuneEntry]:
    pdf_files = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")
    ]
    if len(pdf_files) > config.max_documents:
        logging.warning(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {len(pdf_files)} files, but only processing {config.max_documents}."
        )
        pdf_files = pdf_files[: config.max_documents]

    # Get pipelines
    embedding_pipeline = create_embedding_pipeline(config)
    qa_pipeline = create_qa_pipeline(config)

    # run pipelines
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting to chunk and embed into vectordb"
    )
    await slurp_iterator(embedding_pipeline.process_list(pdf_files))

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting to run qa pipeline"
    )

    cache = Cache(f"cache/{config.config_name}_qa_cache")

    results = []
    for i, file in enumerate(pdf_files):
        logging.info(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing {file} ({i + 1}/{len(pdf_files)})"
        )
        cached_output = cache.get_cached_output(file)
        if cached_output is not None:
            logging.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found cached output for {file}"
            )
            results.extend(cached_output)
        else:
            results.extend(await slurp_iterator(qa_pipeline.process_once(file)))
            cache.set_cached_output(file, results)

    return results


REPO_ID = "CPSC532/arxiv_qa_data"
DEFAULT_LLM_MODEL = "meta-llama-3.1-8b-instruct-q6_k"
DEFAULT_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5@f32"


async def run_pipeline(pipeline_config: PipelineConfig, file_path: str, upload: bool):
    start_time = time.time()
    huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
    if upload and huggingface_api_key is None:
        raise ValueError("HUGGINGFACE_API_KEY is not set")

    # First stage getting initial Q/A Pairs
    finetune_entries = await generate_finetune_entries_for_files_in_directory(
        pipeline_config, file_path
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

    # Split into train and test sets
    test_size = int(len(finetune_entries_filtered) * pipeline_config.test_ratio)
    test_indices = random.sample(range(len(finetune_entries_filtered)), test_size)

    train_entries = []
    test_entries = []

    for idx, entry in enumerate(finetune_entries_filtered):
        if idx in test_indices:
            test_entries.append(entry)
        else:
            train_entries.append(entry)

    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Split into {len(train_entries)} train and {len(test_entries)} test entries"
    )

    test_df = pd.DataFrame(test_entries)
    train_df = pd.DataFrame(train_entries)

    # if outputs folder doesn't exist, create it
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    test_df.to_csv(
        f"outputs/{pipeline_config.config_name}_test_output.csv", index=False
    )
    train_df.to_csv(
        f"outputs/{pipeline_config.config_name}_train_output.csv", index=False
    )

    if upload:
        # Upload to Huggingface
        qa_pairs_dict = list_of_dicts_to_dict_of_lists(train_entries)

        if huggingface_api_key is None:
            raise ValueError("HUGGINGFACE_API_KEY is not set")

        upload_to_hf(
            data=qa_pairs_dict,
            repo_id=REPO_ID,
            api_key=huggingface_api_key,
            config_name=pipeline_config.config_name,
        )

    end_time = time.time()

    total_time_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total time: {end_time - start_time} seconds"
    print(total_time_message)
    with open(f"outputs/{pipeline_config.config_name}_time_log.txt", "w") as time_log_file:
        time_log_file.write(total_time_message)
    tracker.save_to_file(pipeline_config.config_name)


async def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the config file to use",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=datetime.now().strftime("%Y%m%d%H%M%S"),
        help="Name of the config to use (default: timestamp)",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="../final_data",
        help="Path to the directory containing PDF files (default: ../data)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_false",
        dest="upload",
        help="Do not upload to Hugging Face",
    )
    args = parser.parse_args()
    file_path = args.file_path
    if args.config_file is not None:
        with open(args.config_file, "r") as f:
            config = json.load(f)
            pipeline_template = config["template"]["pipeline_config"]
            for include_source in config["matrix"]["pipeline_config.sources"]:
                for llm_config in config["matrix"]["pipeline_config.llm"]:
                    config_name = get_pipeline_config_name(include_source, llm_config["model"])
                    api_key = os.getenv(llm_config["api_key_var"])
                    if api_key is None:
                        raise ValueError(
                            f"API key for {llm_config['model']} is not set"
                        )
                    pipeline_config = PipelineConfig(
                        llm_settings=LLMClientSettings(
                            model=llm_config["model"],
                            base_url=llm_config["base_url"],
                            api_key=api_key,
                            model_provider=llm_config["model_provider"],
                        ),
                        include_source=include_source,
                        test_ratio=pipeline_template["test_ratio"],
                        document_chunk_size=pipeline_template["document_chunk_size"],
                        document_chunk_overlap=pipeline_template[
                            "document_chunk_overlap"
                        ],
                        rag_chunk_size=pipeline_template["rag_chunk_size"],
                        rag_chunk_overlap=pipeline_template["rag_chunk_overlap"],
                        batch_size=pipeline_template["batch_size"],
                        embedding_model=pipeline_template["embedding_model"],
                        config_name=config_name,
                        vector_store=get_vector_store(config_name),
                        embedding_function=get_embedding_func(
                            pipeline_template["embedding_model"]
                        )
                    )
                    await run_pipeline(pipeline_config, file_path, args.upload)
    else:
        client_settings = LLMClientSettings(
            model=DEFAULT_LLM_MODEL,
            base_url=os.getenv("LLM_CLIENT_BASE_URL", LM_STUDIO_BASE_URL),
            api_key=os.getenv("LLM_CLIENT_API_KEY", "apikey"),
            model_provider="LMStudio",
        )
        pipeline_config = PipelineConfig(
            llm_settings=client_settings,
            test_ratio=args.test_ratio,
            document_chunk_size=5000,
            document_chunk_overlap=100,
            rag_chunk_size=500,
            rag_chunk_overlap=100,
            batch_size=10,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            config_name=args.config_name,
            embedding_function=get_embedding_func(DEFAULT_EMBEDDING_MODEL),
            vector_store=get_vector_store(args.config_name),
        )
        await run_pipeline(pipeline_config, file_path, args.upload)


if __name__ == "__main__":
    asyncio.run(main())
