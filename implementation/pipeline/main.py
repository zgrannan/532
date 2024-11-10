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
from generated_qa_processing import RemoveSimilarQuestionsAgent, AddSourceToQuestionAgent
from refine_question import RefineQuestionsAgent
from rag_answer import GetRAGAnswerAgent
from pydantic import BaseModel
from pathlib import Path

load_dotenv(find_dotenv())

class PipelineConfig(BaseModel):
    document_chunk_size: int = 5000
    document_chunk_overlap: int = 100 
    rag_chunk_size: int = 500 
    rag_chunk_overlap: int = 100
    batch_size: int = 10
    llm_model: str 
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5@f32"
    vector_store: Chroma
    embedding_function: OpenAIEmbeddings
    config_name: str

    class Config:
        arbitrary_types_allowed = True

def create_embedding_pipeline(config: PipelineConfig) -> Any:
    """Pipeline for embedding documents"""
    return (
        EnrichPdfFileAgent().with_cache(
            f"cache/{config.config_name}_enrich_pdf_cache.json", batch_size=1
        )
        .and_then(EmbedChunksAgent(
            config.embedding_function,
            config.vector_store,
            config.rag_chunk_size,
            config.rag_chunk_overlap
        ))
        .chunk(10)
    )

def create_qa_pipeline(config: PipelineConfig) -> Any:
    """Main Pipeline for generating Q/A pairs"""
    return (
        EnrichPdfFileAgent().with_cache(
            f"cache/{config.config_name}_enrich_pdf_cache.json", batch_size=1
        )
        .and_then(ChunkTextAgent(config.document_chunk_size, config.document_chunk_overlap))
        .chunk(20)  # We embed 10 text chunks at a time
        .and_then(UnchunkingAgent())  # Undo the previous chunking
        .fan_out(
            20,
            QuestionGenerator(config.llm_model, 10).with_cache(
                filename=f"cache/{config.config_name}_question_generator_cache.json",
                batch_size=10,
            ),
        )
        .and_then(RemoveSimilarQuestionsAgent(config.embedding_function, 0.9))
        .and_then(AddSourceToQuestionAgent(config.llm_model))
        .fan_out(
            20,
            EnrichAgent(
                GetAnswerAgent(config.llm_model).with_cache(
                    filename=f"cache/{config.config_name}_get_answer_cache.json", batch_size=10
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
            20,
            RefineQuestionsAgent(sampling_percent=0.7).with_cache(
                f"cache/{config.config_name}_refine_question_cache.json", batch_size=10
            ),
        )
        .and_then(RemoveSimilarQuestionsAgent(config.embedding_function, 0.9))
        .and_then(AddSourceToQuestionAgent(config.llm_model))
        .fan_out(20, GetRAGAnswerAgent(config.llm_model, config.vector_store))
    )

async def generate_finetune_entries_for_files_in_directory(
        config: PipelineConfig,
        directory: str,
) -> List[FinetuneEntry]:
    pdf_files = [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.endswith(".pdf")
    ]

    # Get pipelines
    embedding_pipeline = create_embedding_pipeline(config)
    qa_pipeline = create_qa_pipeline(config)

    # run pipelines
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting to chunk and embed into vectordb"
    )
    res = await slurp_iterator(embedding_pipeline.process_list(pdf_files))
    
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting to run qa pipeline"
    )
    return await slurp_iterator(qa_pipeline.process_list(pdf_files))

async def main():
    start_time = time.time()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default=datetime.now().strftime("%Y%m%d%H%M%S"),
        help="Name of the config to use (default: timestamp)",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="../data",
        help="Path to the directory containing PDF files (default: ../data)",
    )
    args = parser.parse_args()
    config_name = args.config_name
    file_path = args.file_path

    # INPUTS
    EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5@f32"  # on LM Studio
    LLM_MODEL = "meta-llama-3.1-8b-instruct-q6_k"

    # Hugging Face
    repo_id = "CPSC532/arxiv_qa_data"

    embedding_function= OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://localhost:1234/v1",
            api_key="test",
            check_embedding_ctx_length=False,
        )
    
    vector_store = Chroma(
        collection_name=config_name,
        embedding_function=embedding_function,
        persist_directory="./chroma_langchain_db",
    )

    pipeline_config = PipelineConfig(
                                    document_chunk_size = 5000,
                                    document_chunk_overlap= 100 ,
                                    rag_chunk_size = 500 ,
                                    rag_chunk_overlap = 100,
                                    batch_size = 10,
                                    llm_model = LLM_MODEL ,
                                    embedding_model = EMBEDDING_MODEL,
                                    vector_store = vector_store,
                                    embedding_function = embedding_function,
                                    config_name = config_name
                            )
    


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
