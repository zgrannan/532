from openai import OpenAI
import os
import sys

sys.path.append('../')

import time
from helpers import get_default_client
from utils.pdf import read_pdf
from pydantic import BaseModel, Field
from typing import List
import requests
from pipeline.entity_extraction import ENTITY_EXTRACTION_SYSTEM_PROMPT, EntityExtractionModel, get_entities
from pipeline.question_answer import QUESTION_EXTRACTION_SYSTEM_PROMPT, ANSWER_EXTRACTION_SYSTEM_PROMPT, QuestionAnswerModel, generate_questions, get_answer
from pipeline.helpers import get_json_response, get_messages_response, get_model, split_text, list_of_dicts_to_dict_of_lists, upload_to_hf, remove_duplicates
from pipeline.judge import judge
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import logging
import sys
load_dotenv(find_dotenv())

if __name__ == "__main__":
    start_time = time.time()

    # Inputs
    ollama_base_url_embeddings = "http://localhost:11434/api/embeddings"
    embedding_model = "nomic-embed-text"
    text = read_pdf("../data/Sohn_Visual_Prompt_Tuning_for_Generative_Transfer_Learning_CVPR_2023_paper.pdf")
    source = "Visual Prompt Tuning for Generative Transfer Learning"
    source_type = "paper"
    model = "meta-llama-3.1-8b-instruct-q6_k"

    # Hugging Face
    repo_id="CPSC532/arxiv_qa_data"
    config_name="test_dataset_2024OCT23"

    lm_studio_client = get_default_client()

    chunks = split_text(text, chunk_size=5000, chunk_overlap=100)
    print(f"{len(chunks)} chunks created")

    qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}")
        entities = get_entities(
            client=lm_studio_client,
            text=chunk,
            max_entities=10
        )
        print(f"Number of entites extracted: {len(entities)}")
        question_list = generate_questions(
            {
                "client": lm_studio_client,
                "model": model,
                "chunk": chunk,
                "source": source,
                "source_type": source_type
            },
            entities
        )

        # To do:
        # Perform some data cleaning and remove similar questions

        # Convert generator to list
        question_list = list(question_list)
        # Remove duplicates
        print(f"{len(question_list)} questions extracted")
        question_list = remove_duplicates(question_list)
        print(f"{len(question_list)} after removing duplicates")

        # Question Answering -> Should check for hallucination
        for question in question_list:
            answer = get_answer(
                client=lm_studio_client,
                chunk=chunk,
                question=question
            )

            qa_pairs.append({
                "question": question,
                "answer": answer,
                "source": source
            })


    # Save to csv using Pandas
    df = pd.DataFrame(qa_pairs)
    df.to_csv(f"{config_name}_output.csv", index=False)

    # Upload to Huggingface
    qa_pairs_dict = list_of_dicts_to_dict_of_lists(qa_pairs)
    upload_to_hf(
        data = qa_pairs_dict,
        repo_id=repo_id,
        api_key=os.getenv('HUGGINGFACE_API_KEY'),
        config_name=config_name
    )

    end_time = time.time()
    print(f"Total time: {end_time - start_time} seconds")
