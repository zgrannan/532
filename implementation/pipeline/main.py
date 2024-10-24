from openai import OpenAI
import os 
import sys 

sys.path.append('../')

import time 
from utils.pdf import read_pdf
from pydantic import BaseModel, Field
from typing import List
import requests
from pipeline.entity_extraction import ENTITY_EXTRACTION_SYSTEM_PROMPT, EntityExtractionModel
from pipeline.question_answer import QUESTION_EXTRACTION_SYSTEM_PROMPT, ANSWER_EXTRACTION_SYSTEM_PROMPT, QuestionAnswerModel
from pipeline.helpers import get_json_response, get_messages_response, split_text, list_of_dicts_to_dict_of_lists, upload_to_hf, remove_duplicates
import pandas as pd 
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

if __name__ == "__main__":
    start_time = time.time()

    # Inputs 
    ollama_base_url_embeddings = "http://localhost:11434/api/embeddings"
    embedding_model = "nomic-embed-text"
    lm_studio_base_url = "http://localhost:1234/v1"
    text = read_pdf("../data/Sohn_Visual_Prompt_Tuning_for_Generative_Transfer_Learning_CVPR_2023_paper.pdf") 
    source = "Visual Prompt Tuning for Generative Transfer Learning"
    source_type = "paper"
    model = "meta-llama-3.1-8b-instruct-q6_k"

    # Hugging Face
    repo_id="CPSC532/arxiv_qa_data"
    config_name="test_dataset_2024OCT23"

    lm_studio_client = OpenAI(base_url=lm_studio_base_url, api_key="lm_studio")

    chunks = split_text(text, chunk_size=5000, chunk_overlap=100)   
    print(f"{len(chunks)} chunks created")
    
    qa_pairs = []
    for chunk in chunks:
        # Entity Extraction 
        entity_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(text=chunk)
        print(f"length of entity prompt: {len(entity_prompt)}")
        entities = entities = get_json_response(
                                        client=lm_studio_client,
                                        model=model,
                                        messages=[
                                            
                                            {
                                                "role": "system",
                                                "content": entity_prompt
                                            },

                                        ],
                                        response_format=EntityExtractionModel,
                                    )

        # Question Extraction
        print(f"Number of entites extracted: {len(entities.entities)}")
        question_list = []
        for i in range(0, len(entities.entities), 10): # iterate in batches of 10's
            qa_prompt = QUESTION_EXTRACTION_SYSTEM_PROMPT.format(text=chunk, 
                                                    entities=", ".join(entities.entities[i: i + 10]),
                                                    source=source,
                                                    source_type=source_type # this can get automated
                                                    )
            print(f"Length of qa_prompt: {len(qa_prompt)}")
            questions = get_json_response(
                                            client=lm_studio_client,
                                            # model="llama-3.2-3b-instruct-q8_0"
                                            model=model,
                                            messages=[
                                                
                                                {
                                                    "role": "system",
                                                    "content":qa_prompt
                                                },

                                            ],
                                            response_format=QuestionAnswerModel,
                                        )
            question_list.extend(questions.questions)

        # To do: 
        # Perform some data cleaning and remove similar questions 

        # Remove duplicates
        print(f"{len(question_list)} questions extracted")
        question_list = remove_duplicates(question_list)
        print(f"{len(question_list)} after removing duplicates")

        # Question Answering -> Should check for hallucination
        for question in question_list:
            answer = get_messages_response(
                                        client=lm_studio_client,
                                        model=model,
                                        messages=[
                                            
                                            {
                                                "role": "system",
                                                "content":ANSWER_EXTRACTION_SYSTEM_PROMPT.format(text=chunk)
                                            },

                                            {
                                                "role": "user",
                                                "content": question
                                            },

                                        ],
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