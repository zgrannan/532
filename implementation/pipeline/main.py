from openai import OpenAI
import os 
import sys 

sys.path.append('../')

from utils.pdf import read_pdf
from pydantic import BaseModel, Field
from typing import List

from pipeline.entity_extraction import ENTITY_EXTRACTION_SYSTEM_PROMPT, EntityExtractionModel
from pipeline.question_answer import QUESTION_EXTRACTION_SYSTEM_PROMPT, QuestionAnswerModel
from pipeline.helpers import get_json_response, get_messages_response, split_text, list_of_dicts_to_dict_of_lists, upload_to_hf
import pandas as pd 
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())


if __name__ == "__main__":
    # Inputs 
    lm_studio_base_url = "http://localhost:1234/v1"
    text = read_pdf("../data/Taming Transformers for High Resolution Image Synthesis.pdf") 
    source = "Taming Transformers for High Resolution Image Synthesis"
    source_type = "paper"
    model = "meta-llama-3.1-8b-instruct-q6_k"

    # Hugging Face
    repo_id="owenren/532_finetune_qa_datasets"
    config_name="test_dataset_2024OCT20"


    lm_studio_client = OpenAI(base_url=lm_studio_base_url, api_key="lm_studio")

    chunks = split_text(text, chunk_size=5000, chunk_overlap=100)   
    qa_pairs = []
    
    for chunk in chunks:
        # Entity Extraction 
        entities = entities = get_json_response(
                                        client=lm_studio_client,
                                        model=model,
                                        messages=[
                                            
                                            {
                                                "role": "system",
                                                "content": ENTITY_EXTRACTION_SYSTEM_PROMPT.format(text=chunk)
                                            },

                                        ],
                                        response_format=EntityExtractionModel,
                                    )

        # Question Extraction
        question_list = []
        for i in range(0, len(entities.entities), 10): # iterate in batches of 10's
            qa_prompt = QUESTION_EXTRACTION_SYSTEM_PROMPT.format(text=chunk, 
                                                    entities=", ".join(entities.entities[i: i + 10]),
                                                    source=source,
                                                    source_type=source_type # this can get automated
                                                    )
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

        # Question Answering


        for question in question_list:
            answer = get_messages_response(
                                        client=lm_studio_client,
                                        model=model,
                                        messages=[
                                            
                                            {
                                                "role": "system",
                                                "content":qa_prompt
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
    df.to_csv("output.csv", index=False)

    # Upload to Huggingface
    qa_pairs_dict = list_of_dicts_to_dict_of_lists(qa_pairs)
    upload_to_hf(
        data = qa_pairs_dict,
        repo_id=repo_id,
        api_key=os.getenv('HUGGINGFACE_API_KEY'),
        config_name=config_name
    )