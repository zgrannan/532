import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List, Union
from datasets import Dataset
from langchain_text_splitters import TokenTextSplitter
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)
        end_time = time.time()  # Record the end time
        execution_time = end_time - start_time  # Calculate the difference
        logger.info(f"{func.__name__} took {execution_time:.4f} seconds to execute")
        return result
    return wrapper

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
@timing_decorator
def get_json_response(
                    client: OpenAI, 
                    model: str,
                    messages: Optional[List[dict]] ,
                    response_format: BaseModel,
                    temperature: float = 0.0) -> BaseModel:
    
    response = client.beta.chat.completions.parse(
        messages = messages,
        model = model,
        temperature = temperature,
        response_format=response_format
    )

    return response.choices[0].message.parsed

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
@timing_decorator
def get_messages_response(
                    client: OpenAI, 
                    model: str,
                    messages: Optional[List[dict]] ,
                    temperature: float = 0.0) -> str:
    response = client.chat.completions.create(
        messages = messages,
        model = model,
        temperature = temperature,
    )

    return response.choices[0].message.content

def list_of_dicts_to_dict_of_lists(list_of_dicts: List[dict]) -> dict:
    # Initialize an empty dictionary to store lists
    dict_of_lists = {}

    # Iterate over each dictionary in the list
    for d in list_of_dicts:
        for key, value in d.items():
            # Add the key to the dictionary if it doesn't exist
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            # Append the value to the appropriate list
            dict_of_lists[key].append(value)

    return dict_of_lists

def upload_to_hf(
                data: Union[dict, Dataset],
                repo_id: str,
                config_name: str,
                api_key: str
                ) -> None:
    if isinstance(data, dict):
        data = Dataset.from_dict(data)
    data.push_to_hub(
        repo_id=repo_id,
        config_name=config_name,
        token=api_key
    )

def split_text(
            text: str,
            chunk_size: int = 5000,
            chunk_overlap: int = 100) -> List[str]:

    splitter = TokenTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def remove_duplicates_by_key(dict_list: List[dict], key="question"):
    # Create a new list without duplicates by tracking seen values
    seen = set()
    
    return [d for d in dict_list if d[key] not in seen and not seen.add(d[key])]

def remove_duplicates(strings_list: List[str]):
    seen = set()
    return [s for s in strings_list if s not in seen and not seen.add(s)]

import time

