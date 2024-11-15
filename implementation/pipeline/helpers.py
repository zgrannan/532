from asyncio import Future
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from openai import NotGiven, OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, SecretStr
from typing import Any, AsyncGenerator, Dict, List, TypeVar, Union, cast, AsyncIterator
from datasets import Dataset  # type: ignore
from langchain_text_splitters import TokenTextSplitter
import os
import time
from langchain_openai import OpenAIEmbeddings
from token_tracking import track_llm_usage

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
MAX_TOKENS = 16384
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
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
@track_llm_usage
def get_json_response(
    client: OpenAI,
    model: str,
    messages: List[ChatCompletionMessageParam],
    response_format: type[BaseModel],
    temperature: float = 0.0,
    agent_name: str = "",  # for track_llm_usage
    max_tokens: int = MAX_TOKENS,
    **kwargs: dict,
) -> BaseModel:

    response = client.beta.chat.completions.parse(
        messages=messages,
        model=model,
        temperature=temperature,
        response_format=response_format,
        max_tokens=max_tokens,
        **kwargs,
    )

    return response


T = TypeVar("T", bound=BaseModel)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
@track_llm_usage
async def get_json_response_async(
    client: AsyncOpenAI,
    model: str,
    messages: List[ChatCompletionMessageParam],
    response_format: type[T],
    temperature: float = 0.0,
    agent_name: str = "",  # for track_llm_usage
    max_tokens: int = MAX_TOKENS,
    **kwargs: dict,
) -> T:
    response = await client.beta.chat.completions.parse(
        messages=messages,
        model=model,
        temperature=temperature,
        response_format=response_format,
        max_tokens=max_tokens,
        **kwargs,
    )

    return response


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
@track_llm_usage
async def get_messages_response_async(
    client: AsyncOpenAI,
    model: str,
    messages: List[ChatCompletionMessageParam],
    temperature: float = 0.0,
    agent_name: str = "",  # for track_llm_usage
    max_tokens: int = MAX_TOKENS,
    **kwargs: dict,
) -> str:
    response = await client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    return response


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    before_sleep=before_sleep_log(logger, logging.ERROR),
)
async def get_embeddings_resp_async(
    client: AsyncOpenAI,
    model: str,
    text: str,
) -> List[float]:
    response = await client.embeddings.create(model=model, input=text)

    result = response.data[0].embedding
    return result


async def get_simple_response(
    client: AsyncOpenAI, model: str, message: str, temperature: float = 0.0
) -> str:
    return await get_messages_response_async(
        client, model, [{"role": "user", "content": message}], temperature
    )


def list_of_dicts_to_dict_of_lists(list_of_dicts: List[dict]) -> Dict[str, List[Any]]:
    # Initialize an empty dictionary to store lists
    dict_of_lists: Dict[str, List[Any]] = {}

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
    data: Union[dict, Dataset], repo_id: str, config_name: str, api_key: str
) -> None:
    if isinstance(data, dict):
        data = Dataset.from_dict(data)
    data.push_to_hub(repo_id=repo_id, config_name=config_name, token=api_key)


def split_text(
    text: str, chunk_size: int = 5000, chunk_overlap: int = 100
) -> List[str]:

    splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def remove_duplicates(strings_list: List[str]) -> List[str]:
    return list(set(strings_list))


def get_openai_client() -> OpenAI:
    return OpenAI()


def get_default_client() -> OpenAI:
    base_url = os.getenv("LLM_CLIENT_BASE_URL", LM_STUDIO_BASE_URL)
    api_key = os.getenv("LLM_CLIENT_API_KEY", "lm_studio")
    return OpenAI(base_url=base_url, api_key=api_key)


def get_async_client() -> AsyncOpenAI:
    base_url = os.getenv("LLM_CLIENT_BASE_URL", LM_STUDIO_BASE_URL)
    api_key = os.getenv("LLM_CLIENT_API_KEY", "lm_studio")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def get_model(default_model: str) -> str:
    return os.getenv("LLM_CLIENT_OVERRIDE_MODEL", default_model)


OnceT = TypeVar("OnceT")


async def once(element: OnceT) -> AsyncGenerator[OnceT, None]:
    yield element


def get_embedding_func(embedding_model) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=embedding_model,
        base_url=os.getenv("LLM_CLIENT_BASE_URL", LM_STUDIO_BASE_URL),
        api_key=cast(SecretStr, os.getenv("LLM_CLIENT_API_KEY", "lm_studio")),
        check_embedding_ctx_length=False,  # https://github.com/langchain-ai/langchain/issues/21318
    )


SlurpT = TypeVar("SlurpT")


async def slurp_iterator(generator: AsyncIterator[SlurpT]) -> List[SlurpT]:
    buffer = []
    try:
        async for item in generator:
            buffer.append(item)
        return buffer
    except Exception as e:
        raise e  # Could write buffer to file here
