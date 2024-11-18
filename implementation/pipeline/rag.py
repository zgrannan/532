from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr

from helpers import get_embedding_func


EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5@f32"  # on LM Studio


def get_vector_store(config_name: str):
    embedding_function = get_embedding_func(EMBEDDING_MODEL)
    return Chroma(
        collection_name=config_name,
        embedding_function=embedding_function,
        persist_directory="./chroma_langchain_db",
    )
