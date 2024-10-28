# Second stage refining Q/A Pairs and getting answers from chunk + RAG 
import PyPDF2
import asyncio
from openai import OpenAI, AsyncOpenAI
import os
import time
from helpers import get_default_client
from agent import Agent, OpenAIAgent, ComposedAgent
from entity_extraction import EntityExtractionAgent
from helpers import once
from question_answer import QAPair
from question_answer import GetAnswerAgent
from question_answer import QuestionGenerator
from utils.pdf import read_pdf
from typing import Any, AsyncGenerator, AsyncIterator, Coroutine, Generator, List, TypeVar, TypedDict, Tuple
from helpers import (
    get_async_client,
    split_text,
    list_of_dicts_to_dict_of_lists,
    upload_to_hf,
    remove_duplicates,
    get_messages_response_async,
    get_json_response_async,
)
import pandas as pd
from dotenv import find_dotenv, load_dotenv
import argparse
from datetime import datetime
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import chromadb
from pydantic import BaseModel 
import pandas as pd 
from main import ChunkTextAgent, slurp_iterator, FinetuneEntry, RemoveDuplicateQuestionsAgent
from uuid import uuid4
load_dotenv(find_dotenv())
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity

# DEFAULT_REFINE_QUESTIONS_MODEL = "llama-3.2-3b-instruct"
DEFAULT_REFINE_QUESTIONS_MODEL = "meta-llama-3.1-8b-instruct"

class RefinedQuestionsModel(BaseModel):
    questions: List[str]

REFINED_QUESTIONS_PROMPT = """
    You are provided with a question and an answer.
    Your job is to generate a set of new questions that can be answered with the given answer but is diverse and approaches 
    the original question from different perspectives.

    Ensure that the generated questions are clear, purposeful, specific, and invoke critical thinking
    Question:
    {question}

    Answer:
    {answer}

    Return a list of new questions in JSON format.
"""
class RefineQuestionsAgent(OpenAIAgent[FinetuneEntry, RefinedQuestionsModel]):

    def __init__(self):
        super().__init__(DEFAULT_REFINE_QUESTIONS_MODEL)

    async def _process(self,
                       inputs: AsyncIterator[FinetuneEntry]) -> AsyncIterator[str]:
        
        async for input in inputs:
            print(f"Generating refined question for Question: {input['question']}")
            resp = await get_json_response_async(
                client=self.client,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": REFINED_QUESTIONS_PROMPT.format(
                            question = input['question'], 
                            answer = input['answer'],
                        ),
                    },
                ],
                response_format=RefinedQuestionsModel
        )
            for question in resp.questions:
                print("Refined Question: ", question)
                yield {
                    **input,
                    "question": question,
                    "original_question": input['question'],
                }
               
REFINED_RAG_ANSWER_PROMPT = """
    You are tasked with answering questions based on a provided text.
    You are provided with a question and an initial answer.
    You are also provided with some supporting documentation to help create a new response

    Your goal is to generate high-quality, detailed answers by following these instructions:
    If the answer is not found in the text, respond with "NO ANSWER FOUND"
    
    # Instructions:
    1. Reference the Text: Answer directly using relevant details from the text. Avoid introducing unsupported claims.
    2. Comprehensive Response: Address all parts of the question thoroughly, covering multiple aspects if needed.
    3. Detail-Oriented: Highlight key elements like techniques, processes, models, or challenges, expanding on them for clarity.
    4. Organized Structure: Use clear paragraphs or points for complex answers.
    5. Clarity and Examples: Ensure the answer is precise and easy to follow. Include examples or quotes from the text when applicable.
    6. Include Sources: Clearly reference the source information at the end of the answer.
    7. Include only the answer in your response

    Question:
    {question}

    Initial Answer:
    {answer}

    Supporting Documentation:
    {docs}

"""

class RAGAgent(OpenAIAgent[FinetuneEntry, str]):

    def __init__(self, vector_store: Chroma, k:int=3):
        super().__init__(model=DEFAULT_REFINE_QUESTIONS_MODEL, embedding_model="text-embedding-nomic-embed-text-v1.5@f32")
        self.vector_store = vector_store
        self.k = k
    async def _process(self,
                       inputs: AsyncIterator[FinetuneEntry]) -> AsyncIterator[str]:
        # Test rag 

        # Get docs from vector store
        async for input in inputs:
            print("similarity search")
            docs = await self.vector_store.asimilarity_search(
                                                            query=input['question'],
                                                            k=self.k,
                                                            filter={"filename": input['source']}
                                                            )
            docs_str = "\n".join([r.page_content for r in docs])
            print(f"Generating answer for Question: {input['question']}")
            resp = await get_messages_response_async(
                client=self.client,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": REFINED_RAG_ANSWER_PROMPT.format(question=input['question'], answer=input['answer'], docs=docs_str),
                    },
                ],
            )
            yield {
                **input,
                "original_answer": input['answer'],
                "answer": resp
            }

async def chunk_and_embed_files(
                            filepath: str,
                            embeddings_func: OpenAIEmbeddings,
                            vector_store: Chroma,
                            chunk_size: int = 1000,
                            chunk_overlap: int = 50,
                            ):

    text = read_pdf(filepath)
    chunk_agent = (
        ChunkTextAgent(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        .chunk(10)
    )

    chunks = (await slurp_iterator(chunk_agent.process(once(text))))[0]
    # embed chunks and upload into vectorDB
    docs = [Document(page_content=doc[1],
                     metadata={"filename": os.path.splitext(os.path.basename(filepath))[0]}) 
                     for doc in chunks]
    uuids = [str(uuid4()) for _ in range(len(docs))]
    print(f"Number of chunks uploaded: {len(docs)}")

    return await vector_store.aadd_documents(docs, ids=uuids)
 

async def generate_refined_questions(inputs: FinetuneEntry) -> List[FinetuneEntry]:
    refine_question_agent = (
        RefineQuestionsAgent()

    )
    res = await slurp_iterator(refine_question_agent.process(once(inputs)))

    # rag_agent = RAGAgent()
    return [r for r in res]

async def generate_rag_answers(inputs: FinetuneEntry,
                               vector_store: Chroma) -> List[FinetuneEntry]:
    rag_agent = RAGAgent(vector_store)

    res = await slurp_iterator(rag_agent.process(once(inputs)))
    return res

def remove_similar_questions(inputs: List[FinetuneEntry], embeddings_func: OpenAIEmbeddings, threshold=0.80) -> List[FinetuneEntry]:
    embeddings = [embeddings_func.embed_query(input['question']) for input in inputs]
    similarity_matrix = cosine_similarity(embeddings)
    
    # Set diagonal to 0 to avoid self-matches
    np.fill_diagonal(similarity_matrix, 0)
    
    # Find pairs above threshold
    similar_pairs = []
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                similar_pairs.append((i, j))
    
    indices_to_remove = set()
    # For each similar pair, remove the second question
    for pair in similar_pairs:
        indices_to_remove.add(pair[1])

    filtered_data = [item for i, item in enumerate(inputs) if i not in indices_to_remove]
    return filtered_data

async def main():
    start_time = time.time()

    # Input will be output from first stage main.py() 

    MODEL_3B = "llama-3.2-3b-instruct" 
    EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5@f32" # on LM Studio
    DIRECTORY = "../data"
    CHUNK_SIZE=500
    CHUNK_OVERLAP=100
    RUN_NAME = "agent_instruct_2024OCT27_1"
    embeddings_func = OpenAIEmbeddings(
                                        model=EMBEDDING_MODEL,
                                        base_url="http://localhost:1234/v1",
                                        api_key="test",
                                        check_embedding_ctx_length=False # https://github.com/langchain-ai/langchain/issues/21318
                                    )
    client = chromadb.PersistentClient(path="./chroma_langchain_db"  )
    if RUN_NAME not in [x.name for x in client.list_collections]:
        vector_store = Chroma(
            collection_name=RUN_NAME, # Config name,
            embedding_function=embeddings_func,
            persist_directory="./chroma_langchain_db"  
        )

        pdf_files = [
            os.path.join(DIRECTORY, filename)
            for filename in os.listdir(DIRECTORY)
            if filename.endswith(".pdf")
        ]

        vectorize_tasks = [chunk_and_embed_files(file, embeddings_func, vector_store, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) 
                for file in pdf_files]
        # Embed Documents into a vectorDB ChromaDB 
        vectorize_results = await asyncio.gather(*vectorize_tasks)

    else:
        print(f"Collection {RUN_NAME} already exists in the database, {client.get_collection(name=RUN_NAME).count()} entries")
    # print(client.list_collections().count())

    # From first stage main.py(), refine Q/A Pairs and get answers from chunk + RAG
    filename= "outputs/agent_instruct_2024OCT27_1_output.csv"
    df = pd.read_csv(filename)
    finetune_entries = df.to_dict("records")
    finetune_entries = [FinetuneEntry(**entry) for entry in finetune_entries]

    print(f"Number of finetune entries: {len(finetune_entries)}")

    tasks = [generate_refined_questions(entry) for entry in finetune_entries]
    results = await asyncio.gather(*tasks)
    results = [r for result in results for r in result]

    # remove similar questions 
    print(f"Number of refined questions: {len(results)}")
    results = remove_similar_questions(results, embeddings_func, threshold=0.85)
    print(f"Number of refined questions after removing similar questions: {len(results)}")

    tasks = [generate_rag_answers(entry, vector_store) for entry in results]
    results = await asyncio.gather(*tasks)

    df = pd.DataFrame([r for result in results for r in result])
    df.to_csv("outputs/agent_instruct_2024OCT27_1_refined_output_3_1_8B.csv", index=False)


    # Generate QA Answers from refined questions

    # Save refined Q/A Pairs to a CSV file
    endtime = time.time()
    print(f"Execution time: {endtime - start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())
