from openai import AsyncOpenAI
import pandas as pd
import asyncio
import os, sys 
import argparse

from helpers import (
    get_simple_response,
    get_openai_client,
    get_model,
    get_async_client,
    get_default_client,
)
import time 
from dotenv import find_dotenv, load_dotenv

load_dotenv()
from typing import AsyncIterator, Awaitable, TypedDict, List, Callable
from tqdm import tqdm


class EvalQuestion(TypedDict):
    context: str
    question: str
    true_answer: str



def main(TEST_DATASET, SAVE_NAME, BASE_MODEL, BASE_MODEL_ENDPOINT):
    start_time_main = time.time()
    BASE_MODEL_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

    # Initialize the base client
    base_client = AsyncOpenAI(
        base_url=BASE_MODEL_ENDPOINT,
        api_key=BASE_MODEL_API_KEY,
    )

    # Load the dataset
    df = pd.read_csv(TEST_DATASET)
    print(f"Loaded dataset with shape: {df.shape}")

    # Convert dataframe to list of EvalQuestion objects
    finetune_entries = df.to_dict("records")
    eval_questions = [
        EvalQuestion(
            context=entry["source"],
            question=entry["question"],
            true_answer=entry["answer"],
        )
        for entry in finetune_entries
        if entry["answer"] != "NO ANSWER FOUND"  # Hack for cleanup, in the future these will be stripped earlier
    ]
    print(f"Number of evaluation questions: {len(eval_questions)}")

    async def get_base_response(question: str) -> str:
        return await get_simple_response(base_client, BASE_MODEL, question)

    # Function to fetch all responses concurrently
    async def fetch_all_responses(eval_questions):
        responses = []

        async def fetch_response(question):
            start_time = time.time()
            response = await get_base_response(question.get('question'))
            end_time = time.time()
            print(f"Question: {question.get('question')}")
            print(f"Time taken: {end_time - start_time} seconds")
            responses.append(response)

        tasks = []
        for question in tqdm(eval_questions, desc="Processing questions"):
            task = asyncio.create_task(fetch_response(question))
            tasks.append(task)
            if len(tasks) == 10:
                await asyncio.gather(*tasks)
                tasks = []
        if tasks:
            await asyncio.gather(*tasks)

        return responses

    # Run the async function and collect responses
    responses = asyncio.run(fetch_all_responses(eval_questions))
    print(f"Collected {len(responses)} responses")


    for i in range(len(responses)):
        eval_questions[i]["generated_answer"] = responses[i]

    results_df = pd.DataFrame(eval_questions)
    os.makedirs('evaluations', exist_ok=True)
    results_df.to_csv(f"evaluations/{SAVE_NAME}.csv", index=False)

    print(f"Responses saved to {SAVE_NAME}")
    print(f"Total time taken: {time.time() - start_time_main} seconds")

if __name__ == "__main__":
    # TEST_DATASET = "outputs/2024NOV16_llama_3_1_8b_no_sources_in_question_test_output.csv"
    # SAVE_NAME = "2024NOV16_llama_3_1_8b_no_sources_in_question_test_output_model_finetuned_no_sources_in_question"
    # BASE_MODEL = "2024nov16-llama-3-1-8b-no-so-xax"
    # BASE_MODEL_ENDPOINT = "https://lxpicfuqx9ox3qpu.us-east-1.aws.endpoints.huggingface.cloud"

    parser = argparse.ArgumentParser(description="Run Hugging Face inference and save responses.")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to the test dataset CSV file.")
    parser.add_argument("--save_name", type=str, required=True, help="Name of the file to save responses.")
    parser.add_argument("--base_model", type=str, required=True, help="Name of the base model.")
    parser.add_argument("--base_model_endpoint", type=str, required=True, help="Endpoint URL of the base model.")

    args = parser.parse_args()

    main(args.test_dataset, args.save_name, args.base_model, args.base_model_endpoint)