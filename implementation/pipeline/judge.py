import argparse
import hashlib
import json
from openai import AsyncOpenAI
import pandas as pd
import asyncio
import os
from pydantic import SecretStr
from helpers import (
    get_simple_response,
)
from enum import Enum
from typing import AsyncIterator, Awaitable, TypedDict, List, Callable

from agent import Cache
from helpers import get_output_model_name, get_pipeline_config_name
from helpers import get_results_file_path
from rag import EMBEDDING_MODEL, get_vector_store
import requests


class BetterAnswer(Enum):
    A = "A"
    B = "B"


class EvalWinner(Enum):
    BASE = "BASE"
    FINE_TUNED = "FINE_TUNED"


class EvalQuestion(TypedDict):
    context: str
    question: str
    true_answer: str


JUDGE_SYSTEM_PROMPT = """
    You are an expert judge that determines which one of two answers is more similar to the true answer.
    The context is {context}.
    The question is {question}.
    The true answer is {true_answer}.
    The first answer is {answer_a}.
    The second answer is {answer_b}.

    First, think step by step about which answer is more similar to the true answer.
    If an answer contains factually incorrect information or contradicts the true answer
    it should be strongly penalized.
    If the question refers to a specific source, and the answer indicates that
    it is not aware about the specific source, it should be penalized even if it
    gives good general information.
    An answer that provides specific and correct information should be preferred to a
    more vague answer that provides general information.

    When you have made your decision, complete your response with the string
    "ANSWER_A" if the first answer is better, and "ANSWER_B" if the second
    answer is better (exclude the quotes).
"""

JUDGE_MODEL = "gpt-4o"  # We want to use a good model here.


async def judge(
    context: str, question: str, true_answer: str, answer_a: str, answer_b: str
) -> tuple[BetterAnswer, str]:

    prompt = JUDGE_SYSTEM_PROMPT.format(
        context=context,
        question=question,
        true_answer=true_answer,
        answer_a=answer_a,
        answer_b=answer_b,
    )

    # Use default OpenAI client for gpt4
    print(os.environ["OPENAI_API_KEY"])
    client = AsyncOpenAI()

    response = await get_simple_response(client, JUDGE_MODEL, prompt)

    if response.strip().endswith("ANSWER_A"):
        return BetterAnswer.A, response
    elif response.strip().endswith("ANSWER_B"):
        return BetterAnswer.B, response

    # LLM didn't follow instructions, lets try some basic heuristics

    if "ANSWER_A" in response.strip() and "ANSWER_B" in response.strip():
        raise ValueError(
            f"Shouldn't have both ANSWER_A and ANSWER_B in response: {response}"
        )
    elif "ANSWER_A" in response.strip():
        return BetterAnswer.A, response
    elif "ANSWER_B" in response:
        return BetterAnswer.B, response
    else:
        raise ValueError(
            f"Unexpected response format (should end with ANSWER_A or ANSWER_B): {response}"
        )


ResponseGenerator = Callable[[str], Awaitable[str]]


class JudgeResult:
    def __init__(
        self,
        question: str,
        context: str,
        true_answer: str,
        base_answer: str,
        finetuned_answer: str,
        better_answer: EvalWinner,
        judge_response: str,
    ):
        self.question = question
        self.context = context
        self.true_answer = true_answer
        self.base_answer = base_answer
        self.finetuned_answer = finetuned_answer
        self.better_answer = better_answer
        self.judge_response = judge_response

    def to_json(self) -> dict:
        return {
            "question": self.question,
            "context": self.context,
            "true_answer": self.true_answer,
            "base_answer": self.base_answer,
            "finetuned_answer": self.finetuned_answer,
            "better_answer": self.better_answer.value,
            "judge_response": self.judge_response,
        }

    def save_json(self, filename: str) -> None:
        import json
        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            json.dump(self.to_json(), f, indent=2)

    def __str__(self):
        separator = "=" * 80
        dash = "-" * 80
        return (
            f"Question: {self.question}\n"
            f"Context: {self.context}\n"
            f"\n{separator}\n"
            f"TRUE ANSWER:\n"
            f"{dash}\n"
            f"{self.true_answer}\n"
            f"\n{separator}\n"
            f"BASE MODEL ANSWER:\n"
            f"{dash}\n"
            f"{self.base_answer}\n"
            f"\n{separator}\n"
            f"FINETUNED MODEL ANSWER:\n"
            f"{dash}\n"
            f"{self.finetuned_answer}\n"
            f"{separator}\n"
            f"Better Answer: {self.better_answer}\n"
            f"Judge Response: {self.judge_response}\n"
        )


async def judge_questions(
    eval_questions: List[EvalQuestion],
    get_base_response: ResponseGenerator,
    get_finetuned_response: ResponseGenerator,
) -> AsyncIterator[JudgeResult]:

    judge_cache = Cache(f"cache/{JUDGE_MODEL}_judge_cache")
    max_concurrency = 24
    semaphore = asyncio.Semaphore(max_concurrency)

    async def judge_question(eval_question: EvalQuestion) -> JudgeResult:
        context = eval_question["context"]
        question = eval_question["question"]
        true_answer = eval_question["true_answer"]

        base_answer, finetuned_answer = await asyncio.gather(
            get_base_response(question), get_finetuned_response(question)
        )

        better_answer, judge_response = await judge_cache.apply(
            lambda: judge(
                context, question, true_answer, base_answer, finetuned_answer
            ),
            (
                JUDGE_SYSTEM_PROMPT,
                context,
                question,
                true_answer,
                base_answer,
                finetuned_answer,
            ),
        )

        if better_answer == BetterAnswer.A:
            model_type = EvalWinner.BASE
        elif better_answer == BetterAnswer.B:
            model_type = EvalWinner.FINE_TUNED
        else:
            raise ValueError(f"Unexpected better answer: {better_answer}")

        return JudgeResult(
            question,
            context,
            true_answer,
            base_answer,
            finetuned_answer,
            model_type,
            judge_response,
        )

    async def judge_question_wrapper(eval_question: EvalQuestion) -> JudgeResult:
        async with semaphore:
            return await judge_question(eval_question)

    tasks = [
        asyncio.create_task(judge_question_wrapper(eval_question))
        for eval_question in eval_questions
    ]

    for task in tasks:
        yield await task


FINETUNED_MODEL_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
BASE_MODEL_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


class PayloadOptions(TypedDict):
    max_replica: int
    scale_to_zero_timeout: int
    parallelism: int
    model_repository: str
    endpoint_name: str
    gguf_file: str


def get_create_endpoint_args(options: PayloadOptions):
    return {
        "compute": {
            "accelerator": "gpu",
            "instanceSize": "x1",
            "instanceType": "nvidia-t4",
            "scaling": {
                "maxReplica": options["max_replica"],
                "minReplica": 0,
                "scaleToZeroTimeout": options["scale_to_zero_timeout"],
                "metric": "hardwareUsage",
            },
        },
        "model": {
            "env": {},
            "framework": "llamacpp",
            "image": {
                "llamacpp": {
                    "ctxSize": 81920,
                    "embeddings": False,
                    "healthRoute": "/health",
                    "modelPath": options["gguf_file"],
                    "nParallel": options["parallelism"],
                    "port": 80,
                    "threadsHttp": options["parallelism"] * 2,
                    "url": "ghcr.io/ggerganov/llama.cpp:server-cuda",
                }
            },
            "repository": options["model_repository"],
            "secrets": {},
            "task": "text-generation",
        },
        "name": options["endpoint_name"],
        "provider": {"region": "us-east-1", "vendor": "aws"},
        "type": "protected",
    }


def create_endpoint(options: PayloadOptions):
    url = "https://api.endpoints.huggingface.cloud/v2/endpoint/zgrannan"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BASE_MODEL_API_KEY}",
    }
    data = get_create_endpoint_args(options)

    response = requests.post(url, headers=headers, json=data)
    print(response.status_code, response.text)
    pass


def delete_endpoint(endpoint_name: str):
    url = f"https://api.endpoints.huggingface.cloud/v2/endpoint/zgrannan/{endpoint_name}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BASE_MODEL_API_KEY}",
    }
    response = requests.delete(url, headers=headers)
    print(response.status_code, response.text)

async def get_endpoint_url(options: PayloadOptions) -> str | None:
    endpoint_name = options["endpoint_name"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BASE_MODEL_API_KEY}",
    }
    url = (
        f"https://api.endpoints.huggingface.cloud/v2/endpoint/zgrannan/{endpoint_name}"
    )

    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            response_json = response.json()
            if "status" in response_json and "url" in response_json["status"]:
                return response_json["status"]["url"]
        else:
            return None
        print(f"Waiting for endpoint {endpoint_name} to be ready...")
        await asyncio.sleep(5)


async def get_or_create_endpoint(options: PayloadOptions) -> str | None:
    endpoint_url = await get_endpoint_url(options)
    if endpoint_url is None:
        create_endpoint(options)
        endpoint_url = await get_endpoint_url(options)
    return endpoint_url


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to the config file to use",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        options = json.load(f)
    matrix = options["matrix"]

    runs = []

    for source in matrix["pipeline_config.sources"]:
        for llm in matrix["pipeline_config.llm"]:
            for r in matrix["finetune_config.r"]:
                runs.append((source, llm, r))

    for source, llm, r in runs:
        pipeline_config_name = get_pipeline_config_name(source, llm["model"])
        finetuned_model_name = get_output_model_name(pipeline_config_name, r)
        results_file_path = get_results_file_path(finetuned_model_name)
        if os.path.exists(results_file_path):
            print(f"Results file {results_file_path} already exists. Skipping...")
            continue
        # Needs to be 32 characters or less, a-z, 0-9, and -
        finetuned_endpoint_name = hashlib.sha256(finetuned_model_name.encode()).hexdigest()[:32]
        print(f"Endpoint name: {finetuned_endpoint_name}")
        judge_config_template = options["template"]["judge_config"]
        test_dataset = f"outputs/{pipeline_config_name}_test_output.csv"

        df = pd.read_csv(test_dataset)
        finetune_entries = df.to_dict("records")
        print(f"Loaded {len(finetune_entries)} finetune entries from {test_dataset}")

        eval_questions = [
            EvalQuestion(
                context=entry["source"],
                question=entry["question"],
                true_answer=entry["answer"],
            )
            for entry in finetune_entries
        ]

        finetuned_model_options = judge_config_template["finetuned_model"]

        finetuned_model_options = PayloadOptions(
            max_replica=finetuned_model_options["max_replica"],
            scale_to_zero_timeout=finetuned_model_options["scale_to_zero_timeout"],
            parallelism=finetuned_model_options["parallelism"],
            model_repository=f"CPSC532/{finetuned_model_name}",
            endpoint_name=finetuned_endpoint_name,
            gguf_file=finetuned_model_options["gguf_file"],
        )

        (base_endpoint_url, finetuned_endpoint_url) = await asyncio.gather(
            get_or_create_endpoint(judge_config_template["base_model"]),
            get_or_create_endpoint(finetuned_model_options),
        )

        if base_endpoint_url is None or finetuned_endpoint_url is None:
            raise ValueError("Failed to get or create endpoint")

        base_client = AsyncOpenAI(
            base_url=base_endpoint_url,
            api_key=BASE_MODEL_API_KEY,
        )

        vector_store = get_vector_store(pipeline_config_name)
        rag_k = options["template"]["judge_config"]["rag_k"]

        finetuned_client = AsyncOpenAI(
            base_url=finetuned_endpoint_url,
            api_key=FINETUNED_MODEL_API_KEY,
        )

        base_model_name = judge_config_template["base_model"]["endpoint_name"]

        base_cache = Cache(f"cache/{base_model_name}_judge_cache")
        finetuned_cache = Cache(f"cache/{finetuned_model_name}_judge_cache")

        async def get_base_response(question: str) -> str:
            context = vector_store.similarity_search_with_score(question, k=rag_k)
            prompt = f"""Answer the following question based on the provided context:
            Question: {question}
            Context:
            {context}
            """
            return await base_cache.apply(
                lambda: get_simple_response(base_client, base_model_name, prompt),
                prompt,
            )

        async def get_finetuned_response(question: str) -> str:
            return await finetuned_cache.apply(
                lambda: get_simple_response(
                    finetuned_client, finetuned_model_name, question
                ),
                question,
            )

        results = []
        i = 0
        async for result in judge_questions(
            eval_questions, get_base_response, get_finetuned_response
        ):
            results.append(result)
            print(f"Winner: {result.better_answer}")
            result.save_json(f"judge_results/{pipeline_config_name}/{i}.json")
            i += 1
        # Count winners
        base_wins = sum(1 for r in results if r.better_answer == EvalWinner.BASE)
        finetuned_wins = sum(
            1 for r in results if r.better_answer == EvalWinner.FINE_TUNED
        )

        print("\nFinal Results:")
        print(f"Base Model Wins: {base_wins}")
        print(f"Fine-tuned Model Wins: {finetuned_wins}")
        print(f"Total Comparisons: {len(results)}")
        print(f"Fine-tuned Win Rate: {(finetuned_wins/len(results))*100:.1f}%")

        # Ensure the directory exists
        os.makedirs("../eval_results", exist_ok=True)

        # Write the results to the file
        with open(results_file_path, "w") as results_file:
            json.dump(
                {
                    "base_wins": base_wins,
                    "finetuned_wins": finetuned_wins,
                },
                results_file,
                indent=4,
            )
        print(f"Results written to {results_file_path}")
        delete_endpoint(finetuned_endpoint_name)


if __name__ == "__main__":
    asyncio.run(main())
