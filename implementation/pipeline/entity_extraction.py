from pydantic import BaseModel, Field
from typing import AsyncGenerator, AsyncIterator, List
from openai import OpenAI, AsyncOpenAI
from agent import OpenAIAgent
from helpers import get_json_response_async, get_model
from agent import MapAgent


ENTITY_EXTRACTION_SYSTEM_PROMPT = """
  You are an entity extractor that extracts entities and facts from the following document in JSON. Your goal is to only extract entities and facts as a JSON object.
  Entities can be concepts, topics, objects, people, place, dates, or important information.
  Do not duplicate entities. Extract only the important information that is relevant to the document.

  Below is an example

  ### Task:
  Given a piece of text, extract conceptual entities and facts. Return the output in JSON format with two keys: "entities" and "facts". The value for each key should be a list of strings.

  ### Example 1:
  Text: "Learning generative image models from various domains efficiently needs transferring knowledge from an image synthesis model trained on a large dataset. We present a recipe for learning vision transformers by generative knowledge transfer. We base our framework on generative vision transformers representing an image as a sequence of visual tokens with the autoregressive or non-autoregressive transformers. To adapt to a new domain, we employ prompt tuning, which prepends learnable tokens called prompts to the image token sequence and introduces a new prompt design for our task. We study on a variety of visual domains with varying amounts of training images. We show the effectiveness of knowledge transfer and a significantly better image generation quality."

  Output (JSON):
  ```json
  {{
    "entities": [
      "Generative image models",
      "Knowledge transfer",
      "Vision transformers",
      "Generative knowledge transfer",
      "Image synthesis",
      "Autoregressive transformers",
      "Non-autoregressive transformers",
      "Prompt tuning",
      "Image token sequence",
      "Prompt design"
    ],
    "facts": [
      "Generative image models require knowledge transfer from large datasets for efficient learning.",
      "Vision transformers represent images as sequences of visual tokens.",
      "Prompt tuning adapts vision transformers by adding learnable tokens to the image token sequence.",
      "The study involves different visual domains with varying amounts of training images.",
      "Knowledge transfer improves image generation quality."
    ]
  }}

  Do not generate more than {max_entities} entities.

  Document:
  {text}
"""


class EntityExtractionModel(BaseModel):
    entities: List[str]
    # facts: List[str]


ENTITY_EXTRACTION_MODEL = "meta-llama-3.1-8b-instruct-q6_k"


class EntityExtractionAgent(OpenAIAgent, MapAgent[str, List[str]]):
    def __init__(self, max_entities: int):
        super().__init__(ENTITY_EXTRACTION_MODEL)
        MapAgent.__init__(self, name="Entity Extraction Agent")
        self.max_entities = max_entities

    async def handle(self, input: str) -> List[str]:
        entity_prompt = ENTITY_EXTRACTION_SYSTEM_PROMPT.format(
            text=input, max_entities=self.max_entities
        )
        entities = await get_json_response_async(
            client=self.client,
            model=get_model(ENTITY_EXTRACTION_MODEL),
            messages=[
                {"role": "system", "content": entity_prompt},
            ],
            response_format=EntityExtractionModel,
        )
        # The model may ignore max_entities restriction in the prompt
        return entities.entities[: self.max_entities]
