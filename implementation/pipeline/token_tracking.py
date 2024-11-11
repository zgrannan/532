from typing import TypedDict, List, Dict
import json
import os

from functools import wraps
from datetime import datetime
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
import time 
import asyncio 

class TokenUsage(TypedDict):
    timestamp: str
    agent_name: str
    operation: str
    model: str
    input_text: str
    output_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    temperature: float
    execution_time: float

class TokenTracker:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.usages: List[TokenUsage] = []
        self.lock = asyncio.Lock()  # Add a lock for thread safety
        os.makedirs(log_dir, exist_ok=True)
        
    async def log_usage(self, usage: TokenUsage):
        async with self.lock:
            self.usages.append(usage)
        
    def save_to_file(self, config_name: str):
        print("saving to file: ", config_name)
        # Save detailed logs
        # async with self.lock:
        try:
            with open(f"{self.log_dir}/{config_name}_token_usage.jsonl", 'w') as f:
                for usage in self.usages:
                    f.write(json.dumps(usage) + '\n')
            # self.usages.clear()
            # print("usage cleared")
        except Exception as e:
            print(f"Failed to save to file: {e}")

    def get_usage(self):
        return self.usages
                
    # async def periodic_save(self, config_name: str, interval: int):
    #     while True:
    #         await asyncio.sleep(interval)
    #         async with self.lock:
    #             await self.save_to_file(config_name)

# Global tracker
tracker = TokenTracker("logs/token_usage")

def track_llm_usage(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()

        if not isinstance(result, ChatCompletion) and not isinstance(result, ParsedChatCompletion):
            print("result is not ChatCompletion or ParsedChatCompletion but is ", type(result))
            return result
        else:
            if isinstance(result, ParsedChatCompletion):
                # For TogetherAI and Fireworks, the parsed JSON is in message.content
                output_resp = result.choices[0].message.parsed
                if not output_resp:
                    output_resp = result.choices[0].message.content
            elif isinstance(result, ChatCompletion):
                output_resp = result.choices[0].message.content

        if isinstance(result, ParsedChatCompletion):
            if isinstance(output_resp, str):
                response = json.loads(output_resp)
            else:
                response = output_resp.model_dump()
        else:
            response = output_resp
                
        usage: TokenUsage = {
            'timestamp': datetime.now().isoformat(),
            'agent_name': kwargs.get('agent_name', 'unknown'),
            'operation': 'llm',
            'model': result.model,
            'messages': kwargs.get('messages', kwargs.get('input', [])),
            'response': response,
            'prompt_tokens': result.usage.to_dict().get('prompt_tokens', 0),
            'completion_tokens': result.usage.to_dict().get('completion_tokens', 0),
            'total_tokens': result.usage.to_dict().get('total_tokens', 0),
            'temperature': kwargs.get('temperature', 0.0),
            'execution_time': end_time - start_time
        }
        await tracker.log_usage(usage)
        return output_resp
    return wrapper