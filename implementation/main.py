import asyncio
import logging
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.agents import CodeExecutorAgent, CodingAssistantAgent
from autogen_agentchat.logging import ConsoleLogHandler
from autogen_agentchat.teams import RoundRobinGroupChat, StopMessageTermination
from autogen_ext.code_executor.docker_executor import DockerCommandLineCodeExecutor
from autogen_core.components.models import OpenAIChatCompletionClient

logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.INFO)


async def main() -> None:
        question_generator_agent = CodingAssistantAgent(
            "coding_assistant", model_client=OpenAIChatCompletionClient(model="gpt-4o", api_key="unnecessary", base_url="http://0.0.0.0:4000"),
            system_message="""You are a helpful AI assistant that generates questions based on the dataset content."""
        )
        group_chat = RoundRobinGroupChat([question_generator_agent])
        with open("sample_paper.tex", "r") as file:
            sample_paper_content = file.read()
            result = await group_chat.run(
                task=f"Generate some knock-knock jokes, then write `terminate`",
                termination_condition=StopMessageTermination(),
            )
            print(result)

asyncio.run(main())
