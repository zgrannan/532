from typing import (
    AsyncIterator,
    Callable,
    List,
    Tuple,
    TypeVar,
    Generic,
    AsyncGenerator,
    cast,
)

import asyncio

from helpers import get_async_client, get_model
from abc import ABC, abstractmethod

from helpers import once

Input = TypeVar("Input")
Output = TypeVar("Output")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class Pipeline(ABC, Generic[Input, Output]):
    """
    A `Pipeline` consumes a stream of inputs and produces a stream of outputs.
    In general, a pipeline describes an orchestration of agents.
    """

    @abstractmethod
    async def _process(
        self, input: AsyncIterator[Input]
    ) -> AsyncIterator[Output]:
        yield cast(
            Output, None
        )  # Just to indicate that this is an async generator and appease mypy

    def process(
        self, input: AsyncIterator[Input]
    ) -> AsyncIterator[Output]:
        assert isinstance(input, AsyncIterator), "input must be an AsyncIterator"
        return self._process(input)


    def and_then(
        self, next: "Pipeline[Output, V]"
    ) -> "ComposedAgent[Input, Output, V]":
        return ComposedAgent(self, next)

    def chunk(self, chunk_size: int) -> "ChunkingAgent[Input, Output]":
        return ChunkingAgent(self, chunk_size)

    def zip_with(self, next_agent):
        return ZipWithAgent(self, next_agent)

    def process_once(self, input: Input):
        return self.process(once(input))

class Agent(Pipeline[Input, Output], Generic[Input, Output]):
    """
    An abstract base class for defining *agents*.

    An `Agent[T, U]` is a stateful object that receives an asynchronous stream
    of inputs (each of type `T`) and produces an asynchronous stream of outputs
    (each of type `U`). For each input, the agent can produce zero or more
    outputs; in principle it is also possible for the agent to produce outputs
    before receiving any inputs or ignore the input stream entirely.

    The `Agent` interface does not provide a built-in mechanism for an agent to
    determine where its inputs come from, or where its outputs go to. In
    particular the agent itself does not specify "where" its outputs are
    intended to go, and there is no address system for agents. Rather, to
    facilitate compositionality, the `Pipeline` interface can be used to define
    a topology of agents. Each `Agent[T, U]` is a `Pipeline[T, U]`, and pipeline
    composition operations can be used to orchestrate communication between
    agents.

    Implementors of the `Agent` interface must implement the `_process` method,
    which consumes a stream of inputs and produces a stream of outputs. In
    general, the execution of an agent consists of a single call to this method.
    However, it is possible to call an Agent's `process` method multiple times.
    Such a mechanism could be used to have the agent process multiple streams
    simultaneously (when the order of its outputs does not matter).

    Agents have two mechanisms for storing state. First, agents are objects, and
    therefore can store state in instance variables. Second, the `process`
    method is a coroutine, and can therefore maintain state. Which type of state
    to use depends on the use-case.
    """

class OpenAIAgent(Agent[Input, Output]):
    def __init__(self, model: str):
        self.model = get_model(model)
        self.client = get_async_client()

    @abstractmethod
    async def _process(
        self, input: AsyncIterator[Input]
    ) -> AsyncIterator[Output]:
        yield cast(
            Output, None
        )  # Just to indicate that this is an async generator and appease mypy

async def merge_iterators_in_order(
    iterators: List[AsyncIterator[T]], max_queue_size: int = 10
) -> AsyncIterator[T]:
    """Merge multiple async iterators into a single async iterator.

    Args:
        iterators: A list of asynchronous iterators to merge.
        max_queue_size: Maximum size of the buffer queue for each iterator.

    Yields:
        Items from each iterator in order, processing iterators concurrently.
    """
    queues: List[asyncio.Queue] = [
        asyncio.Queue(maxsize=max_queue_size) for _ in iterators
    ]
    tasks = []
    for iterator, queue in zip(iterators, queues):
        task = asyncio.create_task(_fill_queue(iterator, queue))
        tasks.append(task)

    # Yield items from each queue in order
    for queue in queues:
        while True:
            item = await queue.get()
            if item is _QueueDone:
                break
            yield item
            queue.task_done()

    # Wait for all fill tasks to complete
    await asyncio.gather(*tasks)

async def _fill_queue(iterator: AsyncIterator[T], queue: asyncio.Queue):
    """Fill the queue with items from the iterator."""
    try:
        async for item in iterator:
            await queue.put(item)
    finally:
        await queue.put(_QueueDone)

class _QueueDone:
    """Sentinel value to indicate that the iterator is exhausted."""

    pass

class ParallelAgent(Agent[T, V], Generic[T, U, V]):
    def __init__(self, agent: Agent[T, List[U]], spawn: Callable[[], Agent[U, V]]):
        self.agent = agent
        self.spawn = spawn

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[V]:
        async for list in self.agent.process(input):
            async for elem in merge_iterators_in_order(
                [self.spawn().process(once(elem)) for elem in list]
            ):
                yield elem

class MapAgent(Agent[T, V], Generic[T, U, V]):
    def __init__(self, agent: Agent[T, U], func: Callable[[U], V]):
        self.agent = agent
        self.func = func

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[V]:
        async for elem in self.agent.process(input):
            yield self.func(elem)

class ComposedAgent(Agent[T, V], Generic[T, U, V]):
    def __init__(self, agent1: Pipeline[T, U], agent2: Pipeline[U, V]):
        self.agent1 = agent1
        self.agent2 = agent2

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[V]:
        async for output in self.agent2.process(self.agent1.process(input)):
            yield output

class Duplicate(Agent[T, Tuple[U, U]], Generic[T, U]):
    def __init__(self, agent: Agent[T, U]):
        self.agent = agent

    async def _process(
        self, input: AsyncIterator[T]
    ) -> AsyncIterator[Tuple[U, U]]:
        async for elem in self.agent.process(input):
            yield (elem, elem)

class ZipWithAgent(Agent[T, Tuple[U, V]], Generic[T, U, V]):
    def __init__(self, agent1: Agent[T, U], agent2: Agent[U, V]):
        self.agent1 = agent1
        self.agent2 = agent2

    async def _process(
        self, input: AsyncIterator[T]
    ) -> AsyncIterator[Tuple[U, V]]:
        async for elem in self.agent1.process(input):
            async for elem2 in self.agent2.process(once(elem)):
                yield (elem, elem2)

class ChunkingAgent(Agent[T, List[U]]):
    def __init__(self, agent: Pipeline[T, U], chunk_size: int):
        self.agent = agent
        self.chunk_size = chunk_size

    def parallel(self, spawn: Callable[[], Agent[U, V]]) -> "ParallelAgent[T, U, V]":
        return ParallelAgent(self, spawn)

    async def _process(
        self, input: AsyncIterator[T]
    ) -> AsyncIterator[List[U]]:
        # Get the output generator from the underlying agent
        output_generator = self.agent.process(input)
        chunk: List[U] = []
        async for output in output_generator:
            chunk.append(output)
            if len(chunk) == self.chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk  # Yield any remaining items as the last chunk
