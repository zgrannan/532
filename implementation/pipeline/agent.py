from typing import (
    AsyncIterator,
    Callable,
    Iterator,
    List,
    Tuple,
    TypeVar,
    Generic,
    AsyncGenerator,
    cast,
    Optional,
)

import asyncio

from langchain_openai import OpenAIEmbeddings

from helpers import get_async_client, get_model, get_embedding_func
from abc import ABC, abstractmethod

from helpers import once

Input = TypeVar("Input")
Output = TypeVar("Output")
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
TT = TypeVar("TT")
UU = TypeVar("UU")


class Pipeline(ABC, Generic[Input, Output]):
    """
    A `Pipeline` consumes a stream of inputs and produces a stream of outputs.
    In general, a pipeline describes an orchestration of agents.
    """

    @abstractmethod
    async def _process(self, input: AsyncIterator[Input]) -> AsyncIterator[Output]:
        yield cast(
            Output, None
        )  # Just to indicate that this is an async generator and appease mypy

    def process(self, input: AsyncIterator[Input]) -> AsyncIterator[Output]:
        assert isinstance(input, AsyncIterator), "input must be an AsyncIterator"
        return self._process(input)

    def fan_out(
        self, max_parallelism: int, agent: "Agent[Output, V]"
    ) -> "Agent[Input, V]":
        return self.and_then(ParallelAgent(max_parallelism, agent))

    def map(self, func: Callable[[Output], V]) -> "FuncAgent[Input, Output, V]":
        return FuncAgent(self, func)

    def and_then(
        self, next: "Pipeline[Output, V]"
    ) -> "ComposedAgent[Input, Output, V]":
        return ComposedAgent(self, next)

    def chunk(self, chunk_size: int) -> "ChunkingAgent[Input, Output]":
        return ChunkingAgent(self, chunk_size)

    def zip_with(self, next_agent):
        return ZipWithAgent(self, next_agent)

    def enrich_with(
        self,
        pipeline: "Pipeline[TT, UU]",
        to: Callable[[Output], TT],
        frm: Callable[[Output, UU], V],
    ) -> "Pipeline[Input, V]":
        return self.and_then(EnrichAgent(pipeline, to, frm))

    def process_once(self, input: Input):
        return self.process(once(input))

    def process_list(self, input: List[Input]) -> AsyncIterator[Output]:
        async def async_generator(input: List[Input]) -> AsyncIterator[Input]:
            for item in input:
                yield item

        return self.process(async_generator(input))


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


class MapAgent(Agent[Input, Output], Generic[Input, Output]):
    async def _process(self, input: AsyncIterator[Input]) -> AsyncIterator[Output]:
        async for elem in input:
            yield await self.handle(elem)

    @abstractmethod
    async def handle(self, input: Input) -> Output:
        pass


class EnrichAgent(Agent[Input, Output], Generic[Input, Output, T, U]):
    def __init__(
        self,
        agent: MapAgent[T, U],
        to: Callable[[Input], T],
        frm: Callable[[Input, U], Output],
    ):
        self.agent = agent
        self.to = to
        self.frm = frm

    async def _process(self, input: AsyncIterator[Input]) -> AsyncIterator[Output]:
        queue: asyncio.Queue[Input] = asyncio.Queue()

        async def to_iterator() -> AsyncIterator[T]:
            async for elem in input:
                await queue.put(elem)
                yield self.to(elem)

        async for elem in self.agent.process(to_iterator()):
            orig_input = await queue.get()
            yield self.frm(orig_input, elem)


class OpenAIAgent(Agent[Input, Output]):
    def __init__(self, model: str, embedding_model: Optional[str] = None):
        self.model = get_model(model)
        self.client = get_async_client()
        self.embedding_func: Optional[OpenAIEmbeddings] = (
            get_embedding_func(embedding_model) if embedding_model else None
        )


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


class _AllDone:
    """Sentinel value to indicate that the iterator is exhausted."""

    pass

class ParallelAgent(Agent[T, U]):
    def __init__(self, max_parallelism: int, agent: "Agent[T, U]"):
        self.max_parallelism = max_parallelism
        self.agent = agent
        self.semaphore = asyncio.Semaphore(max_parallelism)

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[U]:
        queue: asyncio.Queue[U] = asyncio.Queue()
        tasks = []

        async def worker(elem: T):
            async with self.semaphore:
                # Process the element and put outputs into the queue
                async for output in self.agent.process_once(elem):
                    await queue.put(output)

        async def producer():
            # Start worker tasks for each input element
            async for elem in input:
                task = asyncio.create_task(worker(elem))
                tasks.append(task)
            # Wait for all tasks to complete
            await asyncio.gather(*tasks)
            # Signal that processing is complete
            await queue.put(None)

        # Start the producer task
        producer_task = asyncio.create_task(producer())

        # Consume outputs from the queue as they become available
        while True:
            output = await queue.get()
            if output is None:
                break
            yield output

        # Ensure the producer task has completed
        await producer_task


class FuncAgent(Agent[T, V], Generic[T, U, V]):
    def __init__(self, agent: Pipeline[T, U], func: Callable[[U], V]):
        self.agent = agent
        self.func = func

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[V]:
        async for elem in self.agent.process(input):
            yield self.func(elem)


class TakeOnly(Agent[T, T]):
    def __init__(self, n: int):
        self.n = n

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[T]:
        i = 0
        async for elem in input:
            yield elem
            i += 1
            if i >= self.n:
                break

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

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[Tuple[U, U]]:
        async for elem in self.agent.process(input):
            yield (elem, elem)


class ZipWithAgent(Agent[T, Tuple[U, V]], Generic[T, U, V]):
    def __init__(self, agent1: Agent[T, U], agent2: Agent[U, V]):
        self.agent1 = agent1
        self.agent2 = agent2

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[Tuple[U, V]]:
        async for elem in self.agent1.process(input):
            async for elem2 in self.agent2.process(once(elem)):
                yield (elem, elem2)


class UnchunkingAgent(Agent[List[T], T]):
    async def _process(self, input: AsyncIterator[List[T]]) -> AsyncIterator[T]:
        async for elem in input:
            for item in elem:
                yield item

class ChunkingAgent(Agent[T, List[U]]):
    def __init__(self, agent: Pipeline[T, U], chunk_size: int):
        self.agent = agent
        self.chunk_size = chunk_size

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[List[U]]:
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
