import itertools
import json
import os
import tempfile
import diskcache
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Tuple,
    TypeVar,
    Generic,
    AsyncGenerator,
    cast,
    Optional,
)
import time
import asyncio

from langchain_openai import OpenAIEmbeddings
from openai.types.chat import ChatCompletionMessageParam

from helpers import get_async_client, get_model, get_embedding_func
from abc import ABC, abstractmethod

from helpers import once
from helpers import get_messages_response_async
from datetime import datetime

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
    def edges(self) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def nodes(self) -> Dict[int, str]:
        pass

    @abstractmethod
    def input_id(self) -> int:
        pass

    @abstractmethod
    def output_id(self) -> int:
        pass

    @abstractmethod
    async def _process(self, input: AsyncIterator[Input]) -> AsyncIterator[Output]:
        yield cast(
            Output, None
        )  # Just to indicate that this is an async generator and appease mypy

    def process(self, input: AsyncIterator[Input]) -> AsyncIterator[Output]:
        assert isinstance(input, AsyncIterator), "input must be an AsyncIterator"

        async def iterator() -> AsyncIterator[Input]:
            async for elem in input:
                assert elem is not None, "`None` should not be used as a pipeline input"
                yield elem

        return self._process(iterator())

    def fan_out(
        self, max_parallelism: int, agent: "Agent[Output, V]"
    ) -> "Pipeline[Input, V]":
        return self.and_then(ParallelAgent(max_parallelism, agent))

    def map(self, func: Callable[[Output], V]) -> "FuncAgent[Input, Output, V]":
        return FuncAgent(self, func)

    def and_then(self, next: "Pipeline[Output, V]") -> "Compose[Input, Output, V]":
        return Compose(self, next)

    def chunk(self, chunk_size: int) -> "Pipeline[Input, List[Output]]":
        return self.and_then(ChunkingAgent(chunk_size))

    def enrich_with(
        self,
        agent: "MapAgent[TT, UU]",
        to: Callable[[Output], TT],
        frm: Callable[[Output, UU], V],
    ) -> "Pipeline[Input, V]":
        return self.and_then(EnrichAgent(agent, to, frm))

    def process_once(self, input: Input):
        return self.process(once(input))

    def process_list(self, input: List[Input]) -> AsyncIterator[Output]:
        async def async_generator(input: List[Input]) -> AsyncIterator[Input]:
            for item in input:
                yield item

        return self.process(async_generator(input))

    def to_dot(self) -> str:
        # Generate the DOT string
        START_ID = next(Agent.id_counter)
        END_ID = next(Agent.id_counter)
        nodes = {**self.nodes(), START_ID: "START", END_ID: "END"}
        edges = [
            (START_ID, self.input_id()),
            *self.edges(),
            (self.output_id(), END_ID),
        ]
        dot_lines = ["digraph G {"]
        for node_id, label in nodes.items():
            # Escape quotes in labels
            label = label.replace('"', '\\"')
            dot_lines.append(f'    node{node_id} [label="{label}"];')
        for from_id, to_id in edges:
            dot_lines.append(f"    node{from_id} -> node{to_id};")
        dot_lines.append("}")
        return "\n".join(dot_lines)


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

    id_counter = itertools.count()

    def __init__(self, name: Optional[str] = None):
        self.id = next(Agent.id_counter)
        self.name = name if name else f"{self.__class__.__name__}-{self.id}"

    def input_id(self) -> int:
        return self.id

    def output_id(self) -> int:
        return self.id

    def edges(self) -> List[Tuple[int, int]]:
        return []

    def nodes(self) -> Dict[int, str]:
        return {self.id: self.name}


class StatelessAgent(Agent[Input, Output]):
    """
    An agent whose `_process` method does not maintain state, i.e. does not
    return different results based on the order of its inputs
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    @abstractmethod
    def process_element(self, input: Input) -> AsyncIterator[Output]:
        pass

    async def _process(self, input: AsyncIterator[Input]) -> AsyncIterator[Output]:
        async for elem in input:
            async for output in self.process_element(elem):
                yield output

    def with_cache(self, filename: str, batch_size: int = 10) -> "Agent[Input, Output]":
        return CacheStatelessAgent(self, filename, batch_size)


class MapAgent(StatelessAgent[Input, Output], Generic[Input, Output]):
    """
    An agent that returns a single output for each input.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    async def process_element(self, input: Input) -> AsyncIterator[Output]:
        yield await self.handle(input)

    @abstractmethod
    async def handle(self, input: Input) -> Output:
        pass

    def with_cache(
        self, filename: str, batch_size: int = 10
    ) -> "MapAgent[Input, Output]":
        return CacheMapAgent(self, filename, batch_size)


class EnrichAgent(Agent[Input, Output], Generic[Input, Output, T, U]):
    def __init__(
        self,
        agent: MapAgent[T, U],
        to: Callable[[Input], T],
        frm: Callable[[Input, U], Output],
    ):
        super().__init__(f"Enrich")
        self.agent = agent
        self.to = to
        self.frm = frm

    def nodes(self) -> Dict[int, str]:
        return {**self.agent.nodes(), self.id: self.name}

    def edges(self) -> List[Tuple[int, int]]:
        return [
            (self.id, self.agent.id),
            (self.agent.id, self.id),
            *self.agent.edges(),
        ]

    async def _process(self, input: AsyncIterator[Input]) -> AsyncIterator[Output]:
        queue: asyncio.Queue[Input] = asyncio.Queue()

        async def to_iterator() -> AsyncIterator[T]:
            async for elem in input:
                await queue.put(elem)
                try:
                    yield self.to(elem)
                except Exception as e:
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in `to` function of {self.name}: {e}"
                    )
                    # raise e

        async for elem in self.agent.process(to_iterator()):
            orig_input = await queue.get()
            yield self.frm(orig_input, elem)


class OpenAIAgent:
    def __init__(self, model: str, embedding_model: Optional[str] = None):
        self.model = get_model(model)
        self.client = get_async_client()
        self.embedding_func: Optional[OpenAIEmbeddings] = (
            get_embedding_func(embedding_model) if embedding_model else None
        )


class OpenAIMessagesAgent(
    OpenAIAgent,
    MapAgent[list[ChatCompletionMessageParam], str],
):
    def __init__(self, model: str):
        super().__init__(model)
        MapAgent.__init__(self, name="OpenAIMessagesAgent")

    async def handle(self, input: list[ChatCompletionMessageParam]) -> str:
        return await get_messages_response_async(
            client=self.client, model=self.model, messages=input, agent_name=self.name
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
        super().__init__(f"Parallelize x{max_parallelism}")
        self.max_parallelism = max_parallelism
        self.agent = agent
        self.semaphore = asyncio.Semaphore(max_parallelism)

    def edges(self) -> List[Tuple[int, int]]:
        return [
            (self.id, self.agent.id),
            (self.agent.id, self.id),
            *self.agent.edges(),
        ]

    def nodes(self) -> Dict[int, str]:
        return {**self.agent.nodes(), self.id: self.name}

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[U]:
        queue: asyncio.Queue[U | None | Exception] = asyncio.Queue()
        tasks = []

        async def worker(elem: T):
            async with self.semaphore:
                try:
                    # Process the element and put outputs into the queue
                    async for output in self.agent.process_once(elem):
                        await queue.put(output)
                except Exception as e:
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Exception from agent {self.agent.name}: {e}"
                    )
                    # await queue.put(e)  # Put exception into queue to propagate
                    # raise e

        async def producer():
            try:
                # Start worker tasks for each input element
                async for elem in input:
                    task = asyncio.create_task(worker(elem))
                    tasks.append(task)
            except Exception as e:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Exception from producer of {self.name}: {e}"
                )
                # await queue.put(e)
            finally:
                await asyncio.gather(*tasks, return_exceptions=True)
                await queue.put(None)

        # Start the producer task
        producer_task = asyncio.create_task(producer())

        # Consume outputs from the queue as they become available
        while True:
            try:
                output = await queue.get()
                if output is None:
                    break
                if isinstance(output, Exception):
                    print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping error - exception from {self.agent.name}: {output}"
                    )
                    continue
                    # await producer_task
                    # raise output
                yield output
            except Exception as e:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error in queue processing {str(e)}"
                )
                continue
                # raise e

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
        if self.n == 0:
            return
        async for elem in input:
            yield elem
            i += 1
            if i >= self.n:
                break


class Compose(Pipeline[T, V], Generic[T, U, V]):
    def __init__(self, agent1: Pipeline[T, U], agent2: Pipeline[U, V]):
        self.left = agent1
        self.right = agent2

    def input_id(self) -> int:
        return self.left.input_id()

    def output_id(self) -> int:
        return self.right.output_id()

    def edges(self) -> List[Tuple[int, int]]:
        return (
            self.left.edges()
            + self.right.edges()
            + [(self.left.output_id(), self.right.input_id())]
        )

    def nodes(self) -> Dict[int, str]:
        return {**self.left.nodes(), **self.right.nodes()}

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[V]:
        async for output in self.right.process(self.left.process(input)):
            yield output


class UnchunkingAgent(Agent[List[T], T]):
    def __init__(self):
        super().__init__("unchunk")

    async def _process(self, input: AsyncIterator[List[T]]) -> AsyncIterator[T]:
        async for elem in input:
            for item in elem:
                yield item


class ChunkingAgent(Agent[T, List[T]]):
    def __init__(self, chunk_size: int):
        super().__init__(f"chunk ({chunk_size})")
        self.chunk_size = chunk_size

    async def _process(self, input: AsyncIterator[T]) -> AsyncIterator[List[T]]:
        # Get the output generator from the underlying agent
        chunk: List[T] = []
        async for elem in input:
            chunk.append(elem)
            if len(chunk) == self.chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk  # Yield any remaining items as the last chunk


class CacheAgentBase(Generic[Input, Output]):
    """
    Base class that handles caching logic for agents.
    """

    def __init__(self, directory: str, batch_size: int = 10):
        self.filename = directory
        self.batch_size = batch_size
        self.pending_writes: List[Tuple[Input, Any]] = []
        # Ensure the directory exists
        dir_name = os.path.dirname(self.filename) or "."
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # Initialize the Diskcache cache
        self.cache = diskcache.Cache(self.filename)

    def get_cached_output(self, input: Input) -> Optional[Any]:
        return self.cache.get(input)

    def set_cached_output(self, input: Input, output: Any):
        self.pending_writes.append((input, output))
        if len(self.pending_writes) >= self.batch_size:
            start_time = time.time()
            # Write all pending items to cache
            for cached_input, cached_output in self.pending_writes:
                self.cache[cached_input] = cached_output
            end_time = time.time()
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Time taken to write {len(self.pending_writes)} items to cache: {end_time-start_time} for {self.filename}"
            )
            self.pending_writes = []


class CacheStatelessAgent(StatelessAgent[Input, Output]):
    """
    Caching agent for StatelessAgents.
    """

    def __init__(
        self, agent: StatelessAgent[Input, Output], filename: str, batch_size: int = 10
    ):
        super().__init__(name=f"Cache")
        self.agent = agent
        self.cache_base = CacheAgentBase[Input, Any](filename, batch_size)

    def nodes(self) -> Dict[int, str]:
        return {**self.agent.nodes(), self.id: self.name}

    def edges(self) -> List[Tuple[int, int]]:
        return [
            (self.id, self.agent.id),
            (self.agent.id, self.id),
            *self.agent.edges(),
        ]

    async def process_element(self, input: Input) -> AsyncIterator[Output]:
        cached_output = self.cache_base.get_cached_output(input)
        if cached_output is not None:
            # Yield cached outputs
            for output in cached_output:
                yield output
        else:
            outputs = []
            # Process and cache outputs
            async for output in self.agent.process_element(input):
                outputs.append(output)
                yield output
            self.cache_base.set_cached_output(input, outputs)


class CacheMapAgent(MapAgent[Input, Output]):
    """
    Caching agent for MapAgents.
    """

    def __init__(
        self, agent: MapAgent[Input, Output], filename: str, batch_size: int = 10
    ):
        super().__init__(name=f"Cache")
        self.agent = agent
        self.cache_base = CacheAgentBase[Input, Output](filename, batch_size)

    def edges(self) -> List[Tuple[int, int]]:
        return [
            (self.id, self.agent.id),
            (self.agent.id, self.id),
            *self.agent.edges(),
        ]

    def nodes(self) -> Dict[int, str]:
        return {**self.agent.nodes(), self.id: self.name}

    async def handle(self, input: Input) -> Output:
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] In CacheMapAgent handle")
        # print(input)
        cached_output = self.cache_base.get_cached_output(input)
        if cached_output is not None:
            return cached_output
        else:
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Now in cache calling self.agent.handle")
            output = await self.agent.handle(input)
            # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Now in cache calling self.cache_base.set_cached_output")
            self.cache_base.set_cached_output(input, output)
            return output
