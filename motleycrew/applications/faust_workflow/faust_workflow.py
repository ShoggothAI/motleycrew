import asyncio
import contextvars
from typing import Callable, Dict, List, Tuple, Type, get_type_hints

from faust import App, Channel, Record


class ExecutionContext(Record):
    execution_id: int
    step_depth: int
    step_name: str


class Event(Record, abstract=True):
    execution_context: ExecutionContext = None


def step(func):
    func._is_step = True
    type_hints = get_type_hints(func)
    func._event_types = type_hints.get("ev", type_hints.get("event", None))
    func._output_types = type_hints.get("return", None)
    return func


class FaustWorkflow:
    """
    Event-driven workflow implementation using Faust.
    """

    result_event_type: Type[Event] = None

    def __init__(self, app: App, verbose: bool = False, timeout: int = 60):
        self.app = app
        self.verbose = verbose
        self.timeout = timeout
        self.steps: Dict[Type[Event], Callable] = {}
        self.channels: Dict[Type[Event], Channel] = {}
        self._initialize_steps()

        self.execution_context = contextvars.ContextVar(
            "execution_context",
            default=ExecutionContext(execution_id=0, step_name=None, step_depth=0),
        )
        self.results_by_execution_id: Dict[int, asyncio.Future] = {}

        self.execution_history_lock = asyncio.Lock()
        self.execution_history_by_execution_id: Dict[int, List[Tuple[str, str, str]]] = {}

        self._create_final_agent()

    def _initialize_steps(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and hasattr(attr, "_is_step"):
                if attr._event_types and attr._output_types:
                    input_types = (
                        attr._event_types.__args__
                        if hasattr(attr._event_types, "__args__")
                        else [attr._event_types]
                    )
                    output_types = (
                        attr._output_types.__args__
                        if hasattr(attr._output_types, "__args__")
                        else [attr._output_types]
                    )
                    for input_type in input_types:
                        self.steps[input_type] = attr
                        self._create_agent(input_type, output_types, attr)
                else:
                    print(f"Warning: Step {attr_name} has incomplete type hints")

    def _create_agent(
        self, input_type: Type[Event], output_types: list[Type[Event]], step_func: Callable
    ):
        if input_type not in self.channels:
            self.channels[input_type] = self.app.channel()
        for output_type in output_types:
            if output_type not in self.channels:
                self.channels[output_type] = self.app.channel()

        @self.app.agent(
            self.channels[input_type], name=f"{step_func.__name__}_{input_type.__name__}"
        )
        async def agent(stream):
            async for event in stream:
                if self.verbose:
                    print(
                        f"Executing step: {step_func.__name__} with input type {input_type.__name__}"
                    )

                self.execution_context.set(
                    ExecutionContext(
                        execution_id=event.execution_context.execution_id,
                        step_depth=event.execution_context.step_depth + 1,
                        step_name=step_func.__name__,
                    )
                )
                result = await step_func(event)
                if result is None:
                    continue

                assert any(
                    isinstance(result, t) for t in output_types
                ), f"Step {step_func.__name__} did not return an event of type {output_types}"
                await self.send_event(result)

    def _create_final_agent(self):
        if self.result_event_type is None:
            return

        if self.result_event_type not in self.channels:
            self.channels[self.result_event_type] = self.app.channel()

        @self.app.agent(self.channels[self.result_event_type])
        async def final_agent(stream):
            async for event in stream:
                if isinstance(event, self.result_event_type):
                    execution_id = event.execution_context.execution_id
                    if execution_id in self.results_by_execution_id:
                        self.results_by_execution_id[execution_id].set_result(event)

    async def send_event(self, ev: Event):
        """
        Sends an event to the relevant channel.
        This method can be used by agents to emit multiple events.
        """
        if type(ev) not in self.channels:
            raise ValueError(f"No channel found for event type: {type(ev).__name__}")

        channel = self.channels[type(ev)]

        current_context = ev.execution_context
        if current_context is None:
            current_context = self.execution_context.get()

        ev.execution_context = current_context
        async with self.execution_history_lock:
            self.execution_history_by_execution_id[current_context.execution_id].append(
                (current_context.step_name, current_context.step_depth, type(ev).__name__)
            )

        await channel.send(value=ev)

        if self.verbose:
            print(f"Sent event of type {type(ev).__name__} to channel")

    async def run(self, ev: Event) -> Event:
        execution_id = self.execution_context.get().execution_id + 1
        async with self.execution_history_lock:
            self.execution_history_by_execution_id[execution_id] = []

        self.results_by_execution_id[execution_id] = asyncio.Future()

        # Start the Faust app if it's not already running
        if not self.app.started:
            await self.app.start()

        self.execution_context.set(
            ExecutionContext(execution_id=execution_id, step_depth=0, step_name=None)
        )
        await self.send_event(ev)

        # Wait for the result with a timeout
        result = await asyncio.wait_for(
            self.results_by_execution_id[execution_id], timeout=self.timeout
        )
        return result
