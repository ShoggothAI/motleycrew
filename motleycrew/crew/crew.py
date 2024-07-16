import asyncio
import threading
import time
from typing import Collection, Generator, Optional, Any

from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.common import logger, AsyncBackend, Defaults
from motleycrew.crew.crew_threads import TaskUnitThreadPool
from motleycrew.storage import MotleyGraphStore
from motleycrew.storage.graph_store_utils import init_graph_store
from motleycrew.tasks import Task, TaskUnit, TaskUnitType
from motleycrew.tools import MotleyTool


class MotleyCrew:
    """The main class for executing tasks and orchestrating agents."""

    _loop: Optional[asyncio.AbstractEventLoop] = None

    def __init__(
        self,
        graph_store: Optional[MotleyGraphStore] = None,
        async_backend: AsyncBackend = AsyncBackend.NONE,
        num_threads: int = Defaults.DEFAULT_NUM_THREADS,
    ):
        """Initialize the crew.

        Args:
            graph_store: The graph store to use.
                If not provided, a new one will be created with default settings.

            async_backend: The type of async backend to use.
                Defaults to :obj:`AsyncBackend.NONE`, which means the crew will run synchronously.
                The other options are :obj:`AsyncBackend.ASYNCIO` and :obj:`AsyncBackend.THREADING`.

            num_threads:
                The number of threads to use when running in threaded mode.
        """
        if graph_store is None:
            graph_store = init_graph_store()
        self.graph_store = graph_store

        self.tools = []
        self.tasks = []
        self.async_backend = async_backend
        self.num_threads = num_threads

    def run(self) -> list[TaskUnit]:
        """Run the crew."""
        if self.async_backend == AsyncBackend.NONE:
            result = self._run_sync()
        elif self.async_backend == AsyncBackend.ASYNCIO:
            try:
                result = asyncio.run(self._run_async())
            except RuntimeError:
                if self._loop is None:
                    self._loop = asyncio.new_event_loop()
                    threading.Thread(target=self._loop.run_forever, daemon=True).start()

                future: asyncio.Future = asyncio.run_coroutine_threadsafe(
                    self._run_async(), self._loop
                )
                result = future.result()
        elif self.async_backend == AsyncBackend.THREADING:
            result = self._run_threaded()
        else:
            raise NotImplementedError()

        return result

    def add_dependency(self, upstream: Task, downstream: Task):
        """Add a dependency between two tasks in the graph store.

        Args:
            upstream: The upstream task.
            downstream: The downstream task.
        """
        self.graph_store.create_relation(
            upstream.node, downstream.node, label=Task.TASK_IS_UPSTREAM_LABEL
        )
        # TODO: check cyclical dependencies; rollback if bad?
        # self.check_cyclical_dependencies()

    def register_tasks(self, tasks: Collection[Task]):
        """Insert tasks into the crew's graph store.

        Args:
            tasks: The tasks to register.
        """
        for task in tasks:
            if task not in self.tasks:
                self.tasks.append(task)
                task.crew = self
                task.prepare_graph_store()
                self.graph_store.insert_node(task.node)

                self.graph_store.ensure_relation_table(
                    from_class=type(task.node),
                    to_class=type(task.node),
                    label=Task.TASK_IS_UPSTREAM_LABEL,
                )  # TODO: remove this workaround, https://github.com/kuzudb/kuzu/issues/3488
                # UPD: this was not fixed in Kuzu even after the issue was closed

    def _prepare_next_unit_for_dispatch(
        self, running_sync_tasks: set
    ) -> Generator[MotleyAgentParent, Task, TaskUnit]:
        """Retrieve and prepare the next unit for dispatch.
        Args:
            running_sync_tasks: Collection of currently running forced synchronous tasks.

        Yields:
            Agent, task, unit to be dispatched.
        """
        available_tasks = self.get_available_tasks()
        logger.info("Available tasks: %s", available_tasks)

        for task in available_tasks:
            if not task.allow_async_units and task in running_sync_tasks:
                continue

            logger.info("Processing task: %s", task)

            next_unit = task.get_next_unit()

            if next_unit is None:
                logger.info("Got no matching units for task %s", task)
                continue

            if not task.allow_async_units:
                running_sync_tasks.add(task)

            logger.info("Got a matching unit for task %s", task)
            logger.info("Processing unit: %s", next_unit)

            extra_tools = self.get_extra_tools(task)
            agent = task.get_worker(extra_tools)

            logger.info("Assigned unit %s to agent %s, dispatching", next_unit, agent)
            next_unit.set_running()
            self.add_task_unit_to_graph(task=task, unit=next_unit)
            task.on_unit_dispatch(next_unit)

            yield agent, task, next_unit

    def _handle_task_unit_completion(
        self,
        task: Task,
        unit: TaskUnit,
        result: Any,
        running_sync_tasks: set,
        done_units: list[TaskUnit],
    ):
        """Handle task unit completion.

        Args:
            task: Task object.
            unit: Task unit object.
            result: Result of the task unit.
            running_sync_tasks: Collection of currently running forced synchronous tasks.
            done_units: List of completed task units.

        """
        if task in running_sync_tasks:
            running_sync_tasks.remove(task)

        unit.output = result
        logger.info("Task unit %s completed, marking as done", unit)
        unit.set_done()
        task.on_unit_completion(unit)
        done_units.append(unit)

    def _run_sync(self) -> list[TaskUnit]:
        """Run the crew synchronously.

        Returns:
            List of completed task units.
        """
        done_units = []
        while True:
            did_something = False

            available_tasks = self.get_available_tasks()
            logger.info("Available tasks: %s", available_tasks)

            for agent, task, unit in self._prepare_next_unit_for_dispatch(set()):
                result = agent.invoke(unit.as_dict())

                self._handle_task_unit_completion(
                    task=task,
                    unit=unit,
                    result=result,
                    running_sync_tasks=set(),
                    done_units=done_units,
                )

                did_something = True

            if not did_something:
                logger.info("Nothing left to do, exiting")
                return done_units

    def _run_threaded(self) -> list[TaskUnit]:
        """Run the crew in threads.

        Returns:
            List of completed task units.
        """

        done_units = []
        running_sync_tasks = set()
        thread_pool = TaskUnitThreadPool(self.num_threads)
        try:
            while True:
                for task, unit, result in thread_pool.get_completed_task_units():
                    self._handle_task_unit_completion(
                        task=task,
                        unit=unit,
                        result=result,
                        running_sync_tasks=running_sync_tasks,
                        done_units=done_units,
                    )

                for agent, next_task, next_unit in self._prepare_next_unit_for_dispatch(
                    running_sync_tasks
                ):
                    thread_pool.add_task_unit(agent, next_task, next_unit)

                if thread_pool.is_completed:
                    logger.info("Nothing left to do, exiting")
                    return done_units

                time.sleep(Defaults.DEFAULT_EVENT_LOOP_SLEEP)
        finally:
            thread_pool.wait_and_close()

    @staticmethod
    async def _async_invoke_agent(agent: MotleyAgentParent, unit: TaskUnit) -> Any:
        return await agent.ainvoke(unit.as_dict())

    async def _run_async(self) -> list[TaskUnit]:
        """Run the crew asynchronously.

        Returns:
            List of completed task units.
        """

        done_units = []
        async_units = {}
        running_tasks = set()

        while True:
            for async_task in list(async_units.keys()):
                if async_task.done():
                    task, unit = async_units.pop(async_task)

                    self._handle_task_unit_completion(
                        task=task,
                        unit=unit,
                        result=async_task.result(),
                        running_sync_tasks=running_tasks,
                        done_units=done_units,
                    )

            for agent, next_task, next_unit in self._prepare_next_unit_for_dispatch(running_tasks):
                async_task = asyncio.create_task(MotleyCrew._async_invoke_agent(agent, next_unit))
                async_units[async_task] = (next_task, next_unit)

            if not async_units:
                logger.info("Nothing left to do, exiting")
                return done_units

            await asyncio.sleep(Defaults.DEFAULT_EVENT_LOOP_SLEEP)

    def get_available_tasks(self) -> list[Task]:
        """Get tasks that are able to dispatch units at the moment.
        These are tasks that have no upstream dependencies that are not done.

        Returns:
            List of tasks that are available.
        """
        query = (
            "MATCH (downstream:{}) "
            "WHERE NOT downstream.done "
            "AND NOT EXISTS {{MATCH (upstream:{})-[:{}]->(downstream) "
            "WHERE NOT upstream.done}} "
            "RETURN downstream"
        ).format(
            Task.NODE_CLASS.get_label(),
            Task.NODE_CLASS.get_label(),
            Task.TASK_IS_UPSTREAM_LABEL,
        )
        available_task_nodes = self.graph_store.run_cypher_query(query, container=Task.NODE_CLASS)
        return [task for task in self.tasks if task.node in available_task_nodes]

    def add_task_unit_to_graph(self, task: Task, unit: TaskUnitType):
        """Add a task unit to the graph and connect it to its task.

        Args:
            task: The task to which the unit belongs.
            unit: The unit to add.
        """
        assert isinstance(unit, task.task_unit_class)
        assert not unit.done

        self.graph_store.upsert_triplet(
            from_node=unit,
            to_node=task.node,
            label=task.task_unit_belongs_label,
        )

    def get_extra_tools(self, task: Task) -> list[MotleyTool]:
        """Get tools that should be added to the agent for a given task.

        Args:
            task: The task for which to get extra tools.
        """
        # TODO: Smart tool selection goes here
        tools = []
        tools += self.tools or []

        return tools

    def check_cyclical_dependencies(self):
        pass  # TODO
