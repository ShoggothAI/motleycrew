import logging
import concurrent.futures
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable, Sequence, Set, Optional
from uuid import uuid4

from motleycrew.agent.shared import MotleyAgentParent
from motleycrew.tasks import TaskRecipe, TaskGraph
from motleycrew.tool.tool import BaseTool
from motleycrew.storage import MotleyGraphStore, MotleyKuzuGraphStore
from motleycrew.tool import MotleyTool


class MotleyCrew:
    def __init__(self, graph_store: Optional[MotleyGraphStore] = None):
        self.uuid = uuid4()
        # TODO: impute number of workers or allow configurable
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.futures: Set[Future] = set()
        if graph_store is None:
            WORKING_DIR = os.path.realpath(os.path.dirname(__file__))
            from kuzu import kuzu

            DB_PATH = WORKING_DIR / "kuzu_db"
            db = kuzu.Database(DB_PATH)
            graph_store = MotleyKuzuGraphStore(
                db,
                node_table_schema={"question": "STRING", "answer": "STRING", "context": "STRING"},
            )
        self.graph_store = graph_store
        self.single_thread = os.environ.get("MC_SINGLE_THREAD", False)
        self.tools = []
        self.tasks = []

    def create_task(
        self,
        description: str,
        agent: MotleyAgentParent,
        name: Optional[str] = None,
        generate_name: bool = False,
        tools: Optional[Sequence[MotleyTool]] = None,
    ) -> TaskRecipe:
        if name is None and generate_name:
            # Call llm to generate a name
            raise NotImplementedError("Name generation not yet implemented")
        task = TaskRecipe(name=name, description=description, agent=agent, tools=tools)
        self.register_task(task)
        return task

    def run(
        self,
        tasks: Optional[Sequence[TaskRecipe]] = None,
        verbose: int = 0,  # TODO: use!
    ) -> TaskGraph:
        if tasks is None:
            tasks = self.tasks

        # TODO: order tasks recipes according to DAG
        tasks = self.order_task_recipes(tasks)

        # TODO: dispatch through call graph fix?
        for agent in self.agents:
            agent.crew = self

        if self.single_thread:
            logging.info("Running in single-thread mode")
            return self._run_sync(tasks)

        logging.info("Running in threaded mode")
        return self._run_async()

    def add_dependency(self, upstream: TaskRecipe, downstream: TaskRecipe):
        self.graph_store.create_relation(upstream, "Generic Dependency", downstream)

    def register_task(self, task: TaskRecipe):
        if task not in self.tasks:
            self.tasks.append(task)
            task.crew = self
            task.node_id = self.graph_store.create_entity(task)
            # TODO: rollback if bad?
            self.graph_store.check_cyclical_dependencies()

    def _run_sync(self, tasks: Sequence[TaskRecipe]):
        while True:
            did_something = False
            for t in tasks:
                matches = t.identify_candidates(self.graph_store)
                if len(matches) > 0:
                    this_match = matches[0]
                    logging.info(f"Dispatching task '{this_match}'")
                    this_match.set_running()
                    self.graph_store.create_entity(this_match)
                    this_match.outputs += self.assign_agent(t).invoke(matches[0])
                    this_match.set_task_done(t)
                    self.graph_store.update_entity(this_match)

                    did_something = True
                    continue
            if not did_something:
                break

    # def _run_async(self):
    #     tasks = self.task_graph
    #     self.adispatch_next_batch()
    #     while self.futures:
    #         # TODO handle errors
    #         # TODO pass results to next task
    #         done, _ = concurrent.futures.wait(
    #             self.futures, return_when=concurrent.futures.FIRST_COMPLETED
    #         )
    #
    #         for future in done:
    #             exc = future.exception()
    #             if exc:
    #                 raise exc
    #
    #             task = future.mc_task
    #             logging.info(f"Finished task '{task.name}'")
    #             self.futures.remove(future)
    #             tasks.set_task_done(task)
    #             self.adispatch_next_batch()
    #
    #     return tasks

    # def adispatch_next_batch(self):
    #     # TODO: refactor and rename
    #     next_ = self.task_graph.get_ready_tasks()
    #     for t in next_:
    #         self.task_graph.set_task_running(t)
    #         logging.info(f"Dispatching task '{t.name}'")
    #         future = self.thread_pool.submit(
    #             self.execute,
    #             t,
    #             True,
    #         )
    #         self.futures.add(future)
    #         future.mc_task = t

    def assign_agent(self, task: TaskRecipe) -> MotleyAgentParent:
        agent = task.get_agent()
        if agent is None:
            agent = spawn_agent(task)

        logging.info("Assigning task '%s' to agent '%s'", task.name, agent.name)

        tools = self.get_extra_tools(task)
        agent.add_tools(tools)

        return agent

    def get_extra_tools(self, task: TaskRecipe) -> Sequence[BaseTool]:
        # TODO: Smart tool selection goes here
        tools = []
        tools += self.tools or []
        tools += task.tools or []

        return tools


def spawn_agent(task: TaskRecipe) -> MotleyAgentParent:
    # TODO: Code to select agent from library, or auto-spawn one, goes here
    raise NotImplementedError("For now, must explicitly assign an agent to each task")
