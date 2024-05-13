import logging
import concurrent.futures
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Collection, Sequence, Set, Optional
from uuid import uuid4

from langchain_core.runnables import Runnable

from motleycrew.agent.shared import MotleyAgentParent
from motleycrew.tasks import TaskRecipe, Task, SimpleTaskRecipe
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
            import kuzu

            DB_PATH = os.path.join(WORKING_DIR, "kuzu_db")
            db = kuzu.Database(DB_PATH)
            graph_store = MotleyKuzuGraphStore(db)
        self.graph_store = graph_store
        self.single_thread = os.environ.get("MC_SINGLE_THREAD", False)
        self.tools = []
        self.task_recipes = []

    def create_simple_task(
        self,
        description: str,
        agent: MotleyAgentParent,
        name: Optional[str] = None,
        generate_name: bool = False,
        tools: Optional[Sequence[MotleyTool]] = None,
    ) -> SimpleTaskRecipe:
        """
        Basic method for creating a simple task recipe
        """
        if name is None and generate_name:
            # Call llm to generate a name
            raise NotImplementedError("Name generation not yet implemented")
        task_recipe = SimpleTaskRecipe(name=name, description=description, agent=agent, tools=tools)
        self.register_task_recipes([task_recipe])
        return task_recipe

    def run(
        self,
        verbose: int = 0,  # TODO: use!
    ) -> list[Task]:
        if not self.single_thread:
            logging.warning("Multithreading is not implemented yet, will run in single thread")

        return self._run_sync(verbose=verbose)

    def add_dependency(self, upstream: TaskRecipe, downstream: TaskRecipe):
        self.graph_store.create_relation(
            upstream.node, downstream.node, label=TaskRecipe.TASK_RECIPE_IS_UPSTREAM_LABEL
        )
        # # TODO: rollback if bad?
        # self.check_cyclical_dependencies()

    def register_task_recipes(self, task_recipes: Collection[TaskRecipe]):
        for task_recipe in task_recipes:
            if task_recipe not in self.task_recipes:
                self.task_recipes.append(task_recipe)
                task_recipe.crew = self
                task_recipe.node_id = self.graph_store.insert_node(task_recipe.node)

    def _run_sync(self, verbose: int = 0) -> list[Task]:
        # TODO: use the verbose arg
        done_tasks = []
        while True:
            did_something = False

            available_task_recipes = self.get_available_task_recipes()
            logging.info("Available task recipes: %s", available_task_recipes)

            for recipe in available_task_recipes:
                logging.info("Processing recipe: %s", recipe)

                matching_tasks = recipe.identify_candidates()
                logging.info("Got %s matching tasks for recipe %s", len(matching_tasks), recipe)
                if len(matching_tasks) > 0:
                    current_task = matching_tasks[0]
                    logging.info("Processing task: %s", current_task)

                    extra_tools = self.get_extra_tools(recipe)

                    agent = recipe.get_worker(extra_tools)
                    logging.info("Assigned task %s to agent %s, dispatching", current_task, agent)
                    current_task.set_running()
                    self.graph_store.insert_node(current_task)

                    # TODO: accept and handle some sort of return value? Or just the final state of the task?
                    result = agent.invoke(current_task.as_dict())
                    current_task.output = result

                    logging.info("Task %s completed, marking as done", current_task)
                    current_task.set_done()
                    recipe.register_completed_task(current_task)
                    done_tasks.append(current_task)

                    did_something = True
                    continue

            if not did_something:
                logging.info("Nothing left to do, exiting")
                return done_tasks

    def get_available_task_recipes(self) -> list[TaskRecipe]:
        query = (
            "MATCH (downstream:{}) "
            "WHERE NOT downstream.done "
            "AND NOT EXISTS {{MATCH (upstream:{})-[:{}]->(downstream) "
            "WHERE NOT upstream.done}} "
            "RETURN downstream"
        ).format(
            TaskRecipe.NODE_CLASS.get_label(),
            TaskRecipe.NODE_CLASS.get_label(),
            TaskRecipe.TASK_RECIPE_IS_UPSTREAM_LABEL,
        )
        available_task_recipe_nodes = self.graph_store.run_cypher_query(
            query, container=TaskRecipe.NODE_CLASS
        )
        return [
            recipe for recipe in self.task_recipes if recipe.node in available_task_recipe_nodes
        ]

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

    def get_extra_tools(self, task_recipe: TaskRecipe) -> list[MotleyTool]:
        # TODO: Smart tool selection goes here
        tools = []
        tools += self.tools or []
        # tools += task_recipe.tools or []

        return tools

    def check_cyclical_dependencies(self):
        pass  # TODO
