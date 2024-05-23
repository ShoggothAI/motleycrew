from typing import Collection, Sequence, Optional
import logging
import os

from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.tasks import Task, TaskUnit, SimpleTask
from motleycrew.storage import MotleyGraphStore
from motleycrew.storage.graph_store_utils import init_graph_store
from motleycrew.tools import MotleyTool


class MotleyCrew:
    def __init__(self, graph_store: Optional[MotleyGraphStore] = None):
        if graph_store is None:
            graph_store = init_graph_store()
        self.graph_store = graph_store

        self.single_thread = os.environ.get("MC_SINGLE_THREAD", False)
        self.tools = []
        self.tasks = []

    def create_simple_task(
        self,
        description: str,
        agent: MotleyAgentParent,
        name: Optional[str] = None,
        generate_name: bool = False,
        tools: Optional[Sequence[MotleyTool]] = None,
    ) -> SimpleTask:
        """
        Basic method for creating a simple task
        """
        if name is None and generate_name:
            # Call llm to generate a name
            raise NotImplementedError("Name generation not yet implemented")
        task = SimpleTask(crew=self, name=name, description=description, agent=agent, tools=tools)
        self.register_tasks([task])
        return task

    def run(self) -> list[TaskUnit]:
        if not self.single_thread:
            logging.warning("Multithreading is not implemented yet, will run in single thread")

        return self._run_sync()

    def add_dependency(self, upstream: Task, downstream: Task):
        self.graph_store.create_relation(
            upstream.node, downstream.node, label=Task.TASK_IS_UPSTREAM_LABEL
        )
        # # TODO: rollback if bad?
        # self.check_cyclical_dependencies()

    def register_tasks(self, tasks: Collection[Task]):
        for task in tasks:
            if task not in self.tasks:
                self.tasks.append(task)
                task.crew = self
                self.graph_store.insert_node(task.node)

                self.graph_store.ensure_relation_table(
                    from_class=type(task.node),
                    to_class=type(task.node),
                    label=Task.TASK_IS_UPSTREAM_LABEL,
                )  # TODO: remove this workaround, https://github.com/kuzudb/kuzu/issues/3488

    def _run_sync(self) -> list[TaskUnit]:
        done_units = []
        while True:
            did_something = False

            available_tasks = self.get_available_tasks()
            logging.info("Available tasks: %s", available_tasks)

            for task in available_tasks:
                logging.info("Processing task: %s", task)

                next_unit = task.get_next_unit()

                if next_unit is None:
                    logging.info("Got no matching units for task %s", task)
                else:
                    logging.info("Got a matching unit for task %s", task)
                    current_unit = next_unit
                    logging.info("Processing task: %s", current_unit)

                    extra_tools = self.get_extra_tools(task)

                    agent = task.get_worker(extra_tools)
                    logging.info("Assigned unit %s to agent %s, dispatching", current_unit, agent)
                    current_unit.set_running()
                    task.register_started_unit(current_unit)

                    # TODO: accept and handle some sort of return value? Or just the final state of the task?
                    result = agent.invoke(current_unit.as_dict())
                    current_unit.output = result

                    logging.info("Task unit %s completed, marking as done", current_unit)
                    current_unit.set_done()
                    task.register_completed_unit(current_unit)
                    done_units.append(current_unit)

                    did_something = True
                    continue

            if not did_something:
                logging.info("Nothing left to do, exiting")
                return done_units

    def get_available_tasks(self) -> list[Task]:
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

    def get_extra_tools(self, task: Task) -> list[MotleyTool]:
        # TODO: Smart tool selection goes here
        tools = []
        tools += self.tools or []
        # tools += task.tools or []

        return tools

    def check_cyclical_dependencies(self):
        pass  # TODO
