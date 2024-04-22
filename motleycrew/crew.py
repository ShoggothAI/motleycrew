import logging
import concurrent.futures
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Iterable, Sequence, Set
from uuid import uuid4

from motleycrew.agent.shared import MotleyAgentParent
from motleycrew.tasks import Task, TaskGraph
from motleycrew.tool.tool import BaseTool


class MotleyCrew:
    def __init__(self):
        self.uuid = uuid4()
        # TODO: impute number of workers or allow configurable
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.futures: Set[Future] = set()
        self.task_graph = TaskGraph()
        self.single_thread = os.environ.get("MC_SINGLE_THREAD", False)
        self.tools = []

    def run(
        self,
        agents: Sequence[MotleyAgentParent] | None = None,
        tools: Sequence[BaseTool] | None = None,
        verbose: int = 0,
    ) -> TaskGraph:
        # TODO: propagate the `verbose` argument

        self.task_graph.check_cyclical_dependencies()

        if isinstance(tools, Sequence):
            self.tools = list(tools)

        # TODO: need to specify agents both to tasks and to crew, redundant?
        self.agents = agents
        for agent in self.agents:
            agent.crew = self

        if self.single_thread:
            logging.info("Running in single-thread mode")
            return self._run_sync()

        logging.info("Running in threaded mode")
        return self._run_async()

    def _run_sync(self):
        tasks = self.task_graph
        while tasks.num_tasks_remaining():
            self.dispatch_next_batch()
        return tasks

    def _run_async(self):
        tasks = self.task_graph
        self.adispatch_next_batch()
        while self.futures:
            # TODO handle errors
            # TODO pass results to next task
            done, _ = concurrent.futures.wait(
                self.futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                exc = future.exception()
                if exc:
                    raise exc

                task = future.mc_task
                logging.info(f"Finished task '{task.name}'")
                self.futures.remove(future)
                tasks.set_task_done(task)
                self.adispatch_next_batch()

        return tasks

    def add_task(self, task: Task):
        self.task_graph.add_task(task)

    def dispatch_next_batch(self):
        next_ = self.task_graph.get_ready_tasks()
        for t in next_:
            logging.info(f"Dispatching task '{t.name}'")
            self.task_graph.set_task_running(t)
            self.execute(t, True)
            self.task_graph.set_task_done(t)

    def adispatch_next_batch(self):
        next_ = self.task_graph.get_ready_tasks()
        for t in next_:
            self.task_graph.set_task_running(t)
            logging.info(f"Dispatching task '{t.name}'")
            future = self.thread_pool.submit(
                self.execute,
                t,
                True,
            )
            self.futures.add(future)
            future.mc_task = t

    def execute(self, task: Task, return_result: bool = True):
        agent = self.assign_agent(task)
        if return_result:
            return agent.invoke(task).outputs[-1]
        else:
            raise NotImplementedError("Async task spawning not yet implemented")
            # return f"Subtask {task.name} spawned successfully"

    def assign_agent(self, task: Task) -> MotleyAgentParent:
        if isinstance(task.agent, MotleyAgentParent):
            # TODO: make a deepcopy here, perhaps via cloudpickle?
            # Agents are meant to be transient, so we don't want to modify the original
            agent = task.agent
        else:
            agent = spawn_agent(task)

        logging.info("Assigning task '%s' to agent '%s'", task.name, agent.name)

        tools = self.get_agent_tools(agent, task)
        agent.add_tools(tools)

        return agent

    def get_agent_tools(
        self, agent: MotleyAgentParent, task: Task
    ) -> Sequence[BaseTool]:
        # Task is needed later when we do smart tool selection
        # TODO: Smart tool selection goes here
        # Add the agents as tools to each other for delegation
        # later might want to get a bit fancier here to prevent endlessly-deep delegation
        # TODO: do we want to auto-include the agents from the tasks in this?
        tools = []
        tools += self.tools

        if agent.delegation is True:
            tools += [aa.as_tool() for aa in self.agents if aa != agent]
        elif isinstance(agent.delegation, Iterable):
            for aa in agent.delegation:
                tools.append(aa.as_tool())
        elif agent.delegation is False:
            pass
        else:
            raise ValueError(
                f"Invalid delegation value: {agent.delegation}, must be bool or iterable of agents"
            )

        return tools


def spawn_agent(task: Task) -> MotleyAgentParent:
    # TODO: Code to select agent from library, or auto-spawn one, goes here
    raise NotImplementedError("For now, must explicitly assign an agent to each task")
