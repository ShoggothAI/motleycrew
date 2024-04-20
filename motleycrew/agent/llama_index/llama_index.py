from typing import Any, Optional, Sequence

from llama_index.core.agent import AgentRunner
from langchain_core.runnables import RunnableConfig

from motleycrew.agent.parent import MotleyAgentAbstractParent
from motleycrew.agent.shared import MotleyAgentParent
from motleycrew.tasks import Task
from motleycrew.common import MotleySupportedTool
from motleycrew.common import MotleyAgentFactory


class LlamaIndexMotleyAgentParent(MotleyAgentParent):
    def __init__(
        self,
        goal: str,
        name: str | None = None,
        agent_factory: MotleyAgentFactory | None = None,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False
    ):
        super().__init__(
            goal=goal,
            name=name,
            agent_factory=agent_factory,
            delegation=delegation,
            tools=tools,
            verbose=verbose
        )

    def invoke(
        self,
        task: Task | str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        self.materialize()
        self.agent: AgentRunner

        if isinstance(task, str):
            assert self.crew, "can't create a task outside a crew"
            # TODO: feed in context/task.message_history correctly
            # TODO: attach the current task, if any, as a dependency of the new task
            # TODO: this preamble should really be a decorator to be shared across agent wrappers
            task = Task(
                description=task,
                name=task,
                agent=self,
                # TODO inject the new subtask as a dep and reschedule the parent
                # TODO probably can't do this from here since we won't know if
                # there are other tasks to schedule
                crew=self.crew,
            )
        elif not isinstance(task, Task):
            # TODO: should really have a conversion function here from langchain tools to crewai tools
            raise ValueError(f"`task` must be a string or a Task, not {type(task)}")

        out = self.agent.chat(task.description)
        task.outputs = [out]
        # TODO: extract message history from agent, attach it to the task
        return task

    @staticmethod
    def from_agent(
        agent: AgentRunner,
        goal: str,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ) -> "LlamaIndexMotleyAgentParent":
        wrapped_agent = LlamaIndexMotleyAgentParent(
            goal=goal,
            delegation=delegation,
            tools=tools,
            verbose=verbose
        )
        wrapped_agent._agent = agent
        return wrapped_agent
