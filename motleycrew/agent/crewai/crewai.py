from typing import Any, Optional, Sequence, Callable, Dict, List

from crewai import Agent
from langchain_core.runnables import RunnableConfig
from langchain.tools.render import render_text_description
from pydantic import Field

from motleycrew.agent.parent import MotleyAgentAbstractParent
from motleycrew.agent.shared import MotleyAgentParent
from motleycrew.tasks import Task
from motleycrew.tool import MotleyTool
from motleycrew.common import MotleySupportedTool
from motleycrew.common import MotleyAgentFactory
from motleycrew.common.utils import to_str
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.tracking import add_default_callbacks_to_langchain_config


class CrewAIAgentWithConfig(Agent):

    def execute_task(
            self,
            task: Any,
            context: Optional[str] = None,
            tools: Optional[List[Any]] = None,
            config: Optional[RunnableConfig] = None
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.
            config: Runnable config
        Returns:
            Output of the agent
        """
        task_prompt = task.prompt()

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        tools = self._parse_tools(tools or self.tools)
        self.create_agent_executor(tools=tools)
        self.agent_executor.tools = tools
        self.agent_executor.task = task
        self.agent_executor.tools_description = render_text_description(tools)
        self.agent_executor.tools_names = self.__tools_names(tools)

        result = self.agent_executor.invoke(
            {
                "input": task_prompt,
                "tool_names": self.agent_executor.tools_names,
                "tools": self.agent_executor.tools_description,
            },
            config=config
        )["output"]

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result

    @staticmethod
    def __tools_names(tools) -> str:
        return ", ".join([t.name for t in tools])


class CrewAIMotleyAgentParent(MotleyAgentParent):
    def __init__(
        self,
        goal: str,
        name: str | None = None,
        agent_factory: MotleyAgentFactory | None = None,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            goal=goal,
            name=name,
            agent_factory=agent_factory,
            delegation=delegation,
            tools=tools,
            verbose=verbose,
        )

    def invoke(
        self,
        task: Task | str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        self.materialize()
        self.agent: Agent

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

        langchain_tools = [tool.to_langchain_tool() for tool in self.tools.values()]
        config = add_default_callbacks_to_langchain_config(config)

        out = self.agent.execute_task(
            task, to_str(task.message_history), tools=langchain_tools, config=config
        )
        task.outputs = [out]
        # TODO: extract message history from agent, attach it to the task
        return task

    # TODO: what do these do?
    def set_cache_handler(self, cache_handler: Any) -> None:
        return self.agent.set_cache_handler(cache_handler)

    def set_rpm_controller(self, rpm_controller: Any) -> None:
        return self.agent.set_rpm_controller(rpm_controller)

    @staticmethod
    def from_crewai_params(
        role: str,
        goal: str,
        backstory: str,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        llm: Optional[Any] = None,
        verbose: bool = False,
    ) -> "CrewAIMotleyAgentParent":
        if tools is None:
            tools = []

        if llm is None:
            # CrewAI uses Langchain LLMs by default
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        def agent_factory(tools: dict[str, MotleyTool]):
            langchain_tools = [t.to_langchain_tool() for t in tools.values()]
            agent = CrewAIAgentWithConfig(
                role=role,
                goal=goal,
                backstory=backstory,
                verbose=verbose,
                allow_delegation=False,  # Delegation handled by MotleyAgentParent
                tools=langchain_tools,
                llm=llm,
            )
            return agent

        return CrewAIMotleyAgentParent(
            goal=goal,
            name=role,
            agent_factory=agent_factory,
            delegation=delegation,
            tools=tools,
            verbose=verbose,
        )

    @staticmethod
    def from_agent(
        agent: CrewAIAgentWithConfig,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ) -> "CrewAIMotleyAgentParent":
        if tools or agent.tools:
            tools = list(tools or []) + list(agent.tools or [])

        wrapped_agent = CrewAIMotleyAgentParent(
            goal=agent.goal,
            name=agent.role,
            delegation=delegation,
            tools=tools,
            verbose=verbose,
        )
        wrapped_agent._agent = agent
        return wrapped_agent
