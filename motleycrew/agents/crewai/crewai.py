""" Module description """

from typing import Any, Optional, Sequence, Callable

from pydantic.v1.fields import ModelField
from pydantic.v1.config import BaseConfig
from langchain_core.runnables import RunnableConfig

from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.agents.crewai import CrewAIAgentWithConfig
from motleycrew.common import MotleySupportedTool
from motleycrew.common import MotleyAgentFactory
from motleycrew.tracking import add_default_callbacks_to_langchain_config
from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.agents.mixins import LangchainOutputHandlerMixin

try:
    from crewai import Task as CrewAI__Task
except ImportError:
    pass


class CrewAIMotleyAgentParent(MotleyAgentParent, LangchainOutputHandlerMixin):
    def __init__(
        self,
        goal: str,
        name: str | None = None,
        agent_factory: MotleyAgentFactory[CrewAIAgentWithConfig] | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        output_handler: MotleySupportedTool | None = None,
        verbose: bool = False,
    ):
        """Description

        Args:
            goal (str):
            name (:obj:`str`, optional):
            agent_factory (:obj:`MotleyAgentFactory`, optional):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional:
            verbose (bool):
        """
        ensure_module_is_installed("crewai")
        super().__init__(
            description=goal,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            verbose=verbose,
        )

        if self.output_handler:
            output_handler_tool = self._prepare_output_handler()
            self.tools[output_handler_tool.name] = output_handler_tool

    def invoke(
        self,
        task_dict: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Description

        Args:
            task_dict (dict):
            config (:obj:`RunnableConfig`, optional):
            **kwargs:

        Returns:
            Any:
        """
        self.materialize()
        prompt = self.compose_prompt(task_dict, task_dict.get("prompt"))

        langchain_tools = [tool.to_langchain_tool() for tool in self.tools.values()]
        config = add_default_callbacks_to_langchain_config(config)

        crewai_task = CrewAI__Task(description=prompt)

        output = self.agent.execute_task(
            task=crewai_task, context=task_dict.get("context"), tools=langchain_tools, config=config
        )
        return output

    def _create_agent_executor_decorator(self):
        """Decorator adding logic for working with output_handler when creating agent_executor"""

        def decorator(func: Callable):
            def wrapper(tools=None):
                result = func(tools)

                object.__setattr__(
                    self._agent.agent_executor.agent,
                    "plan",
                    self.agent_plane_decorator()(self._agent.agent_executor.agent.plan),
                )

                object.__setattr__(
                    self._agent.agent_executor,
                    "_take_next_step",
                    self.take_next_step_decorator()(self._agent.agent_executor._take_next_step),
                )
                return result

            return wrapper

        return decorator

    def materialize(self):
        super().materialize()

        if self.output_handler:
            object.__setattr__(
                self._agent,
                "create_agent_executor",
                self._create_agent_executor_decorator()(self._agent.create_agent_executor),
            )

    # TODO: what do these do?
    def set_cache_handler(self, cache_handler: Any) -> None:
        """Description

        Args:
            cache_handler (Any):

        Returns:
            None:
        """
        return self.agent.set_cache_handler(cache_handler)

    def set_rpm_controller(self, rpm_controller: Any) -> None:
        """Description

        Args:
            rpm_controller (Any):

        Returns:
            None:
        """
        return self.agent.set_rpm_controller(rpm_controller)

    @staticmethod
    def from_agent(
        agent: CrewAIAgentWithConfig,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ) -> "CrewAIMotleyAgentParent":
        """Description

        Args:
            agent (CrewAIAgentWithConfig):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            verbose (bool):

        Returns:
            CrewAIMotleyAgentParent:
        """
        if tools or agent.tools:
            tools = list(tools or []) + list(agent.tools or [])

        wrapped_agent = CrewAIMotleyAgentParent(
            goal=agent.goal,
            name=agent.role,
            tools=tools,
            verbose=verbose,
        )
        wrapped_agent._agent = agent
        return wrapped_agent
