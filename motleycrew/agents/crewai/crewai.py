""" Module description """

from typing import Any, Optional, Sequence

from langchain_core.runnables import RunnableConfig

from motleycrew.agents.crewai import CrewAIAgentWithConfig
from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.common import MotleyAgentFactory
from motleycrew.common import MotleySupportedTool
from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.tools import MotleyTool
from motleycrew.tracking import add_default_callbacks_to_langchain_config

try:
    from crewai import Task as CrewAI__Task
except ImportError:
    pass


class CrewAIMotleyAgentParent(MotleyAgentParent):
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

        if output_handler:
            raise NotImplementedError(
                "Output handler is not supported for CrewAI agents "
                "because of the specificity of CrewAi's prompts."
            )

        ensure_module_is_installed("crewai")
        super().__init__(
            description=goal,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            verbose=verbose,
        )

    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Description

        Args:
            input (dict):
            config (:obj:`RunnableConfig`, optional):
            **kwargs:

        Returns:
            Any:
        """
        prompt = self.prepare_for_invocation(input=input)

        langchain_tools = [tool.to_langchain_tool() for tool in self.tools.values()]
        config = add_default_callbacks_to_langchain_config(config)

        crewai_task = CrewAI__Task(description=prompt)

        output = self.agent.execute_task(
            task=crewai_task,
            context=input.get("context"),
            tools=langchain_tools,
            config=config,
        )
        return output

    def materialize(self):
        super().materialize()

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
