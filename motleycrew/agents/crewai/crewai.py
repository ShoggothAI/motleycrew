""" Module description """

from typing import Any, Optional, Sequence

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field

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
        prompt_prefix: str | None = None,
        description: str | None = None,
        name: str | None = None,
        agent_factory: MotleyAgentFactory[CrewAIAgentWithConfig] | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        output_handler: MotleySupportedTool | None = None,
        verbose: bool = False,
    ):
        """Description

        Args:
            goal (str):
            prompt_prefix (:obj:`str`, optional):
            description (:obj:`str`, optional):
            name (:obj:`str`, optional):
            agent_factory (:obj:`MotleyAgentFactory`, optional):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional:
            verbose (bool):
        """

        if output_handler:
            raise NotImplementedError(
                "Output handler is not supported for CrewAI agents "
                "because of the specificity of CrewAI's prompts."
            )

        ensure_module_is_installed("crewai")
        super().__init__(
            prompt_prefix=prompt_prefix,
            description=description or goal,
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

        additional_params = input.get("additional_params") or {}
        expected_output = additional_params.get("expected_output")
        if not expected_output:
            raise ValueError("Expected output is required for CrewAI tasks")

        crewai_task = CrewAI__Task(
            description=prompt,
            expected_output=expected_output,
        )

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
            prompt_prefix=agent.prompt_prefix,
            description=agent.description,
            name=agent.role,
            tools=tools,
            verbose=verbose,
        )
        wrapped_agent._agent = agent
        return wrapped_agent

    def as_tool(self) -> MotleyTool:
        if not self.description:
            raise ValueError("Agent must have a description to be called as a tool")

        class CrewAIAgentInputSchema(BaseModel):
            prompt: str = Field(..., description="Prompt to be passed to the agent")
            expected_output: str = Field(
                ..., description="Expected output of the agent"
            )

        def call_agent(prompt: str, expected_output: str):
            return self.invoke(
                {
                    "prompt": prompt,
                    "additional_params": {"expected_output": expected_output},
                }
            )

        # To be specialized if we expect structured input
        return MotleyTool.from_langchain_tool(
            StructuredTool(
                name=self.name.replace(
                    " ", "_"
                ).lower(),  # OpenAI doesn't accept spaces in function names
                description=self.description,
                func=call_agent,
                args_schema=CrewAIAgentInputSchema,
            )
        )
