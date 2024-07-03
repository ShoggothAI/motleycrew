""" Module description """
from typing import Optional, Any, Sequence

from motleycrew.tools import MotleyTool
from motleycrew.common import MotleySupportedTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.agents.crewai import CrewAIMotleyAgentParent
from motleycrew.agents.crewai import CrewAIAgentWithConfig


class CrewAIMotleyAgent(CrewAIMotleyAgentParent):
    def __init__(
            self,
            role: str,
            goal: str,
            backstory: str,
            prompt_prefix: str | None = None,
            description: str | None = None,
            delegation: bool = False,
            tools: Sequence[MotleySupportedTool] | None = None,
            llm: Optional[Any] = None,
            output_handler: MotleySupportedTool | None = None,
            verbose: bool = False,
    ):
        """ Description

        Args:
            role (str):
            goal (str):
            backstory (str):
            prompt_prefix (str):
            description (str, optional):
            delegation (bool):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            llm (:obj:'Any', optional):
            verbose (bool):
        """
        if tools is None:
            tools = []

        if llm is None:
            # CrewAI uses Langchain LLMs by default
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if delegation:
            raise ValueError(
                "'delegation' is not supported, pass the agents you want to delegate to as tools instead."
            )

        def agent_factory(tools: dict[str, MotleyTool]):
            langchain_tools = [t.to_langchain_tool() for t in tools.values()]
            agent = CrewAIAgentWithConfig(
                role=role,
                goal=goal,
                backstory=backstory,
                verbose=verbose,
                allow_delegation=False,
                tools=langchain_tools,
                llm=llm,
            )
            return agent

        super().__init__(
            goal=goal,
            prompt_prefix=prompt_prefix,
            description=description,
            name=role,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            verbose=verbose,
        )
