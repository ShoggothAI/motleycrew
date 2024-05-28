""" Module description """
from typing import Optional, Any, Sequence

from motleycrew.tools import MotleyTool
from motleycrew.common import MotleySupportedTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.agents.crewai import CrewAIMotleyAgentParent
from motleycrew.agents.crewai import CrewAIAgentWithConfig


class CrewAIMotleyAgent(CrewAIMotleyAgentParent):
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        delegation: bool = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        llm: Optional[Any] = None,
        verbose: bool = False,
    ):
        """ Description

        Args:
            role (str):
            goal (str):
            backstory (str):
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
            name=role,
            agent_factory=agent_factory,
            tools=tools,
            verbose=verbose,
        )
