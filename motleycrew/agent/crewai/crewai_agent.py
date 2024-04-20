from typing import Optional, Any, Sequence

from motleycrew.common import MotleySupportedTool
from motleycrew.agent.parent import MotleyAgentAbstractParent
from motleycrew.agent.crewai import CrewAIMotleyAgentParent


class CrewAIMotleyAgent(CrewAIMotleyAgentParent):
    def __new__(
        cls,
        role: str,
        goal: str,
        backstory: str,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        llm: Optional[Any] = None,
        verbose: bool = False,
    ):
        return cls.from_crewai_params(
            role=role,
            goal=goal,
            backstory=backstory,
            delegation=delegation,
            tools=tools,
            llm=llm,
            verbose=verbose,
        )
