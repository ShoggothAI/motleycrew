from typing import Sequence

from langchain import hub
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import create_react_agent

from motleycrew.agent.parent import MotleyAgentAbstractParent
from motleycrew.agent.langchain.langchain import LangchainMotleyAgentParent
from motleycrew.common import MotleySupportedTool


class ReactMotleyAgent(LangchainMotleyAgentParent):
    def __new__(
        cls,
        tools: Sequence[MotleySupportedTool],
        goal: str = "",  # gets ignored at the moment
        prompt: str | None = None,
        llm: BaseLanguageModel | None = None,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        verbose: bool = False,
    ):
        if prompt is None:
            # TODO: feed goal into the agent's prompt
            prompt = hub.pull("hwchase17/react")
        return cls.from_function(
            goal=goal,
            llm=llm,
            delegation=delegation,
            tools=tools,
            prompt=prompt,
            function=create_react_agent,
            require_tools=True,
            verbose=verbose,
        )
