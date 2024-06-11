""" Module description"""
from typing import Sequence

from langchain import hub
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import create_react_agent

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.agents.langchain.langchain import LangchainMotleyAgent
from motleycrew.common import MotleySupportedTool


class ReactMotleyAgent(LangchainMotleyAgent):
    def __new__(
        cls,
        tools: Sequence[MotleySupportedTool],
        description: str = "",  # gets ignored at the moment
        name: str | None = None,
        prompt: str | None = None,
        llm: BaseLanguageModel | None = None,
        verbose: bool = False,
    ):
        """ Description

        Args:
            tools (Sequence[MotleySupportedTool]):
            description (:obj:`str`, optional):
            name (:obj:`str`, optional):
            prompt (:obj:`str`, optional):
            llm (:obj:`BaseLanguageModel`, optional):
            verbose (:obj:`bool`, optional):
        """
        if prompt is None:
            # TODO: feed description into the agent's prompt
            prompt = hub.pull("hwchase17/react")
        return cls.from_function(
            description=description,
            name=name,
            llm=llm,
            tools=tools,
            prompt=prompt,
            function=create_react_agent,
            require_tools=True,
            verbose=verbose,
        )
