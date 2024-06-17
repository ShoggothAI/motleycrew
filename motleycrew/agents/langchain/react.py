""" Module description"""

from typing import Sequence, Optional

from langchain import hub
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.history import GetSessionHistoryCallable
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent

from motleycrew.agents.langchain import LangchainMotleyAgent
from motleycrew.tools import MotleyTool
from motleycrew.common import MotleySupportedTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm


class ReactMotleyAgent(LangchainMotleyAgent):
    def __init__(
        self,
        tools: Sequence[MotleySupportedTool],
        description: str | None = None,
        name: str | None = None,
        prompt: str | None = None,
        output_handler: MotleySupportedTool | None = None,
        chat_history: bool | GetSessionHistoryCallable = True,
        llm: BaseLanguageModel | None = None,
        verbose: bool = False,
    ):
        """Description

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

        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if not tools:
            raise ValueError("You must provide at least one tool to the LangchainMotleyAgent")

        def agent_factory(
            tools: dict[str, MotleyTool], output_handler: Optional[MotleyTool] = None
        ) -> AgentExecutor:
            langchain_tools = [t.to_langchain_tool() for t in tools.values()]
            if output_handler:
                langchain_tools.append(output_handler.to_langchain_tool())

            agent = create_react_agent(llm=llm, tools=langchain_tools, prompt=prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=langchain_tools,
                verbose=verbose,
            )
            return agent_executor

        super().__init__(
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            chat_history=chat_history,
            verbose=verbose,
        )
