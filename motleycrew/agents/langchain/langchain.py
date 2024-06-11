""" Module description """

from typing import Any, Optional, Sequence, Callable

from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate


from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent

from motleycrew.tools import MotleyTool
from motleycrew.tracking import add_default_callbacks_to_langchain_config
from motleycrew.common import MotleySupportedTool
from motleycrew.common import MotleyAgentFactory
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm


class LangchainMotleyAgent(MotleyAgentParent):
    def __init__(
        self,
        description: str,
        name: str | None = None,
        agent_factory: MotleyAgentFactory | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
        with_history: bool = False,
        chat_history: BaseChatMessageHistory | None = None,
    ):
        """Description

        Args:
            description (str):
            name (:obj:`str`, optional):
            agent_factory (:obj:`MotleyAgentFactory`, optional):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            verbose (bool):
        """
        super().__init__(
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            verbose=verbose,
        )

        self.with_history = with_history
        self.chat_history = chat_history or InMemoryChatMessageHistory()

    def materialize(self):
        """Materialize the agent and wrap it in RunnableWithMessageHistory if needed."""
        if self.is_materialized:
            return

        super().materialize()
        if self.with_history:
            if isinstance(self._agent, RunnableWithMessageHistory):
                return
            self._agent = RunnableWithMessageHistory(
                runnable=self._agent,
                get_session_history=lambda _: self.chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

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

        """
        self.materialize()

        prompt = task_dict.get("prompt")
        if not prompt:
            raise ValueError("Task must have a prompt")

        config = add_default_callbacks_to_langchain_config(config)
        if self.with_history:
            config["configurable"] = config.get("configurable") or {}
            config["configurable"]["session_id"] = (
                config["configurable"].get("session_id") or "default"
            )

        result = self.agent.invoke({"input": prompt}, config, **kwargs)
        output = result.get("output")
        if output is None:
            raise Exception("Agent {} result does not contain output: {}".format(self, result))

        return output

    @staticmethod
    def from_function(
        function: Callable[..., Any],
        description: str,
        name: str | None = None,
        llm: BaseLanguageModel | None = None,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        prompt: ChatPromptTemplate | Sequence[ChatPromptTemplate] | None = None,
        require_tools: bool = False,
        with_history: bool = False,
        verbose: bool = False,
    ) -> "LangchainMotleyAgent":
        """Description

        Args:
            function (Callable):
            description (str):
            name (:obj:`str`, optional):
            llm (:obj:`BaseLanguageModel`, optional):
            delegation: (:obj:`bool`, :obj:`Sequence[MotleyAgentAbstractParent]`, optional):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            prompt (:obj:`ChatPromptTemplate`, :obj:`Sequence[ChatPromptTemplate]`, optional):
            require_tools (bool):
            verbose (bool):

        Returns:
            LangchainMotleyAgent:
        """
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if require_tools and not tools:
            raise ValueError("You must provide at least one tool to the LangchainMotleyAgent")

        def agent_factory(tools: dict[str, MotleyTool]):
            langchain_tools = [t.to_langchain_tool() for t in tools.values()]
            # TODO: feed description into the agent's prompt
            agent = function(llm=llm, tools=langchain_tools, prompt=prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=langchain_tools,
                verbose=verbose,
            )
            return agent_executor

        return LangchainMotleyAgent(
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            with_history=with_history,
            verbose=verbose,
        )

    @staticmethod
    def from_agent(
        agent: AgentExecutor,
        goal: str,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ) -> "LangchainMotleyAgent":
        """Description

        Args:
            agent (AgentExecutor):
            goal (str):
            tools(:obj:`Sequence[MotleySupportedTool]`, optional):
            verbose (bool):

        Returns:
            LangchainMotleyAgent
        """
        # TODO: do we really need to unite the tools implicitly like this?
        # TODO: confused users might pass tools both ways at the same time
        # TODO: and we will silently unite them, which can have side effects (e.g. doubled tools)
        # TODO: besides, getting tools can be tricky for other frameworks (e.g. LlamaIndex)
        if tools or agent.tools:
            tools = list(tools or []) + list(agent.tools or [])

        wrapped_agent = LangchainMotleyAgent(description=goal, tools=tools, verbose=verbose)
        wrapped_agent._agent = agent
        return wrapped_agent
