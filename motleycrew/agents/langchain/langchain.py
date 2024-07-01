""" Module description """

from typing import Any, Optional, Sequence

from langchain.agents import AgentExecutor
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory, GetSessionHistoryCallable

from motleycrew.agents.mixins import LangchainOutputHandlingAgentMixin
from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.common import MotleyAgentFactory
from motleycrew.common import MotleySupportedTool, logger
from motleycrew.tracking import add_default_callbacks_to_langchain_config


class LangchainMotleyAgent(MotleyAgentParent, LangchainOutputHandlingAgentMixin):
    def __init__(
        self,
        description: str | None = None,
        name: str | None = None,
        agent_factory: MotleyAgentFactory[AgentExecutor] | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        output_handler: MotleySupportedTool | None = None,
        verbose: bool = False,
        chat_history: bool | GetSessionHistoryCallable = True,
    ):
        """Description

        Args:
            description (:obj:`str`, optional):
            name (:obj:`str`, optional):
            agent_factory (:obj:`MotleyAgentFactory`, optional):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            output_handler (:obj:`MotleySupportedTool`, optional):
            verbose (bool):
            chat_history (:obj:`bool`, :obj:`GetSessionHistoryCallable`):
            Whether to use chat history or not. If `True`, uses `InMemoryChatMessageHistory`.
            If a callable is passed, it is used to get the chat history by session_id.
            See Langchain `RunnableWithMessageHistory` get_session_history param for more details.
        """
        super().__init__(
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            verbose=verbose,
        )

        self._agent_finish_blocker_tool = self._create_agent_finish_blocker_tool()

        if chat_history is True:
            chat_history = InMemoryChatMessageHistory()
            self.get_session_history_callable = lambda _: chat_history
        else:
            self.get_session_history_callable = chat_history

    def materialize(self):
        """Materialize the agent and wrap it in RunnableWithMessageHistory if needed."""
        if self.is_materialized:
            return

        super().materialize()
        assert isinstance(self._agent, AgentExecutor)

        if self.output_handler:
            self._agent.tools += [self._agent_finish_blocker_tool]

            object.__setattr__(
                self._agent.agent, "plan", self.agent_plan_decorator(self._agent.agent.plan)
            )

            object.__setattr__(
                self._agent,
                "_take_next_step",
                self.take_next_step_decorator(self._agent._take_next_step),
            )

            prepared_output_handler = None
            for tool in self.agent.tools:
                if tool.name == self.output_handler.name:
                    prepared_output_handler = tool

            object.__setattr__(
                prepared_output_handler,
                "_run",
                self._run_tool_direct_decorator(prepared_output_handler._run),
            )

            object.__setattr__(
                prepared_output_handler,
                "run",
                self.run_tool_direct_decorator(prepared_output_handler.run),
            )

        if self.get_session_history_callable:
            logger.info("Wrapping agent in RunnableWithMessageHistory")

            if isinstance(self._agent, RunnableWithMessageHistory):
                return
            self._agent = RunnableWithMessageHistory(
                runnable=self._agent,
                get_session_history=self.get_session_history_callable,
                input_messages_key="input",
                history_messages_key="chat_history",
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

        """
        prompt = self.prepare_for_invocation(input=input)

        config = add_default_callbacks_to_langchain_config(config)
        if self.get_session_history_callable:
            config["configurable"] = config.get("configurable") or {}
            config["configurable"]["session_id"] = (
                config["configurable"].get("session_id") or "default"
            )

        output = self.agent.invoke({"input": prompt}, config, **kwargs)
        output = output.get("output")
        return output

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
