""" Module description """

from typing import Any, Optional, Sequence, Callable, Union

from langchain.agents import AgentExecutor
from langchain.agents.agent import RunnableAgent
from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory, GetSessionHistoryCallable
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.tools import Tool


from motleycrew.agents.parent import MotleyAgentParent

from motleycrew.tools import MotleyTool
from motleycrew.tracking import add_default_callbacks_to_langchain_config
from motleycrew.common import MotleySupportedTool, logger
from motleycrew.common import MotleyAgentFactory
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm


class LangchainMotleyAgent(MotleyAgentParent):
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

        if chat_history is True:
            chat_history = InMemoryChatMessageHistory()
            self.get_session_history_callable = lambda _: chat_history
        else:
            self.get_session_history_callable = chat_history

        self._agent_finish_blocker_tool = self._create_agent_finish_blocker_tool()

    def _create_agent_finish_blocker_tool(self) -> BaseTool:
        """Create a tool that will force the agent to retry if it attempts to return the output
        bypassing the output handler.
        """

        def create_agent_finish_blocking_message(input: Any) -> str:
            return f"{input}\n\nYou must use {self.output_handler.name} to return the final output."

        return Tool.from_function(
            name="agent_finish_blocker",
            description="",
            func=create_agent_finish_blocking_message,
        )

    def _block_agent_finish(self, input: Any):
        """Intercept AgentFinish for forcing output via output handler.
        If the agent attempts to return the output bypassing the output handler,
        a tool call to the agent_finish_blocker_tool will be made
        so that one more AgentExecutor iteration is forced.
        """
        if isinstance(input, AgentFinish) and self.output_handler:
            return [
                AgentAction(
                    tool=self._agent_finish_blocker_tool.name,
                    tool_input={"input": input.return_values},
                    log="\nDetected AgentFinish, blocking it to force output via output handler.\n",
                )
            ]
        return input

    def materialize(self):
        """Materialize the agent and wrap it in RunnableWithMessageHistory if needed."""
        if self.is_materialized:
            return

        super().materialize()
        assert isinstance(self._agent, AgentExecutor)

        if self.output_handler:
            self._agent.tools += [self._agent_finish_blocker_tool]

            if isinstance(self._agent.agent, Runnable):
                logger.info("Patching agent to force output via output handler")
                agent_with_forced_output_via_handler = self._agent.agent | RunnableLambda(
                    self._block_agent_finish
                )
                self._agent.agent = agent_with_forced_output_via_handler

            elif isinstance(self._agent.agent, RunnableAgent):
                logger.info("Patching agent to force output via output handler")
                agent_with_forced_output_via_handler = self._agent.agent.runnable | RunnableLambda(
                    self._block_agent_finish
                )
                self._agent.agent.runnable = agent_with_forced_output_via_handler

            else:
                logger.warning(
                    "Agent is not a Runnable, cannot patch it to force output via output handler"
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

        caught_direct_output, output = self._run_and_catch_output(
            action=self.agent.invoke,
            action_args=(
                {"input": prompt},
                config,
            ),
            action_kwargs=kwargs,
        )

        if not caught_direct_output:
            output = output.get("output")  # unpack native Langchain output

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
