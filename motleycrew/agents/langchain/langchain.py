from __future__ import annotations

from typing import Any, Optional, Sequence

from langchain.agents import AgentExecutor
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_core.runnables.history import RunnableWithMessageHistory, GetSessionHistoryCallable
from langchain_core.prompts.chat import ChatPromptTemplate

from motleycrew.agents.mixins import LangchainOutputHandlingAgentMixin
from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.common import MotleyAgentFactory
from motleycrew.common import MotleySupportedTool, logger
from motleycrew.tracking import add_default_callbacks_to_langchain_config


class LangchainMotleyAgent(MotleyAgentParent, LangchainOutputHandlingAgentMixin):
    """MotleyCrew wrapper for Langchain agents."""

    def __init__(
        self,
        description: str | None = None,
        name: str | None = None,
        prompt_prefix: str | ChatPromptTemplate | None = None,
        agent_factory: MotleyAgentFactory[AgentExecutor] | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        force_output_handler: bool = False,
        chat_history: bool | GetSessionHistoryCallable = True,
        input_as_messages: bool = False,
        runnable_config: RunnableConfig | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            description: Description of the agent.

                Unlike the prompt prefix, it is not included in the prompt.
                The description is only used for describing the agent's purpose
                when giving it as a tool to other agents.

            name: Name of the agent.
                The name is used for identifying the agent when it is given as a tool
                to other agents, as well as for logging purposes.

                It is not included in the agent's prompt.

            prompt_prefix: Prefix to the agent's prompt.
                Can be used for providing additional context, such as the agent's role or backstory.

            agent_factory: Factory function to create the agent.
                The factory function should accept a dictionary of tools and return
                an AgentExecutor instance.

                See :class:`motleycrew.common.types.MotleyAgentFactory` for more details.

                Alternatively, you can use the :meth:`from_agent` method
                to wrap an existing AgentExecutor.

            tools: Tools to add to the agent.

            force_output_handler: Whether to force the agent to return through an output handler.
                If True, at least one tool must have return_direct set to True.

            chat_history: Whether to use chat history or not.
                If `True`, uses `InMemoryChatMessageHistory`.
                If a callable is passed, it is used to get the chat history by session_id.

                See :class:`langchain_core.runnables.history.RunnableWithMessageHistory`
                for more details.

            input_as_messages: Whether the agent expects a list of messages as input instead of a single string.

            runnable_config: Default Langchain config to use when invoking the agent.
                It can be used to add callbacks, metadata, etc.

            verbose: Whether to log verbose output.
        """
        super().__init__(
            prompt_prefix=prompt_prefix,
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            force_output_handler=force_output_handler,
            verbose=verbose,
        )

        if chat_history is True:
            chat_history = InMemoryChatMessageHistory()
            self.get_session_history_callable = lambda _: chat_history
        else:
            self.get_session_history_callable = chat_history

        self.input_as_messages = input_as_messages
        self.runnable_config = runnable_config

        self._create_agent_error_tool()

    def materialize(self):
        """Materialize the agent and wrap it in RunnableWithMessageHistory if needed."""
        if self.is_materialized:
            return

        super().materialize()
        assert isinstance(self._agent, AgentExecutor)

        if self.get_output_handlers():
            assert self._agent_error_tool
            self._agent.tools += [self._agent_error_tool]

            object.__setattr__(
                self._agent.agent, "plan", self.agent_plan_decorator(self._agent.agent.plan)
            )

            object.__setattr__(
                self._agent,
                "_take_next_step",
                self.take_next_step_decorator(self._agent._take_next_step),
            )

            for tool in self.agent.tools:
                if tool.return_direct:
                    object.__setattr__(
                        tool,
                        "_run",
                        self._run_tool_direct_decorator(tool._run),
                    )

                    object.__setattr__(
                        tool,
                        "run",
                        self.run_tool_direct_decorator(tool.run),
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
        config = merge_configs(self.runnable_config, config)
        prompt = self.prepare_for_invocation(input=input, prompt_as_messages=self.input_as_messages)

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
        description: str | None = None,
        prompt_prefix: str | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        runnable_config: RunnableConfig | None = None,
        verbose: bool = False,
    ) -> "LangchainMotleyAgent":
        """Create a LangchainMotleyAgent from a :class:`langchain.agents.AgentExecutor` instance.

        Using this method, you can wrap an existing AgentExecutor
        without providing a factory function.

        Args:
            agent: AgentExecutor instance to wrap.

            prompt_prefix: Prefix to the agent's prompt.
                Can be used for providing additional context, such as the agent's role or backstory.

            description: Description of the agent.

                Unlike the prompt prefix, it is not included in the prompt.
                The description is only used for describing the agent's purpose
                when giving it as a tool to other agents.

            tools: Tools to add to the agent.

            runnable_config: Default Langchain config to use when invoking the agent.
                It can be used to add callbacks, metadata, etc.

            verbose: Whether to log verbose output.
        """
        # TODO: do we really need to unite the tools implicitly like this?
        # TODO: confused users might pass tools both ways at the same time
        # TODO: and we will silently unite them, which can have side effects (e.g. doubled tools)
        # TODO: besides, getting tools can be tricky for other frameworks (e.g. LlamaIndex)
        if tools or agent.tools:
            tools = list(tools or []) + list(agent.tools or [])

        wrapped_agent = LangchainMotleyAgent(
            prompt_prefix=prompt_prefix,
            description=description,
            tools=tools,
            runnable_config=runnable_config,
            verbose=verbose,
        )
        wrapped_agent._agent = agent
        return wrapped_agent
