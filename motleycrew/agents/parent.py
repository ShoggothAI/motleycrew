from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Optional,
    Sequence,
    Any,
    Union,
)

from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.tools import Tool
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.agents.output_handler import MotleyOutputHandler
from motleycrew.common import MotleyAgentFactory, MotleySupportedTool
from motleycrew.common import logger, Defaults
from motleycrew.common.exceptions import (
    AgentNotMaterialized,
    CannotModifyMaterializedAgent,
    InvalidOutput,
    OutputHandlerMaxIterationsExceeded,
)
from motleycrew.tools import MotleyTool

if TYPE_CHECKING:
    from motleycrew import MotleyCrew


class DirectOutput(BaseException):
    """Auxiliary exception to return the agent output directly through the output handler.

    When the output handler returns an output, this exception is raised with the output.
    It is then handled by the agent, who should gracefully return the output to the user.
    """

    def __init__(self, output: Any):
        self.output = output


class MotleyAgentParent(MotleyAgentAbstractParent, ABC):
    """Parent class for all motleycrew agents.

    This class is abstract and should be subclassed by all agents in motleycrew.

    In most cases, it's better to use one of the specialized agent classes,
    such as LangchainMotleyAgent or LlamaIndexMotleyAgent, which provide various
    useful features, such as observability and output handling, out of the box.

    If you need to create a custom agent, subclass this class and implement the `invoke` method.
    """

    def __init__(
        self,
        prompt_prefix: str | ChatPromptTemplate | None = None,
        description: str | None = None,
        name: str | None = None,
        agent_factory: MotleyAgentFactory | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        output_handler: MotleySupportedTool | None = None,
        verbose: bool = False,
        agent_name: str | None = None,
    ):
        """
        Args:
            prompt_prefix: Prefix to the agent's prompt.
                Can be used for providing additional context, such as the agent's role or backstory.
            description: Description of the agent.

                Unlike the prompt prefix, it is not included in the prompt.
                The description is only used for describing the agent's purpose
                when giving it as a tool to other agents.
            name: Name of the agent.
                The name is used for identifying the agent when it is given as a tool
                to other agents, as well as for logging purposes.

                It is not included in the agent's prompt.
            agent_factory: Factory function to create the agent.
                The factory function should accept a dictionary of tools and return the agent.
                It is usually called right before the agent is invoked for the first time.

                See :class:`motleycrew.common.types.MotleyAgentFactory` for more details.
            tools: Tools to add to the agent.
            output_handler: Output handler for the agent.
            verbose: Whether to log verbose output.
        """
        self.name = name or description
        self.description = description  # becomes tool description
        self.prompt_prefix = prompt_prefix
        self.agent_factory = agent_factory
        self.tools: dict[str, MotleyTool] = {}
        self.output_handler = output_handler
        self.verbose = verbose
        self.crew: MotleyCrew | None = None

        self._agent = None

        if tools:
            self.add_tools(tools)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"

    def __str__(self):
        return self.__repr__()

    def compose_prompt(
        self, input_dict: dict, prompt: ChatPromptTemplate | str, as_messages: bool = False
    ) -> Union[str, list[BaseMessage]]:
        """Compose the agent's prompt from the prompt prefix and the provided prompt.

        Args:
            input_dict: The input dictionary to the agent.
            prompt: The prompt to be added to the agent's prompt.
            as_messages: Whether the prompt should be returned as a Langchain messages list instead of a single string.

        Returns:
            The composed prompt.
        """
        # TODO: always cast description and prompt to ChatPromptTemplate first?
        prompt_messages = []

        if not self.prompt_prefix and not prompt:
            raise Exception("Cannot compose agent prompt without description or prompt")

        if self.prompt_prefix:
            if isinstance(self.prompt_prefix, ChatPromptTemplate):
                prompt_messages += self.prompt_prefix.invoke(input_dict).to_messages()

            elif isinstance(self.prompt_prefix, str):
                prompt_messages.append(SystemMessage(content=self.prompt_prefix))

            else:
                raise ValueError("Agent description must be a string or a ChatPromptTemplate")

        if prompt:
            if isinstance(prompt, ChatPromptTemplate):
                prompt_messages += prompt.invoke(input_dict).to_messages()

            elif isinstance(prompt, str):
                prompt_messages.append(HumanMessage(content=prompt))

            else:
                raise ValueError("Prompt must be a string or a ChatPromptTemplate")

        if as_messages:
            return prompt_messages

        # TODO: pass the unformatted messages list to agents that can handle it
        prompt = "\n\n".join([m.content for m in prompt_messages]) + "\n"
        return prompt

    @property
    def agent(self):
        """
        Getter for the inner agent that makes sure it's already materialized.
        The inner agent should always be accessed via this property method.
        """
        if not self.is_materialized:
            raise AgentNotMaterialized(agent_name=self.name)
        return self._agent

    @property
    def is_materialized(self):
        """Whether the agent is materialized."""
        return self._agent is not None

    def _prepare_output_handler(self) -> Optional[MotleyTool]:
        """
        Wraps the output handler in one more tool layer,
        adding the necessary stuff for returning direct output through output handler.
        """
        if not self.output_handler:
            return None

        # TODO: make this neater by constructing MotleyOutputHandler from tools?
        if isinstance(self.output_handler, MotleyOutputHandler):
            exceptions_to_handle = self.output_handler.exceptions_to_handle
            description = self.output_handler.description
            max_iterations = self.output_handler.max_iterations

        else:
            exceptions_to_handle = (InvalidOutput,)
            description = self.output_handler.description or f"Output handler"
            assert isinstance(description, str)
            description += "\n ONLY RETURN THE FINAL RESULT USING THIS TOOL!"
            max_iterations = Defaults.DEFAULT_OUTPUT_HANDLER_MAX_ITERATIONS

        iteration = 0

        def handle_agent_output(*args, **kwargs):
            assert self.output_handler
            nonlocal iteration

            try:
                iteration += 1
                output = self.output_handler._run(*args, **kwargs, config=RunnableConfig())
            except exceptions_to_handle as exc:
                if iteration <= max_iterations:
                    return f"{exc.__class__.__name__}: {str(exc)}"
                raise OutputHandlerMaxIterationsExceeded(
                    last_call_args=args,
                    last_call_kwargs=kwargs,
                    last_exception=exc,
                )

            raise DirectOutput(output)

        prepared_output_handler = StructuredTool(
            name=self.output_handler.name,
            description=description,
            func=handle_agent_output,
            args_schema=self.output_handler.args_schema,
        )

        return MotleyTool.from_langchain_tool(prepared_output_handler)

    def materialize(self):
        """Materialize the agent by creating the agent instance using the agent factory.
        This method should be called before invoking the agent for the first time.
        """

        if self.is_materialized:
            logger.info("Agent is already materialized, skipping materialization")
            return
        assert self.agent_factory, "Cannot materialize agent without a factory provided"

        output_handler = self._prepare_output_handler()

        if inspect.signature(self.agent_factory).parameters.get("output_handler"):
            logger.info("Agent factory accepts output handler, passing it")
            self._agent = self.agent_factory(tools=self.tools, output_handler=output_handler)
        elif output_handler:
            logger.info("Agent factory does not accept output handler, passing it as a tool")
            tools_with_output_handler = self.tools.copy()
            tools_with_output_handler[output_handler.name] = output_handler
            self._agent = self.agent_factory(tools=tools_with_output_handler)
        else:
            self._agent = self.agent_factory(tools=self.tools)

    def prepare_for_invocation(self, input: dict, prompt_as_messages: bool = False) -> str:
        """Prepare the agent for invocation by materializing it and composing the prompt.

        Should be called in the beginning of the agent's invoke method.

        Args:
            input: the input to the agent
            prompt_as_messages: Whether the prompt should be returned as a Langchain messages list
                instead of a single string.

        Returns:
            str: the composed prompt
        """
        self.materialize()

        if isinstance(self.output_handler, MotleyOutputHandler):
            self.output_handler.agent = self
            self.output_handler.agent_input = input

        prompt = self.compose_prompt(input, input.get("prompt"), as_messages=prompt_as_messages)
        return prompt

    def add_tools(self, tools: Sequence[MotleySupportedTool]):
        """Add tools to the agent.

        Args:
            tools: The tools to add to the agent.
        """
        if self.is_materialized and tools:
            raise CannotModifyMaterializedAgent(agent_name=self.name)

        for t in tools:
            motley_tool = MotleyTool.from_supported_tool(t)
            if motley_tool.name not in self.tools:
                self.tools[motley_tool.name] = motley_tool

    def as_tool(self) -> MotleyTool:
        """Convert the agent to a tool to be used by other agents via delegation.

        Returns:
            The tool representation of the agent.
        """

        if not self.description:
            raise ValueError("Agent must have a description to be called as a tool")

        def call_agent(*args, **kwargs):
            # TODO: this thing is hacky, we should have a better way to pass structured input
            if args:
                return self.invoke({"prompt": args[0]})
            if len(kwargs) == 1:
                return self.invoke({"prompt": list(kwargs.values())[0]})
            return self.invoke(kwargs)

        # To be specialized if we expect structured input
        return MotleyTool.from_langchain_tool(
            Tool(
                name=self.name.replace(
                    " ", "_"
                ).lower(),  # OpenAI doesn't accept spaces in function names
                description=self.description,
                func=call_agent,
            )
        )

    @abstractmethod
    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        pass
