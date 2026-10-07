from __future__ import annotations

import warnings
from typing import Callable, Sequence

from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, RunnableConfig
from langchain_core.runnables.history import GetSessionHistoryCallable
from langchain_core.tools import BaseTool

from motleycrew.common.utils import print_passthrough

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from motleycrew.agents.langchain import LangchainMotleyAgent
from motleycrew.agents.langchain.tool_calling_react_prompts import (
    ToolCallingReActPromptsForAnthropic,
    ToolCallingReActPromptsForOpenAI,
)
from motleycrew.common import Defaults, LLMFramework, MotleySupportedTool
from motleycrew.common.llms import init_llm
from motleycrew.tools import MotleyTool


def check_variables(prompt: ChatPromptTemplate):
    missing_vars = (
        {"agent_scratchpad", "additional_notes"}
        .difference(prompt.input_variables)
        .difference(prompt.optional_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")


def render_text_description(tools: list[BaseTool]) -> str:
    tool_strings = []
    for tool in tools:
        tool_strings.append(f"{tool.name} - {tool.description}")
    return "\n".join(tool_strings)


def get_relevant_internal_prompt(
    llm: BaseChatModel, force_output_handler: bool
) -> ChatPromptTemplate:
    if ChatAnthropic is not None and isinstance(llm, ChatAnthropic):
        prompts = ToolCallingReActPromptsForAnthropic()
    else:
        # Anthropic prompts are more specific, so we use the OpenAI prompts as the default
        prompts = ToolCallingReActPromptsForOpenAI()

    if force_output_handler:
        return prompts.prompt_template_with_output_handler
    return prompts.prompt_template_without_output_handler


def create_tool_calling_react_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    internal_prompt: ChatPromptTemplate,
    output_handlers: Sequence[BaseTool],
    force_output_handler: bool,
    intermediate_steps_processor: Callable | None = None,
) -> Runnable:
    internal_prompt = internal_prompt.partial(
        tools=render_text_description(list(tools)),
        output_handlers=(render_text_description(output_handlers) if force_output_handler else ""),
    )
    check_variables(internal_prompt)

    tools_for_llm = list(tools)
    llm_with_tools = llm.bind_tools(tools=tools_for_llm)

    if not intermediate_steps_processor:
        intermediate_steps_processor = lambda x: x

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(
                intermediate_steps_processor(x["intermediate_steps"])
            ),
            additional_notes=lambda x: x.get("additional_notes") or [],
        )
        | internal_prompt
        | RunnableLambda(print_passthrough)
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    return agent


class ReActToolCallingMotleyAgent(LangchainMotleyAgent):
    """Universal ReAct-style agent that supports tool calling.

    This agent only works with newer models that support tool calling.
    If you are using an older model, you should use
    :class:`motleycrew.agents.langchain.LegacyReActMotleyAgent` instead.
    """

    def __init__(
        self,
        tools: Sequence[MotleySupportedTool],
        description: str | None = None,
        name: str | None = None,
        prompt: str | ChatPromptTemplate | None = None,
        chat_history: bool | GetSessionHistoryCallable = True,
        force_output_handler: bool = False,
        handle_parsing_errors: bool = False,
        llm: BaseChatModel | None = None,
        stream: bool = False,
        max_iterations: int | None = Defaults.DEFAULT_REACT_AGENT_MAX_ITERATIONS,
        internal_prompt: ChatPromptTemplate | None = None,
        intermediate_steps_processor: Callable | None = None,
        runnable_config: RunnableConfig | None = None,
        verbose: bool = False,
        prompt_prefix: str | None = None,
    ):
        """
        Args:
            tools: Tools to add to the agent.
            description: Description of the agent.
            name: Name of the agent.
            prompt: Prompt for the agent.

                If a string, it will be used as a prompt.
                If a string containing f-string-style placeholders, it will be used as a prompt template.
                If a ChatPromptTemplate, it will be used as a prompt template.
            chat_history: Whether to use chat history or not.
                If `True`, uses `InMemoryChatMessageHistory`.
                If a callable is passed, it is used to get the chat history by session_id.
                See :class:`langchain_core.runnables.history.RunnableWithMessageHistory`
                for more details.
            force_output_handler: Whether to force the agent to return through an output handler.
                If True, at least one tool must have return_direct set to True.
            handle_parsing_errors: Whether to handle parsing errors.
            handle_tool_errors: Whether to handle tool errors.
                If True, `handle_tool_error` and `handle_validation_error` in all tools
                are set to True.
            llm: Language model to use.
            max_iterations: The maximum number of agent iterations.
            internal_prompt: The internal ReAct prompt to use.
                See Internal prompt section below for more on the expected input variables.
            intermediate_steps_processor: Function that modifies the intermediate steps array
                in some way before each agent iteration.
            runnable_config: Default Langchain config to use when invoking the agent.
                It can be used to add callbacks, metadata, etc.
            verbose: Whether to log verbose output.
            prompt_prefix: Deprecated. Please use the prompt argument instead.

        Internal prompt:
            Not to be confused with the `prompt` argument, which is user-facing
            and is the recommended way to explain the task to the agent.

            The internal (system) prompt contains the explanations for the low-level
            ReAct agent behavior (tool calling, reasoning, etc). Only modify this if
            you know what you are doing.

            The internal prompt must have `agent_scratchpad`, `chat_history`, and
            `additional_notes` ``MessagesPlaceholder``s.

            The default prompt slightly varies depending on the language model used.
            See :mod:`motleycrew.agents.langchain.tool_calling_react_prompts` for more details.
        """
        if prompt_prefix is not None:
            warnings.warn(
                "prompt_prefix is deprecated and will be removed in the future. "
                "Please use the prompt argument instead.",
                DeprecationWarning,
            )

            if prompt is not None:
                raise ValueError(
                    "`prompt_prefix` is deprecated; `prompt` argument is what you should use now, "
                    "optionally with a `{prompt}` placeholder. To customize the internal prompt, "
                    "use the `internal_prompt` argument."
                )

            prompt = prompt_prefix + "\n\n{prompt}"

        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        llm.bind(stream=stream)

        if not tools:
            raise ValueError("You must provide at least one tool to the ReActToolCallingAgent")

        if internal_prompt is None:
            internal_prompt = get_relevant_internal_prompt(
                llm=llm, force_output_handler=force_output_handler
            )

        def agent_factory(tools: dict[str, MotleyTool]) -> AgentExecutor:
            output_handlers_for_langchain = [
                t.to_langchain_tool() for t in tools.values() if t.return_direct
            ]
            tools_for_langchain = [t.to_langchain_tool() for t in tools.values()]

            agent = create_tool_calling_react_agent(
                llm=llm,
                tools=tools_for_langchain,
                internal_prompt=internal_prompt,
                output_handlers=output_handlers_for_langchain,
                force_output_handler=force_output_handler,
                intermediate_steps_processor=intermediate_steps_processor,
            )

            # As of langchain-core 0.3.81, verbose=True no longer prints output
            # unless callbacks are explicitly provided
            callbacks = [StdOutCallbackHandler()] if verbose else None
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools_for_langchain,
                handle_parsing_errors=handle_parsing_errors,
                verbose=verbose,
                max_iterations=max_iterations,
                stream_runnable=stream,
                callbacks=callbacks,
            )
            return agent_executor

        super().__init__(
            prompt=prompt,
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            force_output_handler=force_output_handler,
            chat_history=chat_history,
            input_as_messages=True,
            runnable_config=runnable_config,
            verbose=verbose,
        )
