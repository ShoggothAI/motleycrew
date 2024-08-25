from __future__ import annotations

from typing import Sequence, Optional, Callable

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import GetSessionHistoryCallable
from langchain_core.tools import BaseTool

from motleycrew.common.utils import print_passthrough

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

from motleycrew.agents.langchain import LangchainMotleyAgent
from motleycrew.agents.langchain.tool_calling_react_prompts import (
    ToolCallingReActPromptsForOpenAI,
    ToolCallingReActPromptsForAnthropic,
)
from motleycrew.common import LLMFramework, Defaults
from motleycrew.common import MotleySupportedTool
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


def get_relevant_default_prompt(
    llm: BaseChatModel, output_handler: Optional[MotleySupportedTool]
) -> ChatPromptTemplate:
    if ChatAnthropic is not None and isinstance(llm, ChatAnthropic):
        prompts = ToolCallingReActPromptsForAnthropic()
    else:
        # Anthropic prompts are more specific, so we use the OpenAI prompts as the default
        prompts = ToolCallingReActPromptsForOpenAI()

    if output_handler:
        return prompts.prompt_template_with_output_handler
    return prompts.prompt_template_without_output_handler


def create_tool_calling_react_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    output_handler: BaseTool | None = None,
    intermediate_steps_processor: Callable | None = None,
) -> Runnable:
    prompt = prompt.partial(
        tools=render_text_description(list(tools)),
        output_handler=render_text_description([output_handler]) if output_handler else "",
    )
    check_variables(prompt)

    tools_for_llm = list(tools)
    if output_handler:
        tools_for_llm.append(output_handler)

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
        | prompt
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
        prompt_prefix: str | ChatPromptTemplate | None = None,
        prompt: ChatPromptTemplate | None = None,
        chat_history: bool | GetSessionHistoryCallable = True,
        output_handler: MotleySupportedTool | None = None,
        handle_parsing_errors: bool = True,
        handle_tool_errors: bool = True,
        llm: BaseChatModel | None = None,
        max_iterations: int | None = Defaults.DEFAULT_REACT_AGENT_MAX_ITERATIONS,
        intermediate_steps_processor: Callable | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            tools: Tools to add to the agent.
            description: Description of the agent.
            name: Name of the agent.
            prompt_prefix: Prefix to the agent's prompt.
            prompt: The prompt to use.
                See Prompt section below for more on the expected input variables.
            chat_history: Whether to use chat history or not.
                If `True`, uses `InMemoryChatMessageHistory`.
                If a callable is passed, it is used to get the chat history by session_id.
                See :class:`langchain_core.runnables.history.RunnableWithMessageHistory`
                for more details.
            output_handler: Output handler for the agent.
            handle_parsing_errors: Whether to handle parsing errors.
            handle_tool_errors: Whether to handle tool errors.
                If True, `handle_tool_error` and `handle_validation_error` in all tools
                are set to True.
            max_iterations: The maximum number of agent iterations.
            intermediate_steps_processor: Function that modifies the intermediate steps array
                in some way before each agent iteration.
            llm: Language model to use.
            verbose: Whether to log verbose output.

        Prompt:
            The prompt must have `agent_scratchpad`, `chat_history`, and `additional_notes`
            ``MessagesPlaceholder``s.
            If a prompt is not passed in, the default one is used.

            The default prompt slightly varies depending on the language model used.
            See :mod:`motleycrew.agents.langchain.tool_calling_react_prompts` for more details.
        """
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if not tools:
            raise ValueError("You must provide at least one tool to the ReActToolCallingAgent")

        if prompt is None:
            prompt = get_relevant_default_prompt(llm=llm, output_handler=output_handler)

        def agent_factory(
            tools: dict[str, MotleyTool], output_handler: Optional[MotleyTool] = None
        ) -> AgentExecutor:
            tools_for_langchain = [t.to_langchain_tool() for t in tools.values()]
            if output_handler:
                output_handler_for_langchain = output_handler.to_langchain_tool()
            else:
                output_handler_for_langchain = None

            agent = create_tool_calling_react_agent(
                llm=llm,
                tools=tools_for_langchain,
                prompt=prompt,
                output_handler=output_handler_for_langchain,
                intermediate_steps_processor=intermediate_steps_processor,
            )

            if output_handler_for_langchain:
                tools_for_langchain.append(output_handler_for_langchain)

            if handle_tool_errors:
                for tool in tools_for_langchain:
                    tool.handle_tool_error = True
                    tool.handle_validation_error = True

            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools_for_langchain,
                handle_parsing_errors=handle_parsing_errors,
                verbose=verbose,
                max_iterations=max_iterations,
            )
            return agent_executor

        super().__init__(
            prompt_prefix=prompt_prefix,
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            chat_history=chat_history,
            input_as_messages=True,
            verbose=verbose,
        )
