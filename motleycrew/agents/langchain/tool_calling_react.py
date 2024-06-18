from typing import Sequence, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import GetSessionHistoryCallable
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentFinish, AgentActionMessageLog
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages.base import merge_content

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser

from langchain.tools.render import render_text_description

from motleycrew.agents.langchain import LangchainMotleyAgent
from motleycrew.tools import MotleyTool
from motleycrew.common import MotleySupportedTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.common.utils import print_passthrough


default_think_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the following questions as best you can; think carefully, one step at a time, 
and outline the next step towards answering the question.

You will later have access to the following tools, but you can't use them yet:
{tools}

Your reply must begin with "Thought:" and then describe what the next step should be, 
given the information so far - either to use one of the tools from the above list 
(and details on how to use it), or give the final answer like described below.
Do NOT return a tool call, just a single thought about what it should be.
If the information so far is not sufficient to answer the question precisely and completely
(rather than sloppily and approximately), don't hesitate to
request to use tools again, until sufficient information is gathered. Don't stop this until
you are certain that you have enough information to answer the question.

If you have sufficient information to answer the question, your reply must look like
```
Thought: I now know the final answer
Final Answer: [the final answer to the original input question]
```
but without the backticks.

Begin!

""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


default_act_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Your objective is to contribute to answering the below question, by using the tools at your disposal.

You have access to the following tools:
{tools}

Your response MUST be only a tool call to one of these unless the last input
message contains the words "Final Answer"; {output_instruction}
{output_handler}

""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
default_act_prompt_without_output_handler = default_act_prompt.partial(
    output_instruction="if it does, just repeat it."
)
default_act_prompt_with_output_handler = default_act_prompt.partial(
    output_instruction="if it does, you must use the output handler:"
)


def check_variables(prompt: ChatPromptTemplate):
    """
    Args:
        prompt (ChatPromptTemplate):

    Returns:

    """
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")


def add_thought_to_background(x: dict):
    """Description

    Args:
        x (dict):

    Returns:

    """
    out = x["background"]
    out["agent_scratchpad"] += [x["thought"]]
    return out


def cast_thought_to_human_message(thought: BaseMessage):
    content = thought.content
    if content:
        # HACK: this cast to a HumanMessage is needed for Anthropic models to work
        # because they will treat an AIMessage in input as their output prefill.
        # Also, we remove tool_use messages from the content in case a model calls a tool.
        if isinstance(content, list):
            content = [chunk for chunk in content if chunk.get("type") != "tool_use"]
    return HumanMessage(content=content)


def add_messages_to_action(
    actions: List[AgentActionMessageLog] | AgentFinish, messages: List[BaseMessage]
) -> List[AgentActionMessageLog] | AgentFinish:
    """
    Args:
        actions (:obj:`List[AgentActionMessageLog]`, :obj:`AgentFinish`):
        messages (List[BaseMessage]):

    Returns:
        List[AgentActionMessageLog] | AgentFinish:
    """
    if not isinstance(actions, AgentFinish):
        for action in actions:
            action.message_log = messages + list(action.message_log)
    return actions


def merge_consecutive_messages(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Tries to merge consecutive messages of the same type in the provided list.
    This mainly serves as a workaround for Anthropic models that don't accept
    multiple AIMessages in a row.

    Args:
        messages (Sequence[BaseMessage]): The list of messages to process.

    Returns:
        List[BaseMessage]: The list of messages with consecutive messages of the same type merged.
    """
    merged_messages = []
    for message in messages:
        if (
            not merged_messages
            or type(merged_messages[-1]) != type(message)
            or isinstance(message, ToolMessage)
        ):
            merged_messages.append(message)
        else:
            merged_messages[-1].content = merge_content(
                merged_messages[-1].content, message.content
            )
    return merged_messages


def create_tool_calling_react_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    think_prompt: ChatPromptTemplate | None = None,
    act_prompt: ChatPromptTemplate | None = None,
    output_handler: BaseTool | None = None,
) -> Runnable:
    """Create a ReAct-style agent that supports tool calling.

    Args:
        llm (BaseChatModel): LLM to use as the agent.
        tools (Sequence[BaseTool]): Tools this agent has access to.
        think_prompt (ChatPromptTemplate, optional): The thinking step prompt to use.
            See Prompt section below for more on the expected input variables.
        act_prompt (ChatPromptTemplate, optional): The acting step prompt to use.
            See Prompt section below for more on the expected input variables.
        output_handler (BaseTool, optional): Tool to use for returning agent's output.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Prompt:
        This agent uses two prompts, one for thinking and one for acting. The prompts
        must have `agent_scratchpad` and `chat_history` ``MessagesPlaceholder``s.
        If a prompt is not passed in, the default one is used.

    """
    if think_prompt is None:
        think_prompt = default_think_prompt

    if act_prompt is None:
        if output_handler:
            act_prompt = default_act_prompt_with_output_handler
        else:
            act_prompt = default_act_prompt_without_output_handler

    think_prompt = think_prompt.partial(tools=render_text_description(list(tools)))
    act_prompt = act_prompt.partial(
        tools=render_text_description(list(tools)),
        output_handler=render_text_description([output_handler]) if output_handler else "",
    )

    check_variables(think_prompt)
    check_variables(act_prompt)

    tools_for_llm = list(tools)
    if output_handler:
        tools_for_llm.append(output_handler)
    llm_with_tools = llm.bind_tools(tools=tools_for_llm)

    think_chain = think_prompt | llm_with_tools | RunnableLambda(cast_thought_to_human_message)
    act_chain = (
        RunnableLambda(add_thought_to_background)
        | act_prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: merge_consecutive_messages(
                format_to_tool_messages(x["intermediate_steps"])
            )
        )
        | {"thought": think_chain, "background": RunnablePassthrough()}
        | RunnableLambda(print_passthrough)
        | {"action": act_chain, "thought": RunnableLambda(lambda x: x["thought"])}
        | RunnableLambda(lambda x: add_messages_to_action(x["action"], [x["thought"]]))
    )
    return agent


class ReActToolCallingAgent(LangchainMotleyAgent):
    def __init__(
        self,
        tools: Sequence[MotleySupportedTool],
        description: str | None = None,
        name: str | None = None,
        think_prompt: ChatPromptTemplate | None = None,
        act_prompt: ChatPromptTemplate | None = None,
        chat_history: bool | GetSessionHistoryCallable = True,
        output_handler: MotleySupportedTool | None = None,
        handle_parsing_errors: bool = True,
        handle_tool_errors: bool = True,
        llm: BaseChatModel | None = None,
        verbose: bool = False,
    ):
        """Universal ReAct-style agent that supports tool calling.

        Args:
            tools (Sequence[MotleySupportedTool]):
            description (:obj:`str`, optional):
            name (:obj:`str`, optional):
            think_prompt (ChatPromptTemplate, optional): The thinking step prompt to use.
                See Prompt section below for more on the expected input variables.
            act_prompt (ChatPromptTemplate, optional): The acting step prompt to use.
                See Prompt section below for more on the expected input variables.
            chat_history (:obj:`bool`, :obj:`GetSessionHistoryCallable`):
                Whether to use chat history or not. If `True`, uses `InMemoryChatMessageHistory`.
                If a callable is passed, it is used to get the chat history by session_id.
                See Langchain `RunnableWithMessageHistory` get_session_history param for more details.
            output_handler (BaseTool, optional): Tool to use for returning agent's output.
            handle_parsing_errors (:obj:`bool`, optional): Whether to handle parsing errors or not.
            handle_tool_errors (:obj:`bool`, optional): Whether to handle tool errors or not.
                If True, `handle_tool_error` and `handle_validation_error` in all tools are set to True.
            llm (:obj:`BaseLanguageModel`, optional):
            verbose (:obj:`bool`, optional):
        """
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if not tools:
            raise ValueError("You must provide at least one tool to the ReActToolCallingAgent")

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
                think_prompt=think_prompt,
                act_prompt=act_prompt,
                output_handler=output_handler_for_langchain,
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
