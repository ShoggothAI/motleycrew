""" Module description

Attributes:
    default_think_prompt:
    default_act_prompt:

"""

from typing import Sequence, List, Union

from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentFinish, AgentActionMessageLog
from langchain_core.prompts import MessagesPlaceholder

from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from langchain.agents.output_parsers.tools import ToolsAgentOutputParser

from langchain.tools.render import render_text_description

from motleycrew.agents.langchain.langchain import LangchainMotleyAgent
from motleycrew.common import MotleySupportedTool
from motleycrew.common.utils import print_passthrough


default_think_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Answer the following questions as best you can; think carefully, one step at a time, and outline the next step
towards answering the question.

You will later have access to the following tools:
{tools}

Your reply must begin with "Thought:" and then
describe what the next step should be, given the information so far - either to use one of the tools from
the above list (and details on how to use it), or give the final answer like described below. ]
Do NOT return a tool call, just a thought about what
it should be.
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

Your response MUST be a tool call to one of these unless the last input
message contains the words "Final Answer"; if it does, just repeat it.

""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def check_variables(prompt: ChatPromptTemplate):
    """Description

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


def add_messages_to_action(
    actions: List[AgentActionMessageLog] | AgentFinish, messages: List[BaseMessage]
) -> List[AgentActionMessageLog] | AgentFinish:
    """Description

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


def create_tool_calling_react_agent(
    llm: BaseChatModel,
    tools: Sequence[BaseTool],
    prompt: tuple[ChatPromptTemplate, ChatPromptTemplate] | None = None,
) -> Runnable:
    """Create a ReAct-style agent that supports tool calling.

    Args:
        llm (BaseChatModel): LLM to use as the agent.
        tools (Sequence[BaseTool]): Tools this agent has access to.
        prompt (Tuple[ChatPromptTemplate, ChatPromptTemplate], optional): The prompts to use.
            See Prompt section below for more on the expected input variables.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Prompt:
        This agent uses two prompts, one for thinking and one for acting. The prompts
        must have `agent_scratchpad` and `chat_history` ``MessagesPlaceholder``s.
        If the prompt is not passed in, the default prompts are used.

    """
    if prompt is None:
        think_prompt = default_think_prompt
        act_prompt = default_act_prompt
    else:
        think_prompt, act_prompt = prompt

    think_prompt = think_prompt.partial(tools=render_text_description(list(tools)))
    act_prompt = act_prompt.partial(tools=render_text_description(list(tools)))

    check_variables(think_prompt)
    check_variables(act_prompt)

    llm_with_tools = llm.bind_tools(tools=tools)

    think_chain = think_prompt | llm
    act_chain = (
        RunnableLambda(add_thought_to_background)
        | act_prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
        )
        | {"thought": think_chain, "background": RunnablePassthrough()}
        | RunnableLambda(print_passthrough)
        | {"action": act_chain, "thought": RunnableLambda(lambda x: x["thought"])}
        | RunnableLambda(lambda x: add_messages_to_action(x["action"], [x["thought"]]))
    )
    return agent


class ReActToolCallingAgent(LangchainMotleyAgent):
    def __new__(
        cls,
        tools: Sequence[MotleySupportedTool],
        goal: str = "",  # gets ignored at the moment
        name: str | None = None,
        prompt: ChatPromptTemplate | Sequence[ChatPromptTemplate] | None = None,
        with_history: bool = False,
        llm: BaseChatModel | None = None,
        verbose: bool = False,
    ):
        """Description

        Args:
            tools (Sequence[MotleySupportedTool]):
            goal (:obj:`str`, optional):
            name (:obj:`str`, optional):
            prompt (:obj:ChatPromptTemplate`, :obj:`Sequence[ChatPromptTemplate]', optional):
            llm (:obj:`BaseLanguageModel`, optional):
            verbose (:obj:`bool`, optional):
        """
        return cls.from_function(
            description=goal,
            name=name,
            llm=llm,
            tools=tools,
            prompt=prompt,
            function=create_tool_calling_react_agent,
            require_tools=True,
            with_history=with_history,
            verbose=verbose,
        )
