from typing import Sequence, List, Union

from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools.render import render_text_description
from langchain_core.agents import AgentFinish, AgentActionMessageLog

from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

from motleycrew.agent.parent import MotleyAgentAbstractParent
from motleycrew.agent.langchain.langchain import LangchainMotleyAgentParent
from motleycrew.common import MotleySupportedTool


default_think_prompt = ChatPromptTemplate.from_template(
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

Question: {input}
Thought: {agent_scratchpad}
"""
)


default_act_prompt = ChatPromptTemplate.from_template(
    """
Your objective is to contribute to answering the below question, by using the tools at your disposal.

You have access to the following tools:
{tools}

Your response MUST be a tool call to one of these unless the last input
message contains the words "Final Answer"; if it does, just repeat it.


Question: {input}
Thought:{agent_scratchpad}
"""
)


def print_passthrough(x):
    return x


def add_thought_to_background(x: dict):
    out = x["background"]
    out["agent_scratchpad"] += [x["thought"]]
    return out


def check_variables(prompt: ChatPromptTemplate):
    missing_vars = {"agent_scratchpad"}.difference(prompt.input_variables)
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")


def create_openai_tools_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: Union[ChatPromptTemplate, Sequence[ChatPromptTemplate]] | None = None,
) -> Runnable:
    """Create an agent that uses OpenAI tools.
    #TODO: this docstring is out of date, need to update it

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use. See Prompt section below for more on the expected
            input variables.

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    Example:

        .. code-block:: python

            from langchain import hub
            from langchain_community.chat_models import ChatOpenAI
            from langchain.agents import AgentExecutor, create_openai_tools_agent

            prompt = hub.pull("hwchase17/openai-tools-agent")
            model = ChatOpenAI()
            tools = ...

            agent = create_openai_tools_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools)

            agent_executor.invoke({"input": "hi"})

            # Using with chat history
            from langchain_core.messages import AIMessage, HumanMessage
            agent_executor.invoke(
                {
                    "input": "what's my name?",
                    "chat_history": [
                        HumanMessage(content="hi! my name is bob"),
                        AIMessage(content="Hello Bob! How can I assist you today?"),
                    ],
                }
            )

    Prompt:

        The agent prompt must have an `agent_scratchpad` key that is a
            ``MessagesPlaceholder``. Intermediate agent actions and tool output
            messages will be passed in here.

        Here's an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant"),
                    MessagesPlaceholder("chat_history", optional=True),
                    ("human", "{input}"),
                    MessagesPlaceholder("agent_scratchpad"),
                ]
            )
    """
    if prompt is None:
        think_prompt = default_think_prompt
        act_prompt = default_act_prompt
    else:
        think_prompt = prompt[0]
        act_prompt = prompt[1]

    think_prompt = think_prompt.partial(tools=render_text_description(list(tools)))
    act_prompt = act_prompt.partial(tools=render_text_description(list(tools)))

    check_variables(think_prompt)
    check_variables(act_prompt)

    llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

    second_llm_call = (
        RunnableLambda(add_thought_to_background)
        | RunnableLambda(print_passthrough)
        | act_prompt
        | RunnableLambda(print_passthrough)
        | llm_with_tools
        | RunnableLambda(print_passthrough)
        | OpenAIToolsAgentOutputParser()
    )

    agent = (
        RunnableLambda(print_passthrough)
        | RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            )
        )
        | {"thought": think_prompt | llm, "background": RunnablePassthrough()}
        | RunnableLambda(print_passthrough)
        | {"action": second_llm_call, "thought": RunnableLambda(lambda x: x["thought"])}
        | RunnableLambda(print_passthrough)
        | RunnableLambda(lambda x: add_messages_to_action(x["action"], [x["thought"]]))
        | RunnableLambda(print_passthrough)
    )
    return agent


def add_messages_to_action(
    actions: List[AgentActionMessageLog] | AgentFinish, messages: List[BaseMessage]
) -> List[AgentActionMessageLog] | AgentFinish:
    if not isinstance(actions, AgentFinish):
        for action in actions:
            action.message_log = messages + list(action.message_log)
    return actions


class ReactOpenAIToolsAgent(LangchainMotleyAgentParent):
    def __new__(
        cls,
        tools: Sequence[MotleySupportedTool],
        goal: str = "",  # gets ignored at the moment
        prompt: ChatPromptTemplate | Sequence[ChatPromptTemplate] | None = None,
        llm: BaseLanguageModel | None = None,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        verbose: bool = False,
    ):
        return cls.from_function(
            goal=goal,
            llm=llm,
            delegation=delegation,
            tools=tools,
            prompt=prompt,
            function=create_openai_tools_react_agent,
            require_tools=True,
            verbose=verbose,
        )
