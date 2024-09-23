from __future__ import annotations

from typing import Optional, Sequence

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import GetSessionHistoryCallable

from motleycrew.agents.langchain import LangchainMotleyAgent
from motleycrew.common import LLMFramework, MotleySupportedTool
from motleycrew.common.llms import init_llm
from motleycrew.tools import MotleyTool

DEFAULT_REACT_PROMPT = PromptTemplate.from_template(
    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
)


class LegacyReActMotleyAgent(LangchainMotleyAgent):
    """Basic ReAct agent compatible with older models without dedicated tool calling support.

    It's probably better to use the more advanced
    :class:`motleycrew.agents.langchain.tool_calling_react.ReActToolCallingAgent` with newer models.
    """

    def __init__(
        self,
        tools: Sequence[MotleySupportedTool],
        description: str | None = None,
        name: str | None = None,
        prompt_prefix: str | None = None,
        chat_history: bool | GetSessionHistoryCallable = True,
        force_output_handler: bool = False,
        prompt: str | None = None,
        handle_parsing_errors: bool = True,
        handle_tool_errors: bool = True,
        llm: BaseLanguageModel | None = None,
        runnable_config: RunnableConfig | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            tools: Tools to add to the agent.
            description: Description of the agent.
            name: Name of the agent.
            prompt_prefix: Prefix to the agent's prompt.
            output_handler: Output handler for the agent.
            chat_history: Whether to use chat history or not.
            force_output_handler: Whether to force the agent to return through an output handler.
            prompt: Custom prompt to use with the agent.
            handle_parsing_errors: Whether to handle parsing errors.
            handle_tool_errors: Whether to handle tool errors.
            llm: Language model to use.
            runnable_config: Default Langchain config to use when invoking the agent.
                It can be used to add callbacks, metadata, etc.
            verbose: Whether to log verbose output.
        """
        if force_output_handler:
            raise Exception("Forced output handler is not supported with legacy ReAct agent.")

        if prompt is None:
            prompt = DEFAULT_REACT_PROMPT

        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if not tools:
            raise ValueError("You must provide at least one tool to the ReActMotleyAgent")

        def agent_factory(
            tools: dict[str, MotleyTool], output_handler: Optional[MotleyTool] = None
        ) -> AgentExecutor:
            langchain_tools = [t.to_langchain_tool() for t in tools.values()]
            if output_handler:
                langchain_tools.append(output_handler.to_langchain_tool())

            if handle_tool_errors:
                for tool in langchain_tools:
                    tool.handle_tool_error = True
                    tool.handle_validation_error = True

            agent = create_react_agent(llm=llm, tools=langchain_tools, prompt=prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=langchain_tools,
                handle_parsing_errors=handle_parsing_errors,
                verbose=verbose,
            )
            return agent_executor

        super().__init__(
            prompt_prefix=prompt_prefix,
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            chat_history=chat_history,
            runnable_config=runnable_config,
            verbose=verbose,
        )
