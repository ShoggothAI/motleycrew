""" Module description"""

from typing import Sequence, Optional

from langchain import hub
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables.history import GetSessionHistoryCallable
from langchain.agents import AgentExecutor
from langchain.agents import create_react_agent

from motleycrew.agents.langchain import LangchainMotleyAgent
from motleycrew.tools import MotleyTool
from motleycrew.common import MotleySupportedTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm


OUTPUT_HANDLER_WITH_DEFAULT_PROMPT_MESSAGE = (
    "Langchain's default ReAct prompt tells the agent to include a final answer keyword, "
    "which later confuses the parser when an output handler is used. "
    "Please provide a custom prompt if using an output handler."
)


class ReActMotleyAgent(LangchainMotleyAgent):
    def __init__(
        self,
        tools: Sequence[MotleySupportedTool],
        description: str | None = None,
        name: str | None = None,
        prompt: str | None = None,
        output_handler: MotleySupportedTool | None = None,
        chat_history: bool | GetSessionHistoryCallable = True,
        handle_parsing_errors: bool = True,
        handle_tool_errors: bool = True,
        llm: BaseLanguageModel | None = None,
        verbose: bool = False,
    ):
        """Basic ReAct agent compatible with older models without dedicated tool calling support.
        It's probably better to use the more advanced `ReActToolCallingAgent` with newer models.

        Args:
            tools (Sequence[MotleySupportedTool]):
            description (:obj:`str`, optional):
            name (:obj:`str`, optional):
            prompt (:obj:`str`, optional): Prompt to use. If not provided, uses hwchase17/react.
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
        if prompt is None:
            if output_handler is not None:
                raise Exception(OUTPUT_HANDLER_WITH_DEFAULT_PROMPT_MESSAGE)
            prompt = hub.pull("hwchase17/react")

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
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            chat_history=chat_history,
            verbose=verbose,
        )
