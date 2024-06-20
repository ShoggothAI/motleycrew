""" Module description """

from typing import TYPE_CHECKING, Optional, Sequence, Any, Callable, Dict, Union, Tuple
from functools import wraps
import inspect

from langchain_core.tools import Tool
from langchain_core.runnables import Runnable
from langchain_core.prompts.chat import ChatPromptTemplate, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.agents.output_handler import MotleyOutputHandler
from motleycrew.tools import MotleyTool
from motleycrew.common import MotleyAgentFactory, MotleySupportedTool
from motleycrew.common.exceptions import (
    AgentNotMaterialized,
    CannotModifyMaterializedAgent,
    InvalidOutput,
)
from motleycrew.common import logger

if TYPE_CHECKING:
    from motleycrew import MotleyCrew


class DirectOutput(BaseException):
    def __init__(self, output: Any):
        self.output = output


class MotleyAgentParent(MotleyAgentAbstractParent, Runnable):
    def __init__(
        self,
        description: str | None = None,
        name: str | None = None,
        agent_factory: MotleyAgentFactory | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        output_handler: MotleySupportedTool | None = None,
        verbose: bool = False,
    ):
        """Description

        Args:
            description (:obj:`str`, optional):
            name (:obj:`str`, optional):
            agent_factory (:obj:`MotleyAgentFactory`, optional):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            output_handler (:obj:`MotleySupportedTool`, optional):
            verbose (:obj:`bool`, optional):
        """
        self.name = name or description
        self.description = description  # becomes tool description
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

    def compose_prompt(self, input_dict: dict, prompt: ChatPromptTemplate | str) -> str:
        # TODO: always cast description and prompt to ChatPromptTemplate first?
        prompt_messages = []

        if not self.description and not prompt:
            raise Exception("Cannot compose agent prompt without description or prompt")

        if self.description:
            if isinstance(self.description, ChatPromptTemplate):
                prompt_messages += self.description.invoke(input_dict).to_messages()

            elif isinstance(self.description, str):
                prompt_messages.append(SystemMessage(content=self.description))

            else:
                raise ValueError("Agent description must be a string or a ChatPromptTemplate")

        if prompt:
            if isinstance(prompt, ChatPromptTemplate):
                prompt_messages += prompt.invoke(input_dict).to_messages()

            elif isinstance(prompt, str):
                prompt_messages.append(HumanMessage(content=prompt))

            else:
                raise ValueError("Prompt must be a string or a ChatPromptTemplate")

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
        return self._agent is not None

    def _prepare_output_handler(self) -> Optional[MotleyTool]:
        """
        Wraps the output handler in one more tool layer,
        adding the necessary stuff for returning direct output through output handler.
        Expects agent's later invocation through _run_and_catch_output method below.
        """
        if not self.output_handler:
            return None

        # TODO: make this neater by constructing MotleyOutputHandler from tools?
        if isinstance(self.output_handler, MotleyOutputHandler):
            exceptions_to_handle = self.output_handler.exceptions_to_handle
            description = self.output_handler.description
        else:
            exceptions_to_handle = (InvalidOutput,)
            description = self.output_handler.description or f"Output handler"
            assert isinstance(description, str)
            description += "\n ONLY RETURN THE FINAL RESULT USING THIS TOOL!"

        def handle_agent_output(*args, **kwargs):
            assert self.output_handler
            try:
                output = self.output_handler._run(*args, **kwargs)
            except exceptions_to_handle as exc:
                return f"{exc.__class__.__name__}: {str(exc)}"

            raise DirectOutput(output)

        prepared_output_handler = StructuredTool(
            name=self.output_handler.name,
            description=description,
            func=handle_agent_output,
            args_schema=self.output_handler.args_schema,
        )

        return MotleyTool.from_langchain_tool(prepared_output_handler)

    @staticmethod
    def _run_and_catch_output(
        action: Callable, action_args: tuple, action_kwargs: Dict[str, Any]
    ) -> Tuple[bool, Any]:
        """
        Catcher for the direct output from the output handler (see _prepare_output_handler).

        Args:
            action (Callable): the action inside which the output handler is supposed to be called.
                Usually the invocation method of the underlying agent.
            action_args (tuple): the args for the action
            action_kwargs (tuple): the kwargs for the action

        Returns:
            tuple[bool, Any]: a tuple with a boolean indicating whether the output was caught
                via DirectOutput and the output itself
        """
        assert callable(action)

        try:
            output = action(*action_args, **action_kwargs)
        except DirectOutput as output_exc:
            return True, output_exc.output

        return False, output

    def materialize(self):
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

    def prepare_for_invocation(self, input: dict) -> str:
        """Prepares the agent for invocation by materializing it and composing the prompt.

        Should be called in the beginning of the agent's invoke method.

        Args:
            input (dict): the input to the agent

        Returns:
            str: the composed prompt
        """
        self.materialize()

        if isinstance(self.output_handler, MotleyOutputHandler):
            self.output_handler.agent = self
            self.output_handler.agent_input = input

        prompt = self.compose_prompt(input, input.get("prompt"))
        return prompt

    def add_tools(self, tools: Sequence[MotleySupportedTool]):
        """Description

        Args:
            tools (Sequence[MotleySupportedTool]):

        Returns:

        """
        if self.is_materialized and tools:
            raise CannotModifyMaterializedAgent(agent_name=self.name)

        for t in tools:
            motley_tool = MotleyTool.from_supported_tool(t)
            if motley_tool.name not in self.tools:
                self.tools[motley_tool.name] = motley_tool

    def as_tool(self, input_schema: Optional[BaseModel] = None) -> MotleyTool:
        """Description

        Args:
            input_schema (:obj:`BaseModel`, optional):

        Returns:
            MotleyTool:
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
                name=self.name,
                description=self.description,
                func=call_agent,
                args_schema=input_schema,
            )
        )

    # def call_as_tool(self, *args, **kwargs) -> Any:
    #     logger.info("Entering delegation for %s", self.name)
    #     assert self.crew, "can't accept delegated task outside of a crew"
    #
    #     if len(args) > 0:
    #         input_ = args[0]
    #     elif "tool_input" in kwargs:
    #         # Is this a crewai notation?
    #         input_ = kwargs["tool_input"]
    #     else:
    #         input_ = json.dumps(kwargs)
    #
    #     logger.info("Made the args: %s", input_)
    #
    #     # TODO: pass context of parent task to agent nicely?
    #     # TODO: mark the current task as depending on the new task
    #     task = SimpleTaskRecipe(
    #         description=input_,
    #         name=input_,
    #         agent=self,
    #         # TODO inject the new subtask as a dep and reschedule the parent
    #         # TODO probably can't do this from here since we won't know if
    #         # there are other tasks to schedule
    #         crew=self.crew,
    #     )
    #
    #     # TODO: make sure tools return task objects, which are properly used by callers
    #     logger.info("Executing subtask '%s'", task.name)
    #     self.crew.task_graph.set_task_running(task=task)
    #     result = self.crew.execute(task, return_result=True)
    #
    #     logger.info("Finished subtask '%s' - %s", task.name, result)
    #     self.crew.task_graph.set_task_done(task=task)
    #
    #     return result
