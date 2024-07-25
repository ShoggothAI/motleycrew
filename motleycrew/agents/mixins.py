from typing import Any, Optional, Callable, Union, Dict, List, Tuple

from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, Tool
from motleycrew.agents.parent import DirectOutput
from motleycrew.tools import MotleyTool


class LangchainOutputHandlingAgentMixin:
    """A mixin for Langchain-based agents that support output handlers."""

    output_handler: Optional[MotleyTool] = None
    _agent_finish_blocker_tool: Optional[BaseTool] = None

    def _create_agent_finish_blocker_tool(self) -> BaseTool:
        """Create a tool that will force the agent to retry if it attempts to return the output
        bypassing the output handler.
        """

        def create_agent_finish_blocking_message(input: Any = None) -> str:
            return (
                f"You must call the `{self.output_handler.name}` tool to return the final output.\n"
            )

        return Tool.from_function(
            name="agent_finish_blocker",
            description="",
            func=create_agent_finish_blocking_message,
        )

    def _is_blocker_action(self, action: AgentAction) -> bool:
        """Checks whether the action of the response blocking tool"""
        return bool(
            isinstance(action, AgentAction) and action.tool == self._agent_finish_blocker_tool.name
        )

    def agent_plan_decorator(self, func: Callable):
        """Decorator for Agent.plan() method that intercepts AgentFinish events"""

        def wrapper(
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: "Callbacks" = None,
            **kwargs: Any,
        ) -> Union[AgentAction, AgentFinish]:
            additional_notes = []

            if self.output_handler:
                to_remove_steps = []
                for intermediate_step in intermediate_steps:
                    action, action_output = intermediate_step
                    if self._is_blocker_action(action):
                        # Add the interaction telling the LLM that it must use the output handler
                        additional_notes.append(("ai", action.tool_input))
                        additional_notes.append(("user", action_output))
                        to_remove_steps.append(intermediate_step)

                for to_remove_step in to_remove_steps:
                    intermediate_steps.remove(to_remove_step)

            if additional_notes:
                kwargs["additional_notes"] = additional_notes

            step = func(intermediate_steps, callbacks, **kwargs)

            if not isinstance(step, AgentFinish):
                return step

            if self.output_handler is not None:
                return AgentAction(
                    tool=self._agent_finish_blocker_tool.name,
                    tool_input=step.log,
                    log="\nDetected AgentFinish, blocking it to force output via output handler.\n",
                )
            return step

        return wrapper

    def take_next_step_decorator(self, func: Callable):
        """
        Decorator for ``AgentExecutor._take_next_step()`` method that catches DirectOutput exceptions.
        """

        def wrapper(
            name_to_tool_map: Dict[str, BaseTool],
            color_mapping: Dict[str, str],
            inputs: Dict[str, str],
            intermediate_steps: List[Tuple[AgentAction, str]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
        ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:

            try:
                step = func(
                    name_to_tool_map, color_mapping, inputs, intermediate_steps, run_manager
                )
            except DirectOutput as direct_ex:
                message = str(direct_ex.output)
                return AgentFinish(
                    return_values={"output": direct_ex.output},
                    messages=[AIMessage(content=message)],
                    log=message,
                )
            return step

        return wrapper

    def _run_tool_direct_decorator(self, func: Callable):
        """Decorator of the tool's _run method, for intercepting a DirectOutput exception"""

        def wrapper(*args, config: RunnableConfig, **kwargs):
            try:
                result = func(*args, **kwargs, config=config)
            except DirectOutput as direct_exc:
                return direct_exc
            return result

        return wrapper

    def run_tool_direct_decorator(self, func: Callable):
        """Decorator of the tool's run method, for intercepting a DirectOutput exception"""

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, DirectOutput):
                raise result
            return result

        return wrapper
