from typing import Any, Optional, Callable, Union, Dict, List, Tuple

from langchain_core.agents import AgentFinish, AgentAction
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.messages import AIMessage
from motleycrew.agents.parent import DirectOutput


class LangchainOutputHandlerMixin:

    def agent_plane_decorator(self):
        """Decorator for inclusion in the call chain of the agent, the output handler tool"""

        def decorator(func: Callable):

            def wrapper(
                intermediate_steps: List[Tuple[AgentAction, str]],
                callbacks: "Callbacks" = None,
                **kwargs: Any,
            ) -> Union[AgentAction, AgentFinish]:
                step = func(intermediate_steps, callbacks, **kwargs)

                if not isinstance(step, AgentFinish):
                    return step

                if self.output_handler is not None:
                    return AgentAction(
                        tool=self.output_handler.name,
                        tool_input=step.return_values,
                        log="Use tool: {}\nInput: {}".format(
                            self.output_handler.name, step.return_values
                        ),
                    )
                return step

            return wrapper

        return decorator

    def take_next_step_decorator(self):
        """DirectOutput exception interception decorator"""

        def decorator(func: Callable):
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
                    message = "Final answer\n" + str(direct_ex.output)
                    return AgentFinish(
                        return_values={"output": direct_ex.output},
                        messages=[AIMessage(content=message)],
                        log=message,
                    )
                return step

            return wrapper

        return decorator
