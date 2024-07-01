""" Module description """

import uuid
from typing import Any, Optional, Sequence

try:
    from llama_index.core.agent import AgentRunner
    from llama_index.core.chat_engine.types import ChatResponseMode
    from llama_index.core.agent.types import TaskStep, TaskStepOutput
    from llama_index.core.chat_engine.types import AgentChatResponse
except ImportError:
    AgentRunner = None
    ChatResponseMode = None
    TaskStep = None
    TaskStepOutput = None
    AgentChatResponse = None

from langchain_core.runnables import RunnableConfig

from motleycrew.agents.parent import MotleyAgentParent, DirectOutput
from motleycrew.common import MotleySupportedTool
from motleycrew.common import MotleyAgentFactory
from motleycrew.common.utils import ensure_module_is_installed


class LlamaIndexMotleyAgent(MotleyAgentParent):

    def __init__(
        self,
        description: str | None = None,
        name: str | None = None,
        agent_factory: MotleyAgentFactory[AgentRunner] | None = None,
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
            verbose (:obj:`bool`, optional):
        """
        super().__init__(
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            verbose=verbose,
        )

    def run_step_decorator(self):
        """Decorator for inclusion in the call chain of the agent, the output handler tool"""
        ensure_module_is_installed("llama_index")
        def decorator(func):
            def wrapper(
                task_id: str,
                step: Optional[TaskStep] = None,
                input: Optional[str] = None,
                mode: ChatResponseMode = ChatResponseMode.WAIT,
                **kwargs: Any,
            ):

                try:
                    cur_step_output = func(task_id, step, input, mode, **kwargs)
                except DirectOutput as output_exc:
                    self.direct_output = output_exc
                    output = AgentChatResponse(str(output_exc.output))
                    task_step = TaskStep(task_id=task_id, step_id=str(uuid.uuid4()))
                    cur_step_output = TaskStepOutput(
                        output=output, is_last=True, next_steps=[], task_step=task_step
                    )
                    return cur_step_output

                if self.output_handler is None:
                    return cur_step_output

                if cur_step_output.is_last:
                    cur_step_output.is_last = False
                    task_id = cur_step_output.task_step.task_id
                    output_task_step = TaskStep(
                        task_id=task_id,
                        step_id=str(uuid.uuid4()),
                        input="You must call the {} tool to return the output.".format(
                            self.output_handler.name
                        ),
                    )

                    cur_step_output.next_steps.append(output_task_step)

                    step_queue = self._agent.state.get_step_queue(task_id)
                    step_queue.extend(cur_step_output.next_steps)

                return cur_step_output

            return wrapper

        return decorator

    def materialize(self):
        super(LlamaIndexMotleyAgent, self).materialize()
        self._agent._run_step = self.run_step_decorator()(self._agent._run_step)

    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Description

        Args:
            input (dict):
            config (:obj:`RunnableConfig`, optional):
            **kwargs:

        Returns:
            Any:
        """
        prompt = self.prepare_for_invocation(input=input)

        output = self.agent.chat(prompt)

        if self.output_handler:
            return self.direct_output.output

        return output.response

    @staticmethod
    def from_agent(
        agent: AgentRunner,
        description: Optional[str] = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ) -> "LlamaIndexMotleyAgent":
        """Description

        Args:
            agent (AgentRunner):
            description (:obj:`str`, optional):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            verbose (:obj:`bool`, optional):

        Returns:
            LlamaIndexMotleyAgent:
        """
        ensure_module_is_installed("llama_index")
        wrapped_agent = LlamaIndexMotleyAgent(description=description, tools=tools, verbose=verbose)
        wrapped_agent._agent = agent
        return wrapped_agent
