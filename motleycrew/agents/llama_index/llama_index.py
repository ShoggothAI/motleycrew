""" Module description """
from typing import Any, Optional, Sequence

try:
    from llama_index.core.agent import AgentRunner
except ImportError:
    AgentRunner = None

from langchain_core.runnables import RunnableConfig

from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.tasks import TaskUnit
from motleycrew.common import MotleySupportedTool
from motleycrew.common import MotleyAgentFactory
from motleycrew.common.utils import ensure_module_is_installed


class LlamaIndexMotleyAgent(MotleyAgentParent):
    def __init__(
        self,
        description: str,
        name: str | None = None,
        agent_factory: MotleyAgentFactory | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ):
        """ Description

        Args:
            description (str):
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
            verbose=verbose,
        )

    def invoke(
        self,
        task_dict: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """ Description

        Args:
            task_dict (dict):
            config (:obj:`RunnableConfig`, optional):
            **kwargs:

        Returns:
            Any:
        """
        self.materialize()

        prompt = task_dict.get("prompt")
        if not prompt:
            raise ValueError("Task must have a prompt")

        output = self.agent.chat(prompt)
        return output.response

    @staticmethod
    def from_agent(
        agent: AgentRunner,
        goal: str,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ) -> "LlamaIndexMotleyAgent":
        """ Description

        Args:
            agent (AgentRunner):
            goal (str):
            tools (:obj:`Sequence[MotleySupportedTool]`, optional):
            verbose (:obj:`bool`, optional):

        Returns:
            LlamaIndexMotleyAgent:
        """
        ensure_module_is_installed("llama_index")
        wrapped_agent = LlamaIndexMotleyAgent(description=goal, tools=tools, verbose=verbose)
        wrapped_agent._agent = agent
        return wrapped_agent
