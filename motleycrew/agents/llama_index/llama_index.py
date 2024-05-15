from typing import Any, Optional, Sequence

from llama_index.core.agent import AgentRunner
from langchain_core.runnables import RunnableConfig

from motleycrew.agents.parent import MotleyAgentParent
from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.tasks import Task
from motleycrew.common import MotleySupportedTool
from motleycrew.common import MotleyAgentFactory


class LlamaIndexMotleyAgentParent(MotleyAgentParent):
    def __init__(
        self,
        goal: str,
        name: str | None = None,
        agent_factory: MotleyAgentFactory | None = None,
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            goal=goal,
            name=name,
            agent_factory=agent_factory,
            delegation=delegation,
            tools=tools,
            verbose=verbose,
        )

    def invoke(
        self,
        task_dict: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
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
        delegation: bool | Sequence[MotleyAgentAbstractParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ) -> "LlamaIndexMotleyAgentParent":
        wrapped_agent = LlamaIndexMotleyAgentParent(
            goal=goal, delegation=delegation, tools=tools, verbose=verbose
        )
        wrapped_agent._agent = agent
        return wrapped_agent
