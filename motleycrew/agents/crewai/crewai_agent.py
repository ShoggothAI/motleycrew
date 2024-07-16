from __future__ import annotations

from typing import Optional, Any, Sequence

from motleycrew.agents.crewai import CrewAIAgentWithConfig
from motleycrew.agents.crewai import CrewAIMotleyAgentParent
from motleycrew.common import LLMFramework
from motleycrew.common import MotleySupportedTool
from motleycrew.common.llms import init_llm
from motleycrew.tools import MotleyTool


class CrewAIMotleyAgent(CrewAIMotleyAgentParent):
    """MotleyCrew wrapper for CrewAI Agent.

    This wrapper is made to mimic the CrewAI agent's interface.
    That is why it has mostly the same arguments.
    """

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        prompt_prefix: str | None = None,
        description: str | None = None,
        delegation: bool = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        llm: Optional[Any] = None,
        output_handler: MotleySupportedTool | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            role: ``role`` param of the CrewAI Agent.

            goal: ``goal`` param of the CrewAI Agent.

            backstory: ``backstory`` param of the CrewAI Agent.

            prompt_prefix: Prefix to the agent's prompt.
                Can be used for providing additional context, such as the agent's role or backstory.

            description: Description of the agent.

                Unlike the prompt prefix, it is not included in the prompt.
                The description is only used for describing the agent's purpose
                when giving it as a tool to other agents.

            delegation: Whether to allow delegation or not.
                **Delegation is not supported in this wrapper.**
                Instead, pass the agents you want to delegate to as tools.

            tools: Tools to add to the agent.

            llm: LLM instance to use.

            output_handler: Output handler for the agent.

            verbose: Whether to log verbose output.
        """
        if tools is None:
            tools = []

        if llm is None:
            # CrewAI uses Langchain LLMs by default
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if delegation:
            raise ValueError(
                "'delegation' is not supported, pass the agents you want to delegate to as tools instead."
            )

        def agent_factory(tools: dict[str, MotleyTool]):
            langchain_tools = [t.to_langchain_tool() for t in tools.values()]
            agent = CrewAIAgentWithConfig(
                role=role,
                goal=goal,
                backstory=backstory,
                verbose=verbose,
                allow_delegation=False,
                tools=langchain_tools,
                llm=llm,
            )
            return agent

        super().__init__(
            goal=goal,
            prompt_prefix=prompt_prefix,
            description=description,
            name=role,
            agent_factory=agent_factory,
            tools=tools,
            output_handler=output_handler,
            verbose=verbose,
        )
