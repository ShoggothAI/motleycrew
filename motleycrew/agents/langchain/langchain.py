from typing import Any, Optional, Sequence, Callable, Union

from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.tools import BaseTool

from motleycrew.agents.parent import MotleyAgentParent

from motleycrew.tools import MotleyTool
from motleycrew.tracking import add_default_callbacks_to_langchain_config
from motleycrew.common import MotleySupportedTool
from motleycrew.common import MotleyAgentFactory
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm


class LangchainMotleyAgent(MotleyAgentParent):
    def __init__(
        self,
        description: str,
        name: str | None = None,
        agent_factory: MotleyAgentFactory[AgentExecutor] | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ):
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
        self.materialize()

        prompt = task_dict.get("prompt")
        if not prompt:
            raise ValueError("Task must have a prompt")

        config = add_default_callbacks_to_langchain_config(config)

        result = self.agent.invoke({"input": prompt}, config, **kwargs)
        output = result.get("output")
        if output is None:
            raise Exception("Agent {} result does not contain output: {}".format(self, result))
        return output

    @staticmethod
    def from_creating_function(
        creating_function: Callable[
            [
                BaseLanguageModel,
                Sequence[BaseTool],
                Union[BasePromptTemplate, Sequence[BasePromptTemplate], None],
            ],
            Any,
        ],
        description: str,
        name: str | None = None,
        llm: BaseLanguageModel | None = None,
        tools: Sequence[MotleySupportedTool] | None = None,
        prompt: BasePromptTemplate | Sequence[BasePromptTemplate] | None = None,
        require_tools: bool = False,
        verbose: bool = False,
    ) -> "LangchainMotleyAgent":
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if require_tools and not tools:
            raise ValueError("You must provide at least one tool to the ReactMotleyAgent")

        for tool in tools:
            MotleyTool.from_supported_tool(tool)

        def agent_factory(tools: dict[str, MotleyTool]) -> AgentExecutor:
            langchain_tools = [t.to_langchain_tool() for t in tools.values()]
            # TODO: feed description into the agent's prompt
            agent = creating_function(llm, langchain_tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=langchain_tools,
                verbose=verbose,
            )
            return agent_executor

        return LangchainMotleyAgent(
            description=description,
            name=name,
            agent_factory=agent_factory,
            tools=tools,
            verbose=verbose,
        )

    @staticmethod
    def from_agent(
        agent: AgentExecutor,
        goal: str,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ) -> "LangchainMotleyAgent":
        # TODO: do we really need to unite the tools implicitly like this?
        # TODO: confused users might pass tools both ways at the same time
        # TODO: and we will silently unite them, which can have side effects (e.g. doubled tools)
        # TODO: besides, getting tools can be tricky for other frameworks (e.g. LlamaIndex)
        if tools or agent.tools:
            tools = list(tools or []) + list(agent.tools or [])

        wrapped_agent = LangchainMotleyAgent(description=goal, tools=tools, verbose=verbose)
        wrapped_agent._agent = agent
        return wrapped_agent
