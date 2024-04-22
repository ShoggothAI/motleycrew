from typing import Any, Optional, Sequence, Callable

from langchain.agents import AgentExecutor
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate

from motleycrew.agent.shared import MotleyAgentParent
from motleycrew.tasks import Task

from motleycrew.tool import MotleyTool
from motleycrew.tracking import add_default_callbacks_to_langchain_config
from motleycrew.common import MotleySupportedTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm


class LangchainMotleyAgentParent(MotleyAgentParent):
    def __init__(
        self,
        goal: str,
        name: str | None = None,
        agent_factory: Callable[[dict[str, MotleyTool]], AgentExecutor] | None = None,
        delegation: bool | Sequence[MotleyAgentParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            goal=goal,
            name=name,
            agent_factory=agent_factory,
            delegation=delegation,
            tools=tools,
            verbose=verbose
        )

    def invoke(
        self,
        task: Task | str,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Task:
        self.materialize()

        config = add_default_callbacks_to_langchain_config(config)
        if isinstance(task, str):
            # TODO: feed in context/task.message_history correctly
            # TODO: attach the current task, if any, as a dependency of the new task
            new_task = Task(description=task, name=task, agent=self)
            return new_task.invoke()
        elif isinstance(task, Task):
            # TODO: feed in context/task.message_history correctly
            # TODO: extract the message history from the agent and attach it to the task
            out = self.agent.invoke({"input": task.description}, config, **kwargs)
            task.outputs = [out["output"]]
            return task
        else:
            raise ValueError(f"`task` must be a string or a Task, not {type(task)}")

    @staticmethod
    def from_function(
        function: Callable[..., Any],
        goal: str,
        llm: BaseLanguageModel | None = None,
        delegation: bool | Sequence[MotleyAgentParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        prompt: ChatPromptTemplate | Sequence[ChatPromptTemplate] | None = None,
        require_tools: bool = False,
        verbose: bool = False,
    ) -> "LangchainMotleyAgentParent":
        if llm is None:
            llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

        if require_tools and not len(tools):
            raise ValueError(
                "You must provide at least one tool to the ReactMotleyAgent"
            )

        def agent_factory(tools: dict[str, MotleyTool]):
            langchain_tools = [t.to_langchain_tool() for t in tools.values()]
            # TODO: feed goal into the agent's prompt
            agent = function(llm=llm, tools=langchain_tools, prompt=prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=langchain_tools,
                verbose=verbose,
            )
            return agent_executor

        return LangchainMotleyAgentParent(
            goal=goal,
            agent_factory=agent_factory,
            delegation=delegation,
            tools=tools,
            verbose=verbose,
        )

    @staticmethod
    def from_agent(
        agent: AgentExecutor,
        goal: str,
        delegation: bool | Sequence[MotleyAgentParent] = False,
        tools: Sequence[MotleySupportedTool] | None = None,
        verbose: bool = False
    ):
        # TODO: do we really need to unite the tools implicitly like this?
        # TODO: confused users might pass tools both ways at the same time
        # TODO: and we will silently unite them, which can have side effects (e.g. doubled tools)
        # TODO: besides, getting tools can be tricky for other frameworks (e.g. LlamaIndex)
        tools += agent.tools
        wrapped_agent = LangchainMotleyAgentParent(
            goal=goal,
            delegation=delegation,
            tools=tools,
            verbose=verbose
        )
        wrapped_agent._agent = agent
        return wrapped_agent
