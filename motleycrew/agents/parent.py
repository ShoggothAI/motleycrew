""" Module description """
from typing import TYPE_CHECKING, Optional, Sequence

from langchain_core.tools import Tool
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.tools import MotleyTool
from motleycrew.common import MotleyAgentFactory, MotleySupportedTool
from motleycrew.common.exceptions import AgentNotMaterialized, CannotModifyMaterializedAgent
from motleycrew.common import logger

if TYPE_CHECKING:
    from motleycrew import MotleyCrew


class MotleyAgentParent(MotleyAgentAbstractParent, Runnable):
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
            verbose (:obj:`bool', optional):
        """
        self.name = name or description
        self.description = description  # becomes tool description
        self.agent_factory = agent_factory
        self.tools: dict[str, MotleyTool] = {}
        self.verbose = verbose
        self.crew: MotleyCrew | None = None

        self._agent = None

        if tools:
            self.add_tools(tools)

    def __repr__(self):
        return f"Agent(name={self.name})"

    def __str__(self):
        return self.__repr__()

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

    def materialize(self):
        if self.is_materialized:
            logger.info("Agent is already materialized, skipping materialization")
            return
        assert self.agent_factory, "Cannot materialize agent without a factory provided"
        self._agent = self.agent_factory(tools=self.tools)

    def add_tools(self, tools: Sequence[MotleySupportedTool]):
        """ Description

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
        """ Description

        Args:
            input_schema (:obj:`BaseModel`, optional):

        Returns:
            MotleyTool:
        """
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
