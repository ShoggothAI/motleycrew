""" Module description """
from typing import Any, Optional, List

from langchain_core.runnables import RunnableConfig
from langchain.tools.render import render_text_description

from motleycrew.common.utils import ensure_module_is_installed

try:
    from crewai import Agent
except ImportError:
    Agent = object


class CrewAIAgentWithConfig(Agent):

    def __init__(self, *args, **kwargs):
        """Subclass for CrewAI Agent that overrides the execute_task method to include a config parameter.

        Args:
            *args:
            **kwargs:

        Todo:
            * get rid of this when https://github.com/joaomdmoura/crewAI/pull/483 is merged.
        """
        ensure_module_is_installed("crewai")
        super(CrewAIAgentWithConfig, self).__init__(*args, **kwargs)

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task (Any): Task to execute.
            context (:obj:`str`, optional): Context to execute the task in.
            tools (:obj:`List[Any]`, optional): Tools to use for the task.
            config (:obj:`RunnableConfig`, optional): Runnable config.

        Returns:
            Any: Output of the agent
        """
        task_prompt = task.prompt()

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        tools = self._parse_tools(tools or self.tools)
        self.create_agent_executor(tools=tools)
        self.agent_executor.tools = tools
        self.agent_executor.task = task
        self.agent_executor.tools_description = render_text_description(tools)
        self.agent_executor.tools_names = self.__tools_names(tools)

        result = self.agent_executor.invoke(
            {
                "input": task_prompt,
                "tool_names": self.agent_executor.tools_names,
                "tools": self.agent_executor.tools_description,
            },
            config=config,
        )["output"]

        if self.max_rpm:
            self._rpm_controller.stop_rpm_counter()

        return result

    @staticmethod
    def __tools_names(tools) -> str:
        return ", ".join([t.name for t in tools])
