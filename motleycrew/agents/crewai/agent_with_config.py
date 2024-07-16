from typing import Any, Optional, List

from langchain.tools.render import render_text_description
from langchain_core.runnables import RunnableConfig

from motleycrew.common.utils import ensure_module_is_installed

try:
    from crewai import Agent
    from crewai.memory.contextual.contextual_memory import ContextualMemory
except ImportError:
    Agent = object
    ContextualMemory = object


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
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        if self.tools_handler:
            # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")
            self.tools_handler.last_used_tool = {}

        task_prompt = task.prompt()

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            contextual_memory = ContextualMemory(
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        tools = tools or self.tools
        # type: ignore # Argument 1 to "_parse_tools" of "Agent" has incompatible type "list[Any] | None"; expected "list[Any]"
        parsed_tools = self._parse_tools(tools or [])
        self.create_agent_executor(tools=tools)
        self.agent_executor.tools = parsed_tools
        self.agent_executor.task = task

        self.agent_executor.tools_description = render_text_description(parsed_tools)
        self.agent_executor.tools_names = self.__tools_names(parsed_tools)

        if self.crew and self.crew._train:
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

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
