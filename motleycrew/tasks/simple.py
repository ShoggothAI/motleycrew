from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, List, Optional

from langchain_core.prompts import PromptTemplate

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.common import logger
from motleycrew.tasks.task import Task
from motleycrew.tasks.task_unit import TaskUnit
from motleycrew.tools import MotleyTool

if TYPE_CHECKING:
    from motleycrew.crew import MotleyCrew


PROMPT_TEMPLATE_WITH_UPSTREAM_TASKS = PromptTemplate.from_template(
    """{description}

You must use the results of these upstream tasks:

{upstream_results}
"""
)


def compose_simple_task_prompt_with_dependencies(
    description: str,
    upstream_task_units: List[TaskUnit],
    prompt_template_with_upstreams: PromptTemplate,
    default_task_name: str = "Unnamed task",
) -> str:
    """Compose a prompt for a simple task with upstream dependencies.

    Args:
        description: Description of the task, to be included in the prompt.
        upstream_task_units: List of upstream task units whose results should be used.
        prompt_template_with_upstreams: Prompt template to use for generating the prompt
            if the task has upstream dependencies. Otherwise, just the description is used.
            The template must have input variables 'description' and 'upstream_results'.
        default_task_name: Name to use for task units that don't have a ``name`` attribute.
    """
    if set(prompt_template_with_upstreams.input_variables) != {
        "description",
        "upstream_results",
    }:
        raise ValueError(
            "Prompt template must have input variables 'description' and 'upstream_results'"
        )
    upstream_results = []
    for unit in upstream_task_units:
        if not unit.output:
            continue

        unit_name = getattr(unit, "name", default_task_name)
        upstream_results.append(f"##{unit_name}\n" + str(unit.output))

    if not upstream_results:
        return description

    upstream_results_section = "\n\n".join(upstream_results)
    return prompt_template_with_upstreams.format(
        description=description,
        upstream_results=upstream_results_section,
    )


class SimpleTaskUnit(TaskUnit):
    """Task unit for a simple task.

    Attributes:
        name: Name of the task unit.
        prompt: Prompt for the task unit.
        additional_params: Additional parameters for the task unit (can be used by the agent).
    """

    name: str
    prompt: str
    additional_params: Optional[dict[str, Any]] = None


class SimpleTask(Task):
    """Simple task class.

    A simple task consists of a description and an agent that can execute the task.
    It produces a single task unit with a prompt based on the description
    and the results of upstream tasks.

    The task is considered done when the task unit is completed.
    """

    def __init__(
        self,
        crew: MotleyCrew,
        description: str,
        name: str | None = None,
        agent: MotleyAgentAbstractParent | None = None,
        tools: Sequence[MotleyTool] | None = None,
        additional_params: dict[str, Any] | None = None,
        prompt_template_with_upstreams: PromptTemplate = PROMPT_TEMPLATE_WITH_UPSTREAM_TASKS,
    ):
        """Initialize the simple task.

        Args:
            crew: Crew to which the task belongs.
            description: Description of the task.
            name: Name of the task (will be used as the name of the task unit).
            agent: Agent to execute the task.
            tools: Tools to use for the task.
            additional_params: Additional parameters for the task.
            prompt_template_with_upstreams: Prompt template to use for generating the prompt
                if the task has upstream dependencies. Otherwise, just the description is used.
                The template must have input variables 'description' and 'upstream_results'.
        """

        super().__init__(name=name or description, task_unit_class=SimpleTaskUnit, crew=crew)
        self.description = description
        self.agent = agent  # to be auto-assigned at crew creation if missing?
        self.tools = tools or []
        self.additional_params = additional_params or {}
        self.prompt_template_with_upstreams = prompt_template_with_upstreams

        self.output = None  # to be filled in by the agent(s) once the task is complete

    def on_unit_completion(self, unit: SimpleTaskUnit) -> None:
        """Handle completion of the task unit.

        Sets the task as done and stores the output of the task unit.

        Args:
            unit: Task unit that has completed.
        """
        assert isinstance(unit, SimpleTaskUnit)
        assert unit.done

        self.output = unit.output
        self.set_done()

    def get_next_unit(self) -> SimpleTaskUnit | None:
        """Get the next task unit to run.

        If all upstream tasks are done, returns a task unit with the prompt
        based on the description and the results of the upstream tasks.
        Otherwise, returns None (the task is not ready to run yet).

        Returns:
            Task unit to run if the task is ready, None otherwise.
        """
        if self.done:
            logger.info("Task %s is already done", self)
            return None

        upstream_tasks = self.get_upstream_tasks()
        if not all(task.done for task in upstream_tasks):
            return None

        upstream_task_units = [unit for task in upstream_tasks for unit in task.get_units()]
        prompt = compose_simple_task_prompt_with_dependencies(
            description=self.description,
            upstream_task_units=upstream_task_units,
            prompt_template_with_upstreams=self.prompt_template_with_upstreams,
        )
        return SimpleTaskUnit(
            name=self.name,
            prompt=prompt,
            additional_params=self.additional_params,
        )

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> MotleyAgentAbstractParent:
        """Get the worker for the task.

        If the task is associated with a crew and an agent, returns the agent.
        Otherwise, raises an exception.

        Args:
            tools: Additional tools to add to the agent.

        Returns:
            Agent to run the task unit.
        """
        if self.crew is None:
            raise ValueError("Task is not associated with a crew")
        if self.agent is None:
            raise ValueError("Task is not associated with an agent")

        if hasattr(self.agent, "is_materialized") and self.agent.is_materialized and tools:
            logger.warning(
                "Agent %s is already materialized, can't add extra tools %s",
                self.agent,
                tools,
            )

        if hasattr(self.agent, "add_tools") and tools:
            logger.info("Adding tools %s to agent %s", tools, self.agent)
            self.agent.add_tools(tools)

        # TODO: that's a pretty big assumption on agent structure. Necessary?
        self.agent.crew = self.crew

        return self.agent
