from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Sequence, List, Optional

from motleycrew.tasks.task import Task
from motleycrew.tasks import TaskUnit

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.tools import MotleyTool

if TYPE_CHECKING:
    pass


PROMPT_TEMPLATE_WITH_DEPS = """
{description}

You must use the results of these upstream tasks:

{upstream_results_section}
"""


def compose_simple_task_prompt_with_dependencies(
    description: str, upstream_tasks: List[TaskUnit], default_task_name: str = "Unnamed task"
) -> str:
    upstream_results = []
    for task in upstream_tasks:
        if not task.output:
            continue

        task_name = getattr(task, "name", default_task_name)
        upstream_results.append(f"##{task_name}\n" + "\n".join(str(out) for out in task.output))

    if not upstream_results:
        return description

    upstream_results_section = "\n\n".join(upstream_results)
    return PROMPT_TEMPLATE_WITH_DEPS.format(
        description=description,
        upstream_results_section=upstream_results_section,
    )


class SimpleTaskUnit(TaskUnit):
    name: str
    prompt: str
    message_history: List[str] = []


class SimpleTask(Task):
    def __init__(
        self,
        description: str,
        name: str | None = None,
        agent: MotleyAgentAbstractParent | None = None,
        tools: Sequence[MotleyTool] | None = None,
        documents: Sequence[Any] | None = None,
        creator_name: str | None = None,
        return_to_creator: bool = False,
    ):
        super().__init__(name or description)
        self.description = description
        self.agent = agent  # to be auto-assigned at crew creation if missing?
        self.tools = tools or []
        # should tasks own agents or should agents own tasks?
        self.documents = documents  # to be passed to an auto-init'd retrieval, later on
        self.creator_name = creator_name or "Human"
        self.return_to_creator = (
            return_to_creator  # for orchestrator to know to send back to creator
        )
        self.output = None  # to be filled in by the agent(s) once the task is complete

        # This will be set by MotleyCrew.register_task
        self.crew = None

    def register_completed_unit(self, task: SimpleTaskUnit) -> None:
        assert isinstance(task, SimpleTaskUnit)
        assert task.done

        self.output = task.output
        self.set_done()

    def identify_candidates(self) -> List[SimpleTaskUnit]:
        if self.done:
            logging.info("Task %s is already done", self)
            return []

        upstream_tasks = self.get_upstream_tasks()
        if not all(task.done for task in upstream_tasks):
            return []

        upstream_task_units = [unit for task in upstream_tasks for unit in task.get_units()]
        return [
            SimpleTaskUnit(
                name=self.name,
                prompt=compose_simple_task_prompt_with_dependencies(
                    self.description, upstream_task_units
                ),
            )
        ]

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> MotleyAgentAbstractParent:
        if self.crew is None:
            raise ValueError("Task is not associated with a crew")
        if self.agent is None:
            raise ValueError("Task is not associated with an agent")

        if hasattr(self.agent, "is_materialized") and self.agent.is_materialized and tools:
            logging.warning(
                "Agent %s is already materialized, can't add extra tools %s", self.agent, tools
            )

        if hasattr(self.agent, "add_tools") and tools:
            logging.info("Adding tools %s to agent %s", tools, self.agent)
            self.agent.add_tools(tools)

        # TODO: that's a pretty big assumption on agent structure. Necessary?
        self.agent.crew = self.crew

        return self.agent
