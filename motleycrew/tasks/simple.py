from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Sequence, List, Optional

from motleycrew.tasks.task_recipe import TaskRecipe
from motleycrew.tasks import Task

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
    description: str, upstream_tasks: List[Task], default_task_name: str = "Unnamed task"
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


class SimpleTask(Task):
    name: str
    prompt: str
    message_history: List[str] = []


class SimpleTaskRecipe(TaskRecipe):
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

    def register_completed_task(self, task: SimpleTask) -> None:
        assert isinstance(task, SimpleTask)
        assert task.done

        self.output = task.output
        self.set_done()

    def identify_candidates(self) -> List[SimpleTask]:
        if self.done:
            logging.info("Task %s is already done", self)
            return []

        upstream_task_recipes = self.get_upstream_task_recipes()
        if not all(recipe.done for recipe in upstream_task_recipes):
            return []

        upstream_tasks = [task for recipe in upstream_task_recipes for task in recipe.get_tasks()]
        return [
            SimpleTask(
                name=self.name,
                prompt=compose_simple_task_prompt_with_dependencies(
                    self.description, upstream_tasks
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
