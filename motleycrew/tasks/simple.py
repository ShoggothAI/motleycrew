""" Module description

Attributes:
   PROMPT_TEMPLATE_WITH_DEPS (str):

"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Sequence, List, Optional

from motleycrew.tasks.task import Task
from motleycrew.tasks import TaskUnit

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.tools import MotleyTool
from motleycrew.common import logger

if TYPE_CHECKING:
    from motleycrew.crew import MotleyCrew


PROMPT_TEMPLATE_WITH_DEPS = """
{description}

You must use the results of these upstream tasks:

{upstream_results_section}
"""


def compose_simple_task_prompt_with_dependencies(
    description: str, upstream_task_units: List[TaskUnit], default_task_name: str = "Unnamed task"
) -> str:
    """ Description

    Args:
        description (str):
        upstream_task_units (:obj:`list` of :obj:`TaskUnit`):
        default_task_name (:obj:`str`, optional):

    Returns:
        str:
    """
    upstream_results = []
    for unit in upstream_task_units:
        if not unit.output:
            continue

        unit_name = getattr(unit, "name", default_task_name)
        upstream_results.append(f"##{unit_name}\n" + str(unit.output))

    if not upstream_results:
        return description

    upstream_results_section = "\n\n".join(upstream_results)
    return PROMPT_TEMPLATE_WITH_DEPS.format(
        description=description,
        upstream_results_section=upstream_results_section,
    )


class SimpleTaskUnit(TaskUnit):
    """ Description

    Attributes:
        name (str):
        prompt (str):
        message_history (:obj:`list` of :obj:`str`):

    """
    name: str
    prompt: str
    message_history: List[str] = []


class SimpleTask(Task):
    def __init__(
        self,
        crew: MotleyCrew,
        description: str,
        name: str | None = None,
        agent: MotleyAgentAbstractParent | None = None,
        tools: Sequence[MotleyTool] | None = None,
        documents: Sequence[Any] | None = None,
        creator_name: str | None = None,
        return_to_creator: bool = False,
    ):
        """ Description

        Args:
            crew (MotleyCrew):
            description (str):
            name (:obj:`str`, optional):
            agent (:obj:`MotleyAgentAbstractParent`, optional):
            tools (:obj:`Sequence[MotleyTool]`, optional):
            documents (:obj:`Sequence[Any]`, optional):
            creator_name (:obj:`str`, optional):
            return_to_creator (:obj:`bool`, optional):
        """
        super().__init__(name=name or description, task_unit_class=SimpleTaskUnit, crew=crew)
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

    def register_completed_unit(self, unit: SimpleTaskUnit) -> None:
        """ Description

        Args:
            unit (SimpleTaskUnit):

        Returns:

        """
        assert isinstance(unit, SimpleTaskUnit)
        assert unit.done

        self.output = unit.output
        self.set_done()

    def get_next_unit(self) -> SimpleTaskUnit | None:
        """ Description

        Returns:
            :obj:`SimpleTaskUnit`, None:
        """
        if self.done:
            logger.info("Task %s is already done", self)
            return None

        upstream_tasks = self.get_upstream_tasks()
        if not all(task.done for task in upstream_tasks):
            return None

        upstream_task_units = [unit for task in upstream_tasks for unit in task.get_units()]
        prompt = compose_simple_task_prompt_with_dependencies(self.description, upstream_task_units)
        return SimpleTaskUnit(
            name=self.name,
            prompt=prompt,
        )

    def get_worker(self, tools: Optional[List[MotleyTool]]) -> MotleyAgentAbstractParent:
        """ Description

        Args:
            tools (:obj:`List[MotleyTool]`, :obj:`None`):

        Returns:
            MotleyAgentAbstractParent
        """
        if self.crew is None:
            raise ValueError("Task is not associated with a crew")
        if self.agent is None:
            raise ValueError("Task is not associated with an agent")

        if hasattr(self.agent, "is_materialized") and self.agent.is_materialized and tools:
            logger.warning(
                "Agent %s is already materialized, can't add extra tools %s", self.agent, tools
            )

        if hasattr(self.agent, "add_tools") and tools:
            logger.info("Adding tools %s to agent %s", tools, self.agent)
            self.agent.add_tools(tools)

        # TODO: that's a pretty big assumption on agent structure. Necessary?
        self.agent.crew = self.crew

        return self.agent
