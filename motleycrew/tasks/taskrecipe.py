from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, Set, List
from uuid import uuid4

try:
    from crewai import Task as CrewAITask
except ImportError:
    CrewAITask = None

from motleycrew.agent.parent import MotleyAgentAbstractParent
from motleycrew.tool import MotleyTool

if TYPE_CHECKING:
    from motleycrew import MotleyCrew

PROMPT_TEMPLATE_WITH_DEPS = """
{description}

You must use the results of these upstream tasks:

{upstream_results_section}
"""


class TaskDependencyCycleError(Exception):
    """Raised when a task is set to depend on itself"""


class TaskRecipe:
    @dataclass
    class TaskRecipeNode:
        node_id: str
        name: str
        description: str
        done: bool
        outputs: List[str]

    @dataclass
    class TaskInput:
        node_id: int

    summary_type = TaskRecipeNode

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
        self.name = name  # does it really need one? Where does it get used?
        self.description = description
        self.agent = agent  # to be auto-assigned at crew creation if missing?
        self.tools = tools or []
        # should tasks own agents or should agents own tasks?
        self.documents = documents  # to be passed to an auto-init'd retrieval, later on
        self.creator_name = creator_name or "Human"
        self.return_to_creator = (
            return_to_creator  # for orchestrator to know to send back to creator
        )
        self.message_history = []  # Useful when task is passed around between agents
        self.outputs = []  # to be filled in by the agent(s) once the task is complete
        self.used_tools = 0  # a hack for CrewAI compatibility

        self.done: bool = False
        self.id = str(uuid4())

        # This will be set by MotleyCrew.register_task
        self.crew = None

    def get_agent(self) -> MotleyAgentAbstractParent:
        # TODO: implement this via agent factory
        if self.agent is None:
            raise ValueError("Task is not associated with an agent")
        return self.agent

    def summary(self) -> TaskRecipeNode:
        return TaskRecipe.TaskRecipeNode(
            id=self.id,
            name=self.name,
            description=self.description,
            done=self.done,
            outputs=self.outputs,
        )

    def identify_candidates(self) -> List["TaskInput"]:
        return not self.done and all(t.done for t in self.upstream_tasks)

    def set_upstream(self, task: TaskRecipe) -> TaskRecipe:
        if self.crew is None or task.crew is None:
            raise ValueError("Both tasks must be registered with a crew")

        if task is self:
            raise TaskDependencyCycleError(f"Task {task.name} can not depend on itself")

        self.crew.add_dependency(upstream=task, downstream=self)

        return self

    def __rshift__(self, other: TaskRecipe | Sequence[TaskRecipe]) -> TaskRecipe:
        if isinstance(other, TaskRecipe):
            tasks = {other}
        else:
            tasks = other

        for task in tasks:
            task.set_upstream(self)

        return self

    def __rrshift__(self, other: Sequence[TaskRecipe]) -> Sequence[TaskRecipe]:
        for task in other:
            self.set_upstream(task)
        return other

    def to_crewai_task(self) -> CrewAITask:
        if CrewAITask is None:
            raise ImportError("crewai is not installed")

        raise NotImplementedError("This method is not yet implemented")
        # return CrewAITask(
        #     name=self.name or self.description, description=self.description, prompt=self.prompt
        # )
