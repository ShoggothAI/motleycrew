from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Sequence, List, TYPE_CHECKING

from motleycrew.agent.parent import MotleyAgentAbstractParent
from motleycrew.tasks.simple import TaskDependencyCycleError

if TYPE_CHECKING:
    from motleycrew.crew import MotleyCrew

try:
    from crewai import Task as CrewAITask
except ImportError:
    CrewAITask = None


@dataclass
class Task:
    status: str = "pending"
    outputs: Optional[Any] = None

    def set_running(self):
        self.status = "running"

    def set_done(self):
        self.status = "done"

    @abstractmethod
    def to_crewai_task(self) -> CrewAITask:
        pass


@dataclass
class TaskRecipeNode:
    node_id: int


class TaskRecipe(ABC):
    def __init__(self, name: str):
        self.name = name
        self.crew: MotleyCrew | None = None

    def set_upstream(self, task: TaskRecipe) -> TaskRecipe:
        if self.crew is None or task.crew is None:
            raise ValueError("Both tasks must be registered with a crew")

        if task is self:
            raise TaskDependencyCycleError(f"Task {self.name} can not depend on itself")

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

    @abstractmethod
    def identify_candidates(self) -> List[Task]:
        pass

    @abstractmethod
    def get_agent(self) -> MotleyAgentAbstractParent:
        pass
