from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, List, Type, TypeVar, Generic, TYPE_CHECKING

from langchain_core.runnables import Runnable
from motleycrew.common.exceptions import TaskDependencyCycleError
from motleycrew.storage import MotleyGraphStore, MotleyGraphNode
from motleycrew.tasks import TaskUnit, TaskUnitType
from motleycrew.tools import MotleyTool

if TYPE_CHECKING:
    from motleycrew.crew import MotleyCrew


class TaskNode(MotleyGraphNode):
    __label__ = "TaskNode"
    name: str
    done: bool = False

    def __eq__(self, other):
        return self.id is not None and self.get_label() == other.get_label() and self.id == other.id


TaskNodeType = TypeVar("TaskNodeType", bound=TaskNode)


class Task(ABC, Generic[TaskUnitType]):
    NODE_CLASS: Type[TaskNodeType] = TaskNode
    TASK_UNIT_CLASS: Type[TaskUnitType] = TaskUnit
    TASK_IS_UPSTREAM_LABEL = "task_is_upstream"
    TASK_UNIT_BELONGS_LABEL = "task_unit_belongs"

    def __init__(self, name: str, crew: Optional[MotleyCrew] = None):
        self.name = name
        self.done = False
        self.node = self.NODE_CLASS(name=name, done=self.done)
        self.crew = crew
        if crew is not None:
            crew.register_tasks([self])

    @property
    def graph_store(self) -> MotleyGraphStore:
        if self.crew is None:
            raise ValueError("Task must be registered with a crew for accessing graph store")
        return self.crew.graph_store

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, done={self.done})"

    def __str__(self) -> str:
        return self.__repr__()

    def set_upstream(self, task: Task) -> Task:
        if self.crew is None or task.crew is None:
            raise ValueError("Both tasks must be registered with a crew")

        if task is self:
            raise TaskDependencyCycleError(f"Task {self.name} can not depend on itself")

        self.crew.add_dependency(upstream=task, downstream=self)

        return self

    def __rshift__(self, other: Task | Sequence[Task]) -> Task:
        if isinstance(other, Task):
            tasks = {other}
        else:
            tasks = other

        for task in tasks:
            task.set_upstream(self)

        return self

    def __rrshift__(self, other: Sequence[Task]) -> Sequence[Task]:
        for task in other:
            self.set_upstream(task)
        return other

    def get_units(self) -> List[TaskUnitType]:
        assert self.crew is not None, "Task must be registered with a crew for accessing task units"

        query = "MATCH (unit)-[{}]->(task:{}) WHERE task.id = $self_id RETURN unit".format(
            self.TASK_UNIT_BELONGS_LABEL,
            self.NODE_CLASS.get_label(),
        )
        task_units = self.crew.graph_store.run_cypher_query(
            query, parameters={"self_id": self.node.id}, container=self.TASK_UNIT_CLASS
        )
        return task_units

    def get_upstream_tasks(self) -> List[Task]:
        assert (
            self.crew is not None and self.node.is_inserted
        ), "Task must be registered with a crew for accessing upstream tasks"

        query = (
            "MATCH (upstream)-[:{}]->(downstream:{}) "
            "WHERE downstream.id = $self_id "
            "RETURN upstream"
        ).format(
            self.TASK_IS_UPSTREAM_LABEL,
            self.NODE_CLASS.get_label(),
        )
        upstream_task_nodes = self.crew.graph_store.run_cypher_query(
            query, parameters={"self_id": self.node.id}, container=self.NODE_CLASS
        )
        return [task for task in self.crew.tasks if task.node in upstream_task_nodes]

    def get_downstream_tasks(self) -> List[Task]:
        assert (
            self.crew is not None and self.node.is_inserted
        ), "Task must be registered with a crew for accessing downstream tasks"

        query = (
            "MATCH (upstream:{})-[:{}]->(downstream) "
            "WHERE upstream.id = $self_id "
            "RETURN downstream"
        ).format(
            self.NODE_CLASS.get_label(),
            self.TASK_IS_UPSTREAM_LABEL,
        )
        downstream_task_nodes = self.crew.graph_store.run_cypher_query(
            query, parameters={"self_id": self.node.id}, container=self.NODE_CLASS
        )
        return [task for task in self.crew.tasks if task.node in downstream_task_nodes]

    def set_done(self, value: bool = True):
        self.done = value
        self.node.done = value

    def register_completed_unit(self, task: TaskUnitType) -> None:
        pass

    @abstractmethod
    def get_next_unit(self) -> TaskUnitType | None:
        pass

    @abstractmethod
    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        pass
