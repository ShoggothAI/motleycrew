from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, List, Type, TypeVar, Generic, TYPE_CHECKING

from langchain_core.runnables import Runnable
from motleycrew.common.exceptions import TaskDependencyCycleError
from motleycrew.storage import MotleyGraphStore, MotleyGraphNode
from motleycrew.tasks import Task, TaskType
from motleycrew.tool import MotleyTool

if TYPE_CHECKING:
    from motleycrew.crew import MotleyCrew


class TaskRecipeNode(MotleyGraphNode):
    __label__ = "TaskRecipeNode"
    name: str
    done: bool = False

    def __eq__(self, other):
        return self.id is not None and self.get_label() == other.get_label() and self.id == other.id


TaskRecipeNodeType = TypeVar("TaskRecipeNodeType", bound=TaskRecipeNode)


class TaskRecipe(ABC, Generic[TaskType]):
    NODE_CLASS: Type[TaskRecipeNodeType] = TaskRecipeNode
    TASK_CLASS: Type[TaskType] = Task
    TASK_RECIPE_IS_UPSTREAM_LABEL = "task_recipe_is_upstream"
    TASK_BELONGS_LABEL = "task_belongs"

    def __init__(self, name: str, crew: Optional[MotleyCrew] = None):
        self.name = name
        self.done = False
        self.node = self.NODE_CLASS(name=name, done=self.done)
        self.crew = crew
        if crew is not None:
            crew.register_task_recipes([self])

    @property
    def graph_store(self) -> MotleyGraphStore:
        if self.crew is None:
            raise ValueError("TaskRecipe must be registered with a crew for accessing graph store")
        return self.crew.graph_store

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, done={self.done})"

    def __str__(self) -> str:
        return self.__repr__()

    def set_upstream(self, task_recipe: TaskRecipe) -> TaskRecipe:
        if self.crew is None or task_recipe.crew is None:
            raise ValueError("Both tasks must be registered with a crew")

        if task_recipe is self:
            raise TaskDependencyCycleError(f"Task {self.name} can not depend on itself")

        self.crew.add_dependency(upstream=task_recipe, downstream=self)

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

    def get_tasks(self) -> List[TaskType]:
        assert (
            self.crew is not None
        ), "TaskRecipe must be registered with a crew for accessing tasks"

        query = "MATCH (task)-[{}]->(recipe:{}) WHERE recipe.id = $self_id RETURN task".format(
            self.TASK_BELONGS_LABEL,
            self.NODE_CLASS.get_label(),
        )
        tasks = self.crew.graph_store.run_cypher_query(
            query, parameters={"self_id": self.node.id}, container=self.TASK_CLASS
        )
        return tasks

    def get_upstream_task_recipes(self) -> List[TaskRecipe]:
        assert (
            self.crew is not None and self.node.is_inserted
        ), "TaskRecipe must be registered with a crew for accessing upstream tasks"

        query = (
            "MATCH (upstream)-[:{}]->(downstream:{}) "
            "WHERE downstream.id = $self_id "
            "RETURN upstream"
        ).format(
            self.TASK_RECIPE_IS_UPSTREAM_LABEL,
            self.NODE_CLASS.get_label(),
        )
        upstream_task_recipe_nodes = self.crew.graph_store.run_cypher_query(
            query, parameters={"self_id": self.node.id}, container=self.NODE_CLASS
        )
        return [
            recipe for recipe in self.crew.task_recipes if recipe.node in upstream_task_recipe_nodes
        ]

    def get_downstream_task_recipes(self) -> List[TaskRecipe]:
        assert (
            self.crew is not None and self.node.is_inserted
        ), "TaskRecipe must be registered with a crew for accessing upstream tasks"

        query = (
            "MATCH (upstream:{})-[:{}]->(downstream) "
            "WHERE upstream.id = $self_id "
            "RETURN downstream"
        ).format(
            self.NODE_CLASS.get_label(),
            self.TASK_RECIPE_IS_UPSTREAM_LABEL,
        )
        downstream_task_recipe_nodes = self.crew.graph_store.run_cypher_query(
            query, parameters={"self_id": self.node.id}, container=self.NODE_CLASS
        )
        return [
            recipe
            for recipe in self.crew.task_recipes
            if recipe.node in downstream_task_recipe_nodes
        ]

    def set_done(self, value: bool = True):
        self.done = value
        self.node.done = value

    def register_completed_task(self, task: TaskType) -> None:
        pass

    @abstractmethod
    def identify_candidates(self) -> List[TaskType]:
        pass

    @abstractmethod
    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        pass
