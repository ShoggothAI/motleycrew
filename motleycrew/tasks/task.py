from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, List, Type, TypeVar, Generic, TYPE_CHECKING

from langchain_core.runnables import Runnable

from motleycrew.common.exceptions import TaskDependencyCycleError
from motleycrew.storage import MotleyGraphStore, MotleyGraphNode, MotleyKuzuGraphStore
from motleycrew.tasks.task_unit import TaskUnitType
from motleycrew.tools import MotleyTool

if TYPE_CHECKING:
    from motleycrew.crew import MotleyCrew


class TaskNode(MotleyGraphNode):
    """Node representing a task in the graph.

    Attributes:
        name: Name of the task.
        done: Whether the task is done.

    """

    __label__ = "TaskNode"
    name: str
    done: bool = False

    def __eq__(self, other):
        return self.id is not None and self.get_label() == other.get_label() and self.id == other.id


TaskNodeType = TypeVar("TaskNodeType", bound=TaskNode)


class Task(ABC, Generic[TaskUnitType]):
    """Base class for describing tasks.

    This class is abstract and must be subclassed to implement the task logic.

    Attributes:
        NODE_CLASS: Class for representing task nodes, can be overridden.
        TASK_IS_UPSTREAM_LABEL: Label for indicating upstream tasks, can be overridden.
    """

    NODE_CLASS: Type[TaskNodeType] = TaskNode
    TASK_IS_UPSTREAM_LABEL = "task_is_upstream"

    def __init__(
        self,
        name: str,
        task_unit_class: Type[TaskUnitType],
        crew: Optional[MotleyCrew] = None,
        allow_async_units: bool = False,
    ):
        """Initialize the task.

        Args:
            name: Name of the task.
            task_unit_class: Class for representing task units.
            crew: Crew to which the task belongs.
                If not provided, the task should be registered with a crew later.
            allow_async_units: Whether the task allows asynchronous units.
                Default is False. If True, the task may be queried for the next unit even if it
                has other units in progress.
        """
        self.name = name
        self.done = False
        self.node = self.NODE_CLASS(name=name, done=self.done)
        self.crew = crew
        self.allow_async_units = allow_async_units

        self.task_unit_class = task_unit_class
        self.task_unit_belongs_label = "{}_belongs".format(self.task_unit_class.get_label())

        if crew is not None:
            crew.register_tasks([self])
            self.prepare_graph_store()

    def prepare_graph_store(self):
        """Prepare the graph store for storing tasks and their units."""
        if isinstance(self.graph_store, MotleyKuzuGraphStore):
            self.graph_store.ensure_node_table(self.NODE_CLASS)
            self.graph_store.ensure_node_table(self.task_unit_class)
            self.graph_store.ensure_relation_table(
                from_class=self.task_unit_class,
                to_class=self.NODE_CLASS,
                label=self.task_unit_belongs_label,
            )

    @property
    def graph_store(self) -> MotleyGraphStore:
        """The graph store where the task is stored.

        This is an alias for the graph store of the crew that the task belongs to.
        """
        if self.crew is None:
            raise ValueError("Task must be registered with a crew for accessing graph store")
        return self.crew.graph_store

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, done={self.done})"

    def __str__(self) -> str:
        return self.__repr__()

    def set_upstream(self, task: Task) -> Task:
        """Set a task as an upstream task for the current task.

        This means that the current task will not be queried for task units
        until the upstream task is marked as done.

        Args:
            task: Upstream task.
        """
        if self.crew is None or task.crew is None:
            raise ValueError("Both tasks must be registered with a crew")

        if task is self:
            raise TaskDependencyCycleError(f"Task {self.name} can not depend on itself")

        self.crew.add_dependency(upstream=task, downstream=self)

        return self

    def __rshift__(self, other: Task | Sequence[Task]) -> Task:
        """Syntactic sugar for setting tasks order with the ``>>`` operator.

        Args:
            other: Task or sequence of tasks to set as downstream.
        """
        if isinstance(other, Task):
            tasks = {other}
        else:
            tasks = other

        for task in tasks:
            task.set_upstream(self)

        return self

    def __rrshift__(self, other: Sequence[Task]) -> Sequence[Task]:
        """Syntactic sugar for setting tasks order with the ``>>`` operator.

        Args:
            other: Task or sequence of tasks to set as upstream.
        """
        for task in other:
            self.set_upstream(task)
        return other

    def get_units(self, status: Optional[str] = None) -> List[TaskUnitType]:
        """Get the units of the task that are already inserted in the graph.

        This method should be used for fetching the existing task units.

        Args:
            status: Status of the task units to filter by.

        Returns:
            List of task units.
        """
        assert self.crew is not None, "Task must be registered with a crew for accessing task units"

        query = (
            "MATCH (unit:{})-[:{}]->(task:{}) WHERE task.id = $self_id"
            + (" AND unit.status = $status" if status is not None else "")
            + " RETURN unit"
        ).format(
            self.task_unit_class.get_label(),
            self.task_unit_belongs_label,
            self.NODE_CLASS.get_label(),
        )

        parameters = {"self_id": self.node.id}
        if status is not None:
            parameters["status"] = status

        task_units = self.graph_store.run_cypher_query(
            query, parameters=parameters, container=self.task_unit_class
        )
        return task_units

    def get_upstream_tasks(self) -> List[Task]:
        """Get the upstream tasks of the current task.

        Returns:
            List of upstream tasks.
        """
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
        upstream_task_nodes = self.graph_store.run_cypher_query(
            query, parameters={"self_id": self.node.id}, container=self.NODE_CLASS
        )
        return [task for task in self.crew.tasks if task.node in upstream_task_nodes]

    def get_downstream_tasks(self) -> List[Task]:
        """Get the downstream tasks of the current task.

        Returns:
            List of downstream tasks.
        """
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
        downstream_task_nodes = self.graph_store.run_cypher_query(
            query, parameters={"self_id": self.node.id}, container=self.NODE_CLASS
        )
        return [task for task in self.crew.tasks if task.node in downstream_task_nodes]

    def set_done(self, value: bool = True):
        """Set the done status of the task.

        Args:
            value: Value to set the done status to.
        """
        self.done = value
        self.node.done = value

    def on_unit_dispatch(self, unit: TaskUnitType) -> None:
        """Method that is called by the crew when a unit of the task is dispatched.

        Should be implemented by the subclass if needed.

        Args:
            unit: Task unit that is dispatched.
        """
        pass

    def on_unit_completion(self, unit: TaskUnitType) -> None:
        """Method that is called by the crew when a unit of the task is completed.

        Should be implemented by the subclass if needed.

        Args:
            unit: Task unit that is completed.
        """
        pass

    @abstractmethod
    def get_next_unit(self) -> TaskUnitType | None:
        """Get the next unit of the task to run. Must be implemented by the subclass.

        This method is called in the crew's main loop repeatedly while the task is not done
        and there are units in progress.

        **Note that returning a unit does not guarantee that it will be dispatched.**
        Because of this, any state changes are strongly discouraged in this method.
        If you need to perform some actions when the unit is dispatched or completed,
        you should implement the ``on_unit_dispatch`` and/or ``on_unit_completion`` methods.

        If you need to find which units already exist in order to generate the next one,
        you can use the ``get_units`` method.

        Returns:
            Next unit to run, or None if there are no units to run at the moment.
        """

        pass

    @abstractmethod
    def get_worker(self, tools: Optional[List[MotleyTool]]) -> Runnable:
        """Get the worker that will run the task units.

        This method is called by the crew when a unit of the task is dispatched.
        The unit will be converted to a dictionary and passed to the worker's ``invoke`` method.

        Typically, the worker is an agent, but it can be any object
        that implements the Langchain Runnable interface.

        Args:
            tools: Tools to be used by the worker.

        Returns:
            Worker that will run the task units.
        """
        pass
