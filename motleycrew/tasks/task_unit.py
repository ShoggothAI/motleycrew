from __future__ import annotations

from abc import ABC
from typing import Optional, Any, TypeVar

from motleycrew.common import TaskUnitStatus
from motleycrew.storage import MotleyGraphNode


class TaskUnit(MotleyGraphNode, ABC):
    """Base class for describing task units.
    A task unit should contain all the input data for the worker (usually an agent).
    When a task unit is dispatched by the crew, it is converted to a dictionary
    and passed to the worker's ``invoke()`` method.

    Attributes:
        status: Status of the task unit.
        output: Output of the task unit.

    """

    status: str = TaskUnitStatus.PENDING
    output: Optional[Any] = None

    def __repr__(self) -> str:
        return f"TaskUnit(status={self.status})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: TaskUnit):
        return self.id is not None and self.get_label() == other.get_label and self.id == other.id

    @property
    def pending(self):
        """Whether the task unit is pending."""
        return self.status == TaskUnitStatus.PENDING

    @property
    def running(self):
        """Whether the task unit is running."""
        return self.status == TaskUnitStatus.RUNNING

    @property
    def done(self):
        """Whether the task unit is done."""
        return self.status == TaskUnitStatus.DONE

    def set_pending(self):
        """Set the task unit status to pending."""
        self.status = TaskUnitStatus.PENDING

    def set_running(self):
        """Set the task unit status to running."""
        self.status = TaskUnitStatus.RUNNING

    def set_done(self):
        """Set the task unit status to done."""
        self.status = TaskUnitStatus.DONE

    def as_dict(self):
        """Represent the task as a dictionary for passing to invoke() methods of runnables."""
        return dict(self)


TaskUnitType = TypeVar("TaskUnitType", bound=TaskUnit)
