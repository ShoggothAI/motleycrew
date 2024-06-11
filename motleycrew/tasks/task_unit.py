""" Module description

Attributes:
    TaskUnitType (TypeVar):

"""
from __future__ import annotations

from abc import ABC
from typing import Optional, Any, TypeVar

from motleycrew.common import TaskUnitStatus
from motleycrew.storage import MotleyGraphNode


class TaskUnit(MotleyGraphNode, ABC):
    """ Description

    Attributes:
        status (:obj:`str`, optional):
        output (:obj:`Any`, optional):

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
        return self.status == TaskUnitStatus.PENDING

    @property
    def running(self):
        return self.status == TaskUnitStatus.RUNNING

    @property
    def done(self):
        return self.status == TaskUnitStatus.DONE

    def set_pending(self):
        self.status = TaskUnitStatus.PENDING

    def set_running(self):
        self.status = TaskUnitStatus.RUNNING

    def set_done(self):
        self.status = TaskUnitStatus.DONE

    def as_dict(self):
        """Represent the task as a dictionary for passing to invoke() methods of runnables."""
        return dict(self)


TaskUnitType = TypeVar("TaskUnitType", bound=TaskUnit)
