from __future__ import annotations

from abc import ABC
from typing import Optional, Any, TypeVar

from motleycrew.common import TaskStatus
from motleycrew.storage import MotleyGraphNode


class Task(MotleyGraphNode, ABC):
    status: str = TaskStatus.PENDING
    output: Optional[Any] = None

    def __repr__(self) -> str:
        return f"Task(status={self.status})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: Task):
        return self.id is not None and self.get_label() == other.get_label and self.id == other.id

    @property
    def pending(self):
        return self.status == TaskStatus.PENDING

    @property
    def running(self):
        return self.status == TaskStatus.RUNNING

    @property
    def done(self):
        return self.status == TaskStatus.DONE

    def set_pending(self):
        self.status = TaskStatus.PENDING

    def set_running(self):
        self.status = TaskStatus.RUNNING

    def set_done(self):
        self.status = TaskStatus.DONE

    def as_dict(self):
        """Represent the task as a dictionary for passing to invoke() methods of runnables."""
        return dict(self)


TaskType = TypeVar("TaskType", bound=Task)
