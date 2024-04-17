from abc import ABC, abstractmethod
from typing import Sequence

from motleycrew.task import Task


class TaskCoordinator(ABC):
    @abstractmethod
    def order(self, tasks: Sequence[Task]) -> Sequence[Task]:
        pass
