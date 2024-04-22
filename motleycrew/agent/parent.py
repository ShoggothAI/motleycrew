from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Any, Union

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from motleycrew.tasks import Task


class MotleyAgentAbstractParent(ABC):
    @abstractmethod
    def invoke(
        self,
        task: Union["Task", str],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> "Task":
        pass
