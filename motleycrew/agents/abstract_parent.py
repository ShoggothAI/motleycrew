from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from motleycrew.tools import MotleyTool


class MotleyAgentAbstractParent(ABC):
    @abstractmethod
    def invoke(
        self,
        task_dict: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def as_tool(self) -> "MotleyTool":
        pass
