from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Any, Union

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from motleycrew.tasks import TaskRecipe
    from motleycrew.tool import MotleyTool


class MotleyAgentAbstractParent(ABC):
    @abstractmethod
    def invoke(
        self,
        task: Union["SimpleTaskRecipe", str],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> "SimpleTaskRecipe":
        pass

    @abstractmethod
    def as_tool(self) -> "MotleyTool":
        pass
