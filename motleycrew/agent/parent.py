from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Union

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig


class MotleyAgentAbstractParent(ABC):
    @abstractmethod
    def invoke(
        self,
        task: Union["Task", str],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> "Task":
        pass
