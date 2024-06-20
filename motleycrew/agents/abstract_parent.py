""" Module description"""
from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from motleycrew.tools import MotleyTool


class MotleyAgentAbstractParent(ABC):
    @abstractmethod
    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """ Description

            Args:
                input (dict):
                config (:obj:`RunnableConfig`, optional):
                **kwargs:

            Returns:
                Any:
        """
        pass

    @abstractmethod
    def as_tool(self) -> "MotleyTool":
        """ Description

        Returns:
            MotleyTool

        """
        pass
