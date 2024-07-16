from abc import ABC, abstractmethod
from typing import Optional, Any, TYPE_CHECKING

from langchain_core.runnables import Runnable, RunnableConfig

if TYPE_CHECKING:
    from motleycrew.tools import MotleyTool


class MotleyAgentAbstractParent(Runnable, ABC):
    """Abstract class for describing agents.

    Agents in motleycrew implement the Langchain Runnable interface.
    """

    @abstractmethod
    def invoke(
        self,
        input: dict,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def as_tool(self) -> "MotleyTool":
        """Convert the agent to a MotleyTool to be used by other agents via delegation.

        Returns:
            The tool representation of the agent.
        """
        pass
