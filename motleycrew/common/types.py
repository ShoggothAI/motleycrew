""" Module description"""
from typing import TYPE_CHECKING, Union, Optional, Protocol, TypeVar

if TYPE_CHECKING:
    from langchain.tools import BaseTool

    try:
        from llama_index.core.tools import BaseTool as LlamaIndex__BaseTool
    except ImportError:
        LlamaIndex__BaseTool = "LlamaIndex__BaseTool"

    from motleycrew.tools import MotleyTool
    from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent

else:
    MotleyTool = "MotleyTool"
    BaseTool = "BaseTool"
    LlamaIndex__BaseTool = "LlamaIndex__BaseTool"
    MotleyAgentAbstractParent = "MotleyAgentAbstractParent"


MotleySupportedTool = Union[MotleyTool, BaseTool, LlamaIndex__BaseTool, MotleyAgentAbstractParent]


AgentType = TypeVar("AgentType")


class MotleyAgentFactory(Protocol[AgentType]):
    """
    Type protocol for an agent factory.
    It is a function that accepts tools as an argument
    and returns an agent instance of an appropriate class.
    """

    def __call__(
        self,
        tools: dict[str, MotleyTool],
        output_handler: Optional[MotleyTool] = None,
    ) -> AgentType: ...
