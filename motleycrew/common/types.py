"""Various types and type protocols used in motleycrew.

Attributes:
    MotleySupportedTool: Type that represents a tool that is supported by motleycrew.
       It includes tools from motleycrew, langchain, llama_index, and motleycrew agents.
"""

from __future__ import annotations

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
    """Type protocol for an agent factory.

    It is a function that accepts tools as an argument and returns an agent instance
    of an appropriate class.

    Agent factory is typically needed because the agent may need the list of available tools
    or other context at the time of its creation (e.g. to compose the prompt),
    and it may not be available at the time of the agent wrapper initialization.
    """

    def __call__(
        self,
        tools: dict[str, MotleyTool],
    ) -> AgentType: ...
