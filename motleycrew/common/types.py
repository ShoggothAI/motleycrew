""" Module description"""
from typing import TYPE_CHECKING
from typing import Any
from typing import Protocol

if TYPE_CHECKING:
    from motleycrew.tools import MotleyTool

MotleySupportedTool = Any  # TODO: more specific typing for supported tools


class MotleyAgentFactory(Protocol):
    """
    Type protocol for an agent factory.
    It is a function that accepts tools as an argument and returns an agent instance of an appropriate class.
    """

    def __call__(self, tools: dict[str, "MotleyTool"]) -> Any: ...
