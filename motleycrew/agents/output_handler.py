from typing import Optional
from abc import ABC, abstractmethod
from langchain_core.tools import StructuredTool, BaseTool
from langchain_core.pydantic_v1 import BaseModel

from motleycrew.agents.abstract_parent import MotleyAgentAbstractParent
from motleycrew.common.exceptions import InvalidOutput
from motleycrew.tools import MotleyTool


class MotleyOutputHandler(MotleyTool, ABC):
    _name: str = "output_handler"
    """Name of the output handler tool."""

    _description: str = "Output handler. ONLY RETURN THE FINAL RESULT USING THIS TOOL!"
    """Description of the output handler tool."""

    _args_schema: Optional[BaseModel] = None
    """Pydantic schema for the arguments of the output handler tool.
    Inferred from the `handle_output` method if not provided."""

    _exceptions_to_handle: tuple[Exception] = (InvalidOutput,)
    """Exceptions that should be returned to the agent when raised in the `handle_output` method."""

    def __init__(self):
        langchain_tool = self._create_langchain_tool()
        super().__init__(langchain_tool)

        self.agent: Optional[MotleyAgentAbstractParent] = None
        self.agent_input: Optional[dict] = None

    @property
    def exceptions_to_handle(self):
        return self._exceptions_to_handle

    def _create_langchain_tool(self):
        return StructuredTool.from_function(
            name=self._name,
            description=self._description,
            args_schema=self._args_schema,
            func=self.handle_output,
        )

    @abstractmethod
    def handle_output(self, *args, **kwargs):
        pass
