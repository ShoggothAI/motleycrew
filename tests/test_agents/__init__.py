from typing import Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class MockToolInput(BaseModel):
    """Input for the MockTool tool."""

    tool_input: str = Field(description="tool_input")


class MockTool(BaseTool):
    """Mock tool for run agent tests"""

    name: str = "mock tool"
    description: str = "Mock tool for tests"

    args_schema: Type[BaseModel] = MockToolInput

    def _run(self, tool_input: str, *args, **kwargs):
        return tool_input
