import pytest

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from motleycrew.tools import MotleyTool


@pytest.fixture
def mock_tool_args_schema():
    class MockToolInput(BaseModel):
        mock_input: str = Field(description="mock")

    return MockToolInput


def mock_tool_function(mock_input: str):
    return mock_input


@pytest.fixture
def mock_input():
    return {"mock_input": "some_value"}


@pytest.fixture
def langchain_tool(mock_tool_args_schema):
    from langchain.tools import Tool

    return Tool.from_function(
        func=mock_tool_function,
        name="mock_tool",
        description="mock_description",
        args_schema=mock_tool_args_schema,
    )


@pytest.fixture
def llama_index_tool(mock_tool_args_schema):
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(
        fn=mock_tool_function,
        name="mock_tool",
        description="mock_description",
        fn_schema=mock_tool_args_schema,
    )


class TestMotleyTool:
    def test_langchain_tool_conversion(self, langchain_tool, mock_input):
        motley_tool = MotleyTool.from_supported_tool(langchain_tool)
        assert isinstance(motley_tool.tool, BaseTool)

        converted_langchain_tool = motley_tool.to_langchain_tool()

        assert type(langchain_tool) is type(converted_langchain_tool)
        assert motley_tool.name == langchain_tool.name
        assert langchain_tool.name == converted_langchain_tool.name
        assert langchain_tool.description == converted_langchain_tool.description
        assert langchain_tool.args_schema == converted_langchain_tool.args_schema

        assert langchain_tool.invoke(mock_input) == converted_langchain_tool.invoke(mock_input)

    def test_llama_index_tool_conversion(self, llama_index_tool, mock_input):
        motley_tool = MotleyTool.from_supported_tool(llama_index_tool)
        assert isinstance(motley_tool.tool, BaseTool)

        converted_llama_index_tool = motley_tool.to_llama_index_tool()

        assert motley_tool.name == llama_index_tool.metadata.name
        assert llama_index_tool.metadata.name == converted_llama_index_tool.metadata.name
        assert (
            llama_index_tool.metadata.description == converted_llama_index_tool.metadata.description
        )
        assert llama_index_tool.metadata.fn_schema == converted_llama_index_tool.metadata.fn_schema

        assert llama_index_tool(mock_input) == converted_llama_index_tool(mock_input)

    def test_motley_tool_self_conversion(self, langchain_tool):
        motley_tool = MotleyTool.from_langchain_tool(langchain_tool)
        motley_tool_2 = MotleyTool.from_supported_tool(motley_tool)

        assert motley_tool.name == motley_tool_2.name
