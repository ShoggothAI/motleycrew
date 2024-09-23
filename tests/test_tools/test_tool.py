import pytest
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from crewai_tools import Tool as CrewAiTool
except ImportError:
    CrewAiTool = None

from motleycrew.tools import DirectOutput, MotleyTool


@pytest.fixture
def mock_tool_args_schema():
    class MockToolInput(BaseModel):
        mock_input: str = Field(description="mock")

    return MockToolInput


def mock_tool_function(mock_input: str):
    if mock_input == "raise":
        raise ValueError("test")

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


@pytest.fixture
def crewai_tool(mock_tool_args_schema):
    if CrewAiTool is None:
        return None
    return CrewAiTool(
        name="mock_tool",
        description="mock_description",
        func=mock_tool_function,
        args_schema=mock_tool_args_schema,
    )


@pytest.fixture
def motley_agent(langchain_tool):
    from motleycrew.agents.langchain import ReActToolCallingMotleyAgent

    return ReActToolCallingMotleyAgent(
        name="mock_agent",
        description="mock_description",
        tools=[langchain_tool],
    )


class TestMotleyTool:
    def test_tool_return_direct(self, langchain_tool, mock_input):
        motley_tool = MotleyTool.from_supported_tool(langchain_tool, return_direct=True)

        with pytest.raises(DirectOutput) as e:
            motley_tool.invoke(mock_input)

        assert e.value.output == mock_input.get("mock_input")

    def test_tool_reflect_exception(self, langchain_tool, mock_input):
        motley_tool = MotleyTool.from_supported_tool(
            langchain_tool, exceptions_to_reflect=[ValueError]
        )
        output = motley_tool.invoke({"mock_input": "raise"})
        assert output == "ValueError: test"

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

    def test_motley_agent_conversion(self, motley_agent, mock_input):
        motley_tool = MotleyTool.from_supported_tool(motley_agent)

        assert isinstance(motley_tool.tool, BaseTool)
        assert motley_tool.name == motley_agent.name
        assert motley_tool.description == motley_agent.description

    def test_autogen_tool_conversion(self, langchain_tool, mock_input):
        motley_tool = MotleyTool.from_supported_tool(langchain_tool)
        assert isinstance(motley_tool.tool, BaseTool)

        converted_autogen_tool = motley_tool.to_autogen_tool()
        assert converted_autogen_tool(mock_input.get("mock_input")) == motley_tool.invoke(
            mock_input
        )

    def test_crewai_tool_conversion(self, crewai_tool, mock_input):
        if crewai_tool is None:
            return

        motley_tool = MotleyTool.from_supported_tool(crewai_tool)
        assert isinstance(motley_tool.tool, BaseTool)

        converted_crewai_tool = motley_tool.to_crewai_tool()
        assert isinstance(converted_crewai_tool, CrewAiTool)
        assert motley_tool.name == converted_crewai_tool.name
        assert crewai_tool.name == converted_crewai_tool.name
        assert crewai_tool.description == converted_crewai_tool.description
        assert crewai_tool.args_schema == converted_crewai_tool.args_schema
        assert crewai_tool.run(**mock_input) == converted_crewai_tool.run(**mock_input)
