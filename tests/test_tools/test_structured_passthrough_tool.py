import pytest
from pydantic import BaseModel, ValidationError
from motleycrew.tools.structured_passthrough import StructuredPassthroughTool


class SampleSchema(BaseModel):
    name: str
    age: int


@pytest.fixture
def sample_schema():
    return SampleSchema


@pytest.fixture
def structured_passthrough_tool(sample_schema):
    return StructuredPassthroughTool(schema=sample_schema)


def test_structured_passthrough_tool_initialization(structured_passthrough_tool, sample_schema):
    assert structured_passthrough_tool.schema == sample_schema
    assert structured_passthrough_tool.name == "structured_passthrough_tool"
    assert (
        structured_passthrough_tool.description
        == "A tool that enforces a certain output shape, raising an error if the output is not as expected."
    )


def test_structured_passthrough_tool_run_valid_input(structured_passthrough_tool):
    input_data = {"name": "John Doe", "age": 30}
    result = structured_passthrough_tool.run(**input_data)
    assert result.name == "John Doe"
    assert result.age == 30


def test_structured_passthrough_tool_run_invalid_input(structured_passthrough_tool):
    input_data = {"name": "John Doe", "age": "thirty"}
    with pytest.raises(ValidationError):
        structured_passthrough_tool.run(**input_data)


def test_structured_passthrough_tool_post_process(structured_passthrough_tool):
    def post_process(data):
        data.name = data.name.upper()
        return data

    tool_with_post_process = StructuredPassthroughTool(
        schema=structured_passthrough_tool.schema, post_process=post_process
    )

    input_data = {"name": "John Doe", "age": 30}
    result = tool_with_post_process.run(**input_data)
    assert result.name == "JOHN DOE"
    assert result.age == 30


def test_structured_passthrough_tool_post_process_noop(structured_passthrough_tool):
    def post_process(data):
        return data

    tool_with_post_process = StructuredPassthroughTool(
        schema=structured_passthrough_tool.schema, post_process=post_process
    )

    input_data = {"name": "John Doe", "age": 30}
    result = tool_with_post_process.run(**input_data)
    assert result.name == "John Doe"
    assert result.age == 30
