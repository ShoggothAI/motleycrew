import pytest

from motleycrew.tools import MotleyTool


class ToolMock:
    def invoke(self, input_dict: dict, *args, **kwargs):
        return input_dict


@pytest.fixture
def tools():
    tool1 = MotleyTool(ToolMock())
    tool2 = MotleyTool(ToolMock())
    return [tool1, tool2]


def test_tool_chain(tools):
    tool1, tool2 = tools
    tool_chain = tool1 | tool2
    assert hasattr(tool_chain, "invoke")
    prompt = {"prompt": "test prompt"}
    assert tool_chain.invoke(prompt) == prompt
