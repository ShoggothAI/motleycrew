import pytest

from motleycrew.tools.code import PostgreSQLLinterTool, PythonLinterTool
from motleycrew.common.exceptions import ModuleNotInstalled


@pytest.fixture
def pgsql_linter_tool():
    tool = PostgreSQLLinterTool()
    return tool


@pytest.fixture
def python_linter_tool():
    try:
        tool = PythonLinterTool()
    except ModuleNotInstalled:
        tool = None
    return tool


@pytest.mark.parametrize(
    "query, expected",
    [
        ("select a from table_name", "SELECT a\nFROM table_name"),
        ("select a from table_name where a = 1", "SELECT a\nFROM table_name\nWHERE a = 1"),
        ("selec a from table_name where a = 1", 'syntax error at or near "selec", at index 0'),
    ],
)
def test_pgsql_tool(pgsql_linter_tool, query, expected):
    parse_result = pgsql_linter_tool.invoke({"query": query})
    assert expected == parse_result


@pytest.mark.parametrize(
    "code, file_name, valid_code, raises",
    [
        ("def plus(a, b):\n\treturn a + b", None, True, False),
        ("def plus(a):\n\treturn a + b", "test_code.py", False, False),
        ("def plus(a, b):\nreturn a + b", "test_code.py", False, False),
        ("def plus(a, b):\n\treturn a + b", "code.js", True, True),
    ],
)
def test_python_tool(python_linter_tool, code, file_name, valid_code, raises):
    if python_linter_tool is None:
        return

    params = {"code": code}
    if file_name:
        params["file_name"] = file_name

    if raises:
        with pytest.raises(ValueError):
            python_linter_tool.invoke(params)
    else:
        linter_result = python_linter_tool.invoke(params)
        if valid_code:
            assert linter_result is None
        else:
            assert isinstance(linter_result, str)
