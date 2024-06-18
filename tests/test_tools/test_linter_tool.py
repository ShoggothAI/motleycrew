import pytest

from motleycrew.tools import PgSqlLinterTool

@pytest.fixture
def pgsql_linter_tool():
    tool = PgSqlLinterTool()
    return tool

@pytest.mark.parametrize(
    "query, expected",
    [
        ("select a from table_name", "SELECT a\nFROM table_name"),
        ("select a from table_name where a = 1", "SELECT a\nFROM table_name\nWHERE a = 1"),
        ("selec a from table_name where a = 1", 'syntax error at or near "selec", at index 0')

    ])
def test_pgsql_tool(pgsql_linter_tool, query, expected):
    parse_result = pgsql_linter_tool.invoke({"query": query})
    assert expected == parse_result
