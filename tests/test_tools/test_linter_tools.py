import pytest

from motleycrew.tools.code import PostgreSQLLinterTool, PythonLinterTool, SQLLinterTool
from motleycrew.common.exceptions import ModuleNotInstalled, InvalidOutput


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


@pytest.fixture
def sql_linter_tool():
    try:
        tool = SQLLinterTool(dialect="postgres")
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


@pytest.mark.parametrize(
    "dialect,query,should_pass",
    [
        ("postgres", "SELECT * FROM users WHERE age > 21", True),
        ("mysql", "SELECT * FROM `users` WHERE age > 21", True),
        ("postgres", "SELEC * FROM users", False),
        ("postgres", "SELECT * FROM users WHERE age >>= 21", False),
        ("mysql", "SELECT * FROM users WERE age > 21", False),
    ],
)
def test_sql_linter_validation(sql_linter_tool, dialect, query, should_pass):
    if sql_linter_tool is None:
        return

    tool = SQLLinterTool(dialect=dialect)
    
    if should_pass:
        # Valid SQL should return formatted query
        result = tool.invoke({"query": query})
        assert isinstance(result, str)
        assert "SELECT" in result
    else:
        # Invalid SQL should raise InvalidOutput
        with pytest.raises(InvalidOutput):
            tool.invoke({"query": query})


@pytest.mark.parametrize(
    "dialect,query,expected_format",
    [
        (
            "postgres",
            "SELECT id,name,email FROM users WHERE age>21",
            "SELECT id, name, email\nFROM users\nWHERE age > 21"
        ),
        (
            "mysql",
            "SELECT * FROM `users` where active=1",
            "SELECT *\nFROM `users`\nWHERE active = 1"
        ),
    ],
)
def test_sql_linter_formatting(sql_linter_tool, dialect, query, expected_format):
    if sql_linter_tool is None:
        return

    tool = SQLLinterTool(dialect=dialect)
    result = tool.invoke({"query": query})
    
    # Remove any trailing whitespace/newlines for comparison
    result = result.strip()
    expected_format = expected_format.strip()
    
    assert result == expected_format


def test_sql_linter_error_details():
    tool = SQLLinterTool(dialect="postgres")
    query = "SELECT id,name FROM users WHERE age>>21"
    
    with pytest.raises(InvalidOutput) as exc_info:
        tool.invoke({"query": query})
    
    error_msg = str(exc_info.value)
    # Check that error message contains line and position info
    assert "Line" in error_msg
    assert "Position" in error_msg
    # Check that it contains rule code
    assert "L003" in error_msg  # Common SQLFluff rule for comma spacing


def test_dialect_specific_syntax():
    # Snowflake-specific query using table stage (@) and FLATTEN function
    snowflake_query = """
    SELECT 
        value:id::integer as id,
        value:name::string as name
    FROM @my_stage/data.json,
    LATERAL FLATTEN(input => $1)
    """
    
    # Test with Snowflake dialect - should pass
    snowflake_linter = SQLLinterTool(dialect="snowflake")
    result = snowflake_linter.invoke({"query": snowflake_query})
    assert isinstance(result, str)
    assert "SELECT" in result
    
    # Test with PostgreSQL dialect - should fail
    postgres_linter = SQLLinterTool(dialect="postgres")
    with pytest.raises(InvalidOutput) as exc_info:
        postgres_linter.invoke({"query": snowflake_query})
    
    error_msg = str(exc_info.value)
    assert "Failed to parse SQL" in error_msg or "SQL validation failed" in error_msg
