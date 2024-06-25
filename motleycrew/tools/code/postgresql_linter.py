from langchain_core.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

try:
    from pglast import parse_sql, prettify
    from pglast.parser import ParseError
except ImportError:
    parse_sql = None
    prettify = None
    ParseError = None

from motleycrew.tools import MotleyTool
from motleycrew.common.utils import ensure_module_is_installed


class PostgreSQLLinterTool(MotleyTool):

    def __init__(self):
        """PostgreSQL code verification tool
        """
        ensure_module_is_installed("pglast")

        langchain_tool = create_pgsql_linter_tool()
        super().__init__(langchain_tool)


class PostgreSQLLinterInput(BaseModel):
    """Input for the PostgreSQLLinterTool.

    Attributes:
        query (str):
    """

    query: str = Field(description="SQL code for validation")


def create_pgsql_linter_tool() -> Tool:
    """Create the underlying langchain tool for PostgreSQLLinterTool

    Returns:
        Tool:
    """
    def parse_func(query: str) -> str:
        try:
            parse_sql(query)
            return prettify(query)
        except ParseError as e:
            return str(e)

    return Tool.from_function(
        func=parse_func,
        name="PostgreSQL linter tool",
        description="Tool for validating PostgreSQL code",
        args_schema=PostgreSQLLinterInput,
    )
