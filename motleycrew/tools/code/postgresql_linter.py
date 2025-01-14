from typing import List, Optional

from langchain_core.tools import Tool
from pydantic import BaseModel, Field

try:
    from pglast import parse_sql, prettify
    from pglast.parser import ParseError
except ImportError:
    parse_sql = None
    prettify = None
    ParseError = None

from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.tools import MotleyTool


class PostgreSQLLinterTool(MotleyTool):
    """PostgreSQL code verification tool."""

    def __init__(
        self,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        ensure_module_is_installed("pglast")

        langchain_tool = create_pgsql_linter_tool()
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


class PostgreSQLLinterInput(BaseModel):
    """Input for the PostgreSQLLinterTool."""

    query: str = Field(description="SQL code for validation")


def create_pgsql_linter_tool() -> Tool:
    def parse_func(query: str) -> str:
        try:
            parse_sql(query)
            return prettify(query)
        except ParseError as e:
            return str(e)

    return Tool.from_function(
        func=parse_func,
        name="postgresql_linter",
        description="Tool for validating PostgreSQL code",
        args_schema=PostgreSQLLinterInput,
    )
