from typing import Callable

from pglast import parse_sql, prettify
from pglast.parser import ParseError
from langchain.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

from motleycrew.tools import MotleyTool


class PgSqlLinterTool(MotleyTool):

    def __init__(self):
        """Pgsql code verification tool
        """
        def parse_func(query: str) -> str:
            try:
                parse_sql(query)
                return prettify(query)
            except ParseError as e:
                return str(e)

        langchain_tool = create_pgsql_linter_tool(parse_func)
        super().__init__(langchain_tool)


class PgSqlLinterInput(BaseModel):
    """Input for the PgSqlLinterTool.

    Attributes:
        query (str):
    """

    query: str = Field(description="sql code for verification")

def create_pgsql_linter_tool(parse_func: Callable) -> Tool:
    """Create langchain tool from parse_func for PgSqlLinterTool

    Returns:
        Tool:
    """
    return Tool.from_function(
        func=parse_func,
        name="pgsql linter tool",
        description="Tool for checking the health of the sql code of the postgresql database",
        args_schema=PgSqlLinterInput,
    )
