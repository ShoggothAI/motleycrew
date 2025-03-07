from typing import List, Optional

from langchain_core.tools import Tool
from pydantic import BaseModel, Field

try:
    import sqlfluff
except ImportError:
    sqlfluff = None

from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.tools import MotleyTool
from motleycrew.common.exceptions import InvalidOutput


class SQLLinterTool(MotleyTool):
    """SQL code verification tool using SQLFluff."""

    def __init__(
        self,
        dialect: str = "ansi",
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        ensure_module_is_installed("sqlfluff")

        self.dialect = dialect
        langchain_tool = create_sql_linter_tool(self.dialect)
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


class SQLLinterInput(BaseModel):
    """Input for the SQLLinterTool."""

    query: str = Field(description="SQL code for validation")


def create_sql_linter_tool(dialect: str) -> Tool:
    def lint_func(query: str) -> str:
        try:
            # Try to format - if it works, SQL is valid
            formatted = sqlfluff.fix(query, dialect=dialect)
            if not formatted:
                raise InvalidOutput("SQL syntax error: Failed to format query")
            return formatted

        except Exception as e:
            raise InvalidOutput(f"SQL syntax error: {str(e)}")

    return Tool.from_function(
        func=lint_func,
        name="sql_linter",
        description=f"Tool for validating {dialect} SQL code",
        args_schema=SQLLinterInput,
    ) 