import os
from typing import Callable, Union

from pglast import parse_sql, prettify
from pglast.parser import ParseError
from langchain.tools import Tool, StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field

try:
    from aider.linter import Linter
except ImportError:
    Linter = None

from motleycrew.tools import MotleyTool
from motleycrew.common.utils import ensure_module_is_installed


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


class PythonLinterTool(MotleyTool):

    def __init__(self):
        """Python code verification tool
        """
        ensure_module_is_installed("aider", "pip install aider-chat")

        def lint(code: str, file_name: str = None) -> Union[str, None]:
            # create temp python file
            temp_file_name = file_name or "code.py"
            _, file_ext = os.path.splitext(temp_file_name)
            if file_ext != ".py":
                raise ValueError("The file extension must be py")

            with open(temp_file_name, 'w') as f:
                f.write(code)

            # lint code
            try:
                linter = Linter()
                return linter.lint(temp_file_name)
            except Exception as e:
                return str(e)
            finally:
                os.remove(temp_file_name)

        langchain_tool = create_python_linter_tool(lint)
        super().__init__(langchain_tool)


class PythonLinterInput(BaseModel):
    """Input for the PgSqlLinterTool.

    Attributes:
        code (str): python code
        file_name (str): name temp python file
    """

    code: str = Field(description="python code for verification")
    file_name: str = Field(description="file name python code", default="code.py")

def create_python_linter_tool(lint_func: Callable) -> StructuredTool:
    """Create langchain tool from lint_func for PythonLinterTool

    Returns:
        Tool:
    """
    return StructuredTool.from_function(
        func=lint_func,
        name="python linter tool",
        description="Tool for checking the health of python code",
        args_schema=PythonLinterInput,
    )
