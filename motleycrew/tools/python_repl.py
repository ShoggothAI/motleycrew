from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional, List

from .tool import MotleyTool


class PythonREPLTool(MotleyTool):
    """Python REPL tool. Use this to execute python commands.

    Note that the tool's output is the content printed to stdout by the executed code.
    Because of this, any data you want to be in the output should be printed using `print(...)`.
    """

    def __init__(
        self,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        langchain_tool = create_repl_tool()
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


class REPLToolInput(BaseModel):
    """Input for the REPL tool."""

    command: str = Field(description="code to execute")


def create_repl_tool():
    return Tool.from_function(
        func=PythonREPL().run,
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. "
        "MAKE SURE TO PRINT OUT THE RESULTS YOU CARE ABOUT USING `print(...)`.",
        args_schema=REPLToolInput,
    )
