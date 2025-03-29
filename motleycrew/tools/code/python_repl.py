from typing import List, Optional

from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field

from motleycrew.tools import MotleyTool


class MissingPrintStatementError(Exception):
    """Exception raised when a print statement is missing from the command."""

    def __init__(self, command: str):
        self.command = command
        super().__init__(
            f"Command must contain at least one print statement. Remember to print the results you want to see using print(...)."
        )


class PythonREPLTool(MotleyTool):
    """Python REPL tool. Use this to execute python commands.

    Note that the tool's output is the content printed to stdout by the executed code.
    Because of this, any data you want to be in the output should be printed using `print(...)`.
    """

    def __init__(
        self, return_direct: bool = False, exceptions_to_reflect: Optional[List[Exception]] = None
    ):
        exceptions_to_reflect = (exceptions_to_reflect or []) + [MissingPrintStatementError]
        super().__init__(
            name="python_repl",
            description="A Python shell. Use this to execute python commands. Input should be a valid python command. "
            "MAKE SURE TO PRINT OUT THE RESULTS YOU CARE ABOUT USING `print(...)`.",
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
            args_schema=REPLToolInput,
        )

    def run(self, command: str) -> str:
        self.validate_input(command)
        return PythonREPL().run(command)

    def validate_input(self, command: str):
        if "print(" not in command:
            raise MissingPrintStatementError(command)


class REPLToolInput(BaseModel):
    """Input for the REPL tool."""

    command: str = Field(description="code to execute")
