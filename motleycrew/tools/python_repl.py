""" Module description """
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_core.pydantic_v1 import BaseModel, Field

from .tool import MotleyTool


class PythonREPLTool(MotleyTool):
    def __init__(self):
        """ Description

        """
        langchain_tool = create_repl_tool()
        super().__init__(langchain_tool)


class REPLToolInput(BaseModel):
    """Input for the REPL tool.

    Attributes:
        command (str):
    """

    command: str = Field(description="code to execute")


# You can create the tool to pass to an agent
def create_repl_tool():
    """ Description

    Returns:
        Tool:
    """
    return Tool.from_function(
        func=PythonREPL().run,
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. "
        "MAKE SURE TO PRINT OUT THE RESULTS YOU CARE ABOUT USING `print(...)`.",
        args_schema=REPLToolInput,
    )
