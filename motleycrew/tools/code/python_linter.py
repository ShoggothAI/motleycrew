import os
from typing import Union

from langchain_core.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field

from motleycrew.common.utils import ensure_module_is_installed
from motleycrew.tools import MotleyTool

Linter = None


class PythonLinterTool(MotleyTool):

    def __init__(self):
        """Python code verification tool
        """
        ensure_module_is_installed("aider")

        langchain_tool = create_python_linter_tool()
        super().__init__(langchain_tool)


class PythonLinterInput(BaseModel):
    """Input for the PythonLinterTool.

    Attributes:
        code (str):
        file_name (str):
    """

    code: str = Field(description="Python code for linting")
    file_name: str = Field(description="file name for the code", default="code.py")


def create_python_linter_tool() -> StructuredTool:
    """Create the underlying langchain tool for PythonLinterTool

    Returns:
        Tool:
    """

    def lint(code: str, file_name: str = None) -> Union[str, None]:
        # create temp python file
        temp_file_name = file_name or "code.py"
        _, file_ext = os.path.splitext(temp_file_name)
        if file_ext != ".py":
            raise ValueError("The file extension must be .py")

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

    return StructuredTool.from_function(
        func=lint,
        name="python linter tool",
        description="Tool for validating Python code",
        args_schema=PythonLinterInput,
    )
