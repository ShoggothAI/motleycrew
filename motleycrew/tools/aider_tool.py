from motleycrew.common.utils import ensure_module_is_installed

try:
    from aider.coders import Coder
    from aider.models import Model
except ImportError:
    ensure_module_is_installed("aider-chat", "pip install aider-chat")

from langchain.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

from motleycrew.common import Defaults
from motleycrew.tools import MotleyTool


class AiderTool(MotleyTool):

    def __init__(self, model: str = None, **kwargs):
        """ Tool for generating program code

        Args:
            model (str): model name
            **kwargs:
        """

        model = model or Defaults.DEFAULT_LLM_NAME
        llm_model = Model(model=model)
        coder = Coder.create(main_model=llm_model, **kwargs)

        langchain_tool = create_aider_tool(coder)
        super(AiderTool, self).__init__(langchain_tool)


class AiderToolInput(BaseModel):
    """Input for the Aider tool.

    Attributes:
        with_message (str):
    """

    with_message: str = Field(description="instructions for code generation")


def create_aider_tool(coder: Coder):
    """ Create langchain tool from aider coder

    Returns:
        Tool:
    """
    return Tool.from_function(
        func=coder.run,
        name="aider tool",
        description="Tool for generating program code",
        args_schema=AiderToolInput,
    )
