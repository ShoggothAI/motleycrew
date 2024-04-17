from typing import Optional

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field

from motleycrew.tool import MotleyTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm


class LLMTool(MotleyTool):
    def __init__(
        self,
        name: str,
        description: str,
        prompt: str,
        llm: Optional[BaseLanguageModel] = None,
    ):
        langchain_tool = create_llm_langchain_tool(name, description, prompt, llm)
        super().__init__(langchain_tool)


def create_llm_langchain_tool(
    name: str,
    description: str,
    prompt: str,
    llm: Optional[BaseLanguageModel] = None,
    input_description: Optional[str] = "Input for the tool.",
):
    if llm is None:
        llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

    class LLMToolInput(BaseModel):
        """Input for the tool."""

        input: str = Field(description=input_description)

    p = PromptTemplate.from_template(prompt)
    assert "input" in p.input_variables, "Prompt must contain an `input` variable"

    def call_llm(input: str) -> str:
        chain = p | llm
        return chain.invoke({"input": input})

    return Tool.from_function(
        func=call_llm,
        name=name,
        description=description,
        args_schema=LLMToolInput,
    )
