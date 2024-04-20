from typing import Optional

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
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
        prompt: str | BasePromptTemplate,
        llm: Optional[BaseLanguageModel] = None,
    ):
        langchain_tool = create_llm_langchain_tool(name, description, prompt, llm)
        super().__init__(langchain_tool)


def create_llm_langchain_tool(
    name: str,
    description: str,
    prompt: str | BasePromptTemplate,
    llm: Optional[BaseLanguageModel] = None,
    input_description: Optional[str] = "Input for the tool.",
):
    if llm is None:
        llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

    if not isinstance(prompt, BasePromptTemplate):
        prompt = PromptTemplate.from_template(prompt)

    assert (
        len(prompt.input_variables) == 1
    ), "Prompt must contain exactly one input variable"
    input_var = prompt.input_variables[0]

    class LLMToolInput(BaseModel):
        """Input for the tool."""

        # TODO: how hard is it to get that name from prompt.input_variables?
        input: str = Field(description=input_description)

    def call_llm(input: str) -> str:
        chain = prompt | llm
        return chain.invoke({input_var: input})

    return Tool.from_function(
        func=call_llm,
        name=name,
        description=description,
        args_schema=LLMToolInput,
    )
