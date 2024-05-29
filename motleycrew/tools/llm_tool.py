""" Module description"""
from typing import Optional, Type

from langchain_core.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import BaseModel, Field, create_model

from motleycrew.tools import MotleyTool
from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm


class LLMTool(MotleyTool):
    def __init__(
        self,
        name: str,
        description: str,
        prompt: str | BasePromptTemplate,
        llm: Optional[BaseLanguageModel] = None,
        input_schema: Optional[Type[BaseModel]] = None,
    ):
        """ Description

        Args:
            name (str):
            description (str):
            prompt (:obj:`str`, :obj:`BasePromptTemplate`):
            llm (:obj:`BaseLanguageModel`, optional):
            input_schema (:obj:`Type[BaseModel]`, optional):
        """
        langchain_tool = create_llm_langchain_tool(
            name=name,
            description=description,
            prompt=prompt,
            llm=llm,
            input_schema=input_schema,
        )
        super().__init__(langchain_tool)


def create_llm_langchain_tool(
    name: str,
    description: str,
    prompt: str | BasePromptTemplate,
    llm: Optional[BaseLanguageModel] = None,
    input_schema: Optional[Type[BaseModel]] = None,
):
    """ Description

    Args:
        name (str):
        description (str):
        prompt (:obj:`str`, :obj:`BasePromptTemplate`):
        llm (:obj:`BaseLanguageModel`, optional):
        input_schema (:obj:`Type[BaseModel]`, optional):

    Returns:

    """
    if llm is None:
        llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

    if not isinstance(prompt, BasePromptTemplate):
        prompt = PromptTemplate.from_template(prompt)

    if input_schema is None:
        fields = {
            var: (str, Field(description=f"Input {var} for the tool."))
            for var in prompt.input_variables
        }

        # Create the LLMToolInput class dynamically
        input_schema = create_model("LLMToolInput", **fields)

    def call_llm(**kwargs) -> str:
        chain = prompt | llm
        return chain.invoke(kwargs)

    return StructuredTool.from_function(
        func=call_llm,
        name=name,
        description=description,
        args_schema=input_schema,
    )
