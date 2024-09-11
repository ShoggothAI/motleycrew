from typing import Optional, Type, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.tools import MotleyTool


class LLMTool(MotleyTool):
    """A tool that uses a language model to generate output based on a prompt."""

    def __init__(
        self,
        name: str,
        description: str,
        prompt: str | BasePromptTemplate,
        llm: Optional[BaseLanguageModel] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        """
        Args:
            name: Name of the tool.
            description: Description of the tool.
            prompt: Prompt to use for the tool. Can be a string or a PromptTemplate object.
            llm: Language model to use for the tool.
            input_schema: Input schema for the tool.
                The input variables should match the variables in the prompt.
                If not provided, a schema will be generated based on the input variables
                in the prompt, if any, with string fields.
        """
        langchain_tool = create_llm_langchain_tool(
            name=name,
            description=description,
            prompt=prompt,
            llm=llm,
            input_schema=input_schema,
        )
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


def create_llm_langchain_tool(
    name: str,
    description: str,
    prompt: str | BasePromptTemplate,
    llm: Optional[BaseLanguageModel] = None,
    input_schema: Optional[Type[BaseModel]] = None,
):
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
