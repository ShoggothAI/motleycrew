from typing import Optional, List

from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field

import motleycrew.common.utils as motley_utils
from motleycrew.common import LLMFramework
from motleycrew.common import logger
from motleycrew.common.llms import init_llm
from motleycrew.tools.image.download_image import download_url_to_directory
from motleycrew.tools.tool import MotleyTool

DEFAULT_REFINE_PROMPT = """Generate a detailed DALL-E prompt to generate an image
based on the following description:
```{text}```
Your output MUST NOT exceed 3500 characters"""

DEFAULT_DALL_E_PROMPT = """{text}
Note: Do not include any text in the images.
"""


class DallEImageGeneratorTool(MotleyTool):
    """A tool for generating images using the OpenAI DALL-E API.

    See the OpenAI API reference for more information:
    https://platform.openai.com/docs/guides/images/usage
    """

    def __init__(
        self,
        images_directory: Optional[str] = None,
        refine_prompt_with_llm: bool = True,
        dall_e_prompt_template: str | PromptTemplate = DEFAULT_DALL_E_PROMPT,
        refine_prompt_template: str | PromptTemplate = DEFAULT_REFINE_PROMPT,
        model: str = "dall-e-3",
        quality: str = "standard",
        size: str = "1024x1024",
        style: Optional[str] = None,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        """
        Args:
            images_directory: Directory to save the generated images.
            refine_prompt_with_llm: Whether to refine the prompt using a language model.
            model: DALL-E model to use.
            quality: Image quality. Can be "standard" or "hd".
            size: Image size.
            style: Style to use for the model.
        """
        langchain_tool = create_dalle_image_generator_langchain_tool(
            images_directory=images_directory,
            refine_prompt_with_llm=refine_prompt_with_llm,
            dall_e_prompt_template=dall_e_prompt_template,
            refine_prompt_template=refine_prompt_template,
            model=model,
            quality=quality,
            size=size,
            style=style,
        )
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


class DallEToolInput(BaseModel):
    """Input for the Dall-E tool."""

    description: str = Field(description="image description")


def run_dalle_and_save_images(
    description: str,
    images_directory: Optional[str] = None,
    refine_prompt_with_llm: bool = True,
    dall_e_prompt_template: str | PromptTemplate = DEFAULT_DALL_E_PROMPT,
    refine_prompt_template: str | PromptTemplate = DEFAULT_REFINE_PROMPT,
    model: str = "dall-e-3",
    quality: str = "standard",
    size: str = "1024x1024",
    style: Optional[str] = None,
    file_name_length: int = 8,
) -> Optional[list[str]]:
    dall_e_prompt = PromptTemplate.from_template(dall_e_prompt_template)

    if refine_prompt_with_llm:
        prompt = PromptTemplate.from_template(refine_prompt_template)
        llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)
        dall_e_prompt = prompt | llm | (lambda x: {"text": x.content}) | dall_e_prompt

    prompt_value = dall_e_prompt.invoke({"text": description})

    dalle_api = DallEAPIWrapper(
        model=model,
        quality=quality,
        size=size,
        model_kwargs={"style": style} if (model == "dall-e-3" and style) else {},
    )

    dalle_result = dalle_api.run(prompt_value.text)
    logger.info("Dall-E API output: %s", dalle_result)

    urls = dalle_result.split(dalle_api.separator)
    if not len(urls) or not motley_utils.is_http_url(urls[0]):
        logger.error("Dall-E API did not return a valid url: %s", dalle_result)
        return

    if images_directory:
        return [download_url_to_directory(url, images_directory, file_name_length) for url in urls]
    else:
        logger.info("Images directory is not provided, returning URLs")
        return urls


def create_dalle_image_generator_langchain_tool(
    images_directory: Optional[str] = None,
    refine_prompt_with_llm: bool = True,
    dall_e_prompt_template: str | PromptTemplate = DEFAULT_DALL_E_PROMPT,
    refine_prompt_template: str | PromptTemplate = DEFAULT_REFINE_PROMPT,
    model: str = "dall-e-3",
    quality: str = "standard",
    size: str = "1024x1024",
    style: Optional[str] = None,
):
    def run_dalle_and_save_images_partial(description: str):
        return run_dalle_and_save_images(
            description=description,
            images_directory=images_directory,
            refine_prompt_with_llm=refine_prompt_with_llm,
            dall_e_prompt_template=dall_e_prompt_template,
            refine_prompt_template=refine_prompt_template,
            model=model,
            quality=quality,
            size=size,
            style=style,
        )

    return Tool(
        name="dalle_image_generator",
        func=run_dalle_and_save_images_partial,
        description="A wrapper around OpenAI DALL-E API. Useful for when you need to generate images from a text description. "
        "Input should be an image description.",
        args_schema=DallEToolInput,
    )
