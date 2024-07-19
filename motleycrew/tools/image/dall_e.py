import mimetypes
import os
from typing import Optional

import requests
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field

import motleycrew.common.utils as motley_utils
from motleycrew.common import LLMFramework
from motleycrew.common import logger
from motleycrew.common.llms import init_llm
from motleycrew.tools.tool import MotleyTool


def download_image(url: str, file_path: str) -> Optional[str]:
    response = requests.get(url, stream=True)
    if response.status_code == requests.codes.ok:
        try:
            content_type = response.headers.get("content-type")
            extension = mimetypes.guess_extension(content_type)
        except Exception as e:
            logger.error("Failed to guess content type: %s", e)
            extension = None

        if not extension:
            extension = ".png"

        file_path_with_extension = file_path + extension
        logger.info("Downloading image %s to %s", url, file_path_with_extension)

        with open(file_path_with_extension, "wb") as f:
            for chunk in response:
                f.write(chunk)

        return file_path_with_extension
    else:
        logger.error("Failed to download image. Status code: %s", response.status_code)


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
        super().__init__(langchain_tool)


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
        os.makedirs(images_directory, exist_ok=True)
        file_paths = []
        for url in urls:
            file_name = motley_utils.generate_hex_hash(url, length=file_name_length)
            file_path = os.path.join(images_directory, file_name)

            file_path_with_extension = download_image(url=url, file_path=file_path).replace(
                os.sep, "/"
            )
            file_paths.append(file_path_with_extension)
        return file_paths
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
