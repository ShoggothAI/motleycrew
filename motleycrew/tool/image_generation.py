from langchain.agents import Tool
from langchain.agents import load_tools
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

from .tool import MotleyTool


class DallEImageGeneratorTool(MotleyTool):
    def __init__(self):
        langchain_tool = create_dalle_image_generator_langchain_tool()
        super().__init__(langchain_tool)


class DallEToolInput(BaseModel):
    """Input for the Dall-E tool."""

    query: str = Field(description="image generation query")


def create_dalle_image_generator_langchain_tool():
    return Tool(
        name="Dall-E-Image-Generator",
        func=DallEAPIWrapper().run,
        description="A wrapper around OpenAI DALL-E API. Useful for when you need to generate images from a text description. "
                    "Input should be an image description.",
        args_schema=DallEToolInput,
    )
