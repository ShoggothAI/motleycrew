from typing import Optional, List

import replicate

from langchain.agents import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

import motleycrew.common.utils as motley_utils
from motleycrew.tools.image.download_image import download_url_to_directory
from motleycrew.tools.tool import MotleyTool
from motleycrew.common import logger

model_map = {
    "sdxl": "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
    "flux-pro": "black-forest-labs/flux-pro",
    "flux-dev": "black-forest-labs/flux-dev",
    "flux-schnell": "black-forest-labs/flux-schnell",
}

# Each model has a different set of extra parameters, documented at pages like
# https://replicate.com/black-forest-labs/flux-dev/api/schema


def run_model_in_replicate(model_name: str, prompt: str, **kwargs) -> str | List[str]:
    if model_name in model_map:
        model_name = model_map[model_name]
    output = replicate.run(model_name, input={"prompt": prompt, **kwargs})
    return output


def run_model_in_replicate_and_save_images(
    model_name: str, prompt: str, directory_name: Optional[str] = None, **kwargs
) -> List[str]:
    download_urls = run_model_in_replicate(model_name, prompt, **kwargs)
    if isinstance(download_urls, str):
        download_urls = [download_urls]
    if directory_name is None:
        logger.info("Images directory is not provided, returning URLs")
        return download_urls
    out_files = []
    for url in download_urls:
        if motley_utils.is_http_url(url):
            out_files.append(download_url_to_directory(url, directory_name))
    return out_files


class ImageToolInput(BaseModel):
    """Input for the Dall-E tool."""

    description: str = Field(description="image description")


class ReplicateImageGeneratorTool(MotleyTool):
    def __init__(
        self,
        model_name: str,
        images_directory: Optional[str] = None,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
        **kwargs,
    ):
        """
        A tool for generating images from text descriptions using the Replicate API.
        :param model_name: one of "sdxl", "flux-pro", "flux-dev", "flux-schnell", or a full model name supported by replicate
        :param images_directory: the directory to save the images to
        :param kwargs: model-specific parameters, from pages such as https://replicate.com/black-forest-labs/flux-dev/api/schema
        """
        self.model_name = model_name
        self.kwargs = kwargs
        langchain_tool = create_replicate_image_generator_langchain_tool(
            model_name=model_name, images_directory=images_directory, **kwargs
        )

        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


def create_replicate_image_generator_langchain_tool(
    model_name: str, images_directory: Optional[str] = None, **kwargs
):
    def run_replicate_image_generator(description: str):
        return run_model_in_replicate_and_save_images(
            model_name=model_name,
            prompt=description,
            directory_name=images_directory,
            **kwargs,
        )

    return Tool(
        name=f"{model_name}_image_generator",
        func=run_replicate_image_generator,
        description=f"A wrapper around the {model_name} image generation model. Useful for when you need to generate images from a text description. "
        "Input should be an image description.",
        args_schema=ImageToolInput,
    )


if __name__ == "__main__":
    image_dir = os.path.join(os.path.expanduser("~"), "images")
    tool = ReplicateImageGeneratorTool("flux-pro", image_dir, aspect_ratio="3:2")
    output = tool.invoke(
        "A beautiful sunset over the mountains, with a dragon flying into the sunset, photorealistic style."
    )
    print(output)
    print("yay!")
