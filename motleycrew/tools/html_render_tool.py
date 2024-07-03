from datetime import datetime
from pathlib import Path
from typing import Tuple
from motleycrew.common.utils import ensure_module_is_installed

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
except ImportError:
    ensure_module_is_installed(
        "selenium", "documentation https://pypi.org/project/selenium/ we use Chrome driver"
    )
from langchain.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

from motleycrew.tools import MotleyTool
from motleycrew.common import logger


class HtmlRender:
    def __init__(
        self,
        work_dir: str,
        executable_path: str | None,
    ):
        """Class for rendering html code to image"""

        self.work_dir = Path(work_dir).resolve()
        self.html_dir = self.work_dir / "html"
        self.images_dir = self.work_dir / "images"

        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--headless")
        self.service = Service(executable_path=executable_path)

    def render_image(self, html: str, file_name: str | None = None):
        """Create image with png extension from html code

        Args:
            html (str): html code for rendering image
            file_name (str): file name with not extension
        Returns:
            file path to created image
        """

        html_path, image_path = self.build_save_file_paths(file_name)
        browser = webdriver.Chrome(options=self.options, service=self.service)
        try:
            url = "data:text/html;charset=utf-8,{}".format(html)
            browser.get(url)
            is_created_img = browser.get_screenshot_as_file(image_path)
        finally:
            browser.close()
            browser.quit()

        if not is_created_img:
            logger.error("Failed to create image from html code {}".format(image_path))
            return "Failed to create image from html code"

        with open(html_path, "w") as f:
            f.write(html)
        logger.info("Save html code to {}".format(html_path))
        logger.info("Save image from html code to {}".format(image_path))

        return image_path

    def build_save_file_paths(self, file_name: str | None = None) -> Tuple[str, str]:
        """Builds paths to html and image files

        Args:
            file_name (str): file name with not extension

        Returns:
            tuple[str, str]: html file path and image file path
        """

        # check exists dirs:
        for _dir in (self.work_dir, self.html_dir, self.images_dir):
            if not _dir.exists():
                _dir.mkdir(parents=True)

        file_name = file_name or datetime.now().strftime("%Y_%m_%d__%H_%M")
        html_path = self.html_dir / "{}.html".format(file_name)
        image_path = self.images_dir / "{}.png".format(file_name)

        return str(html_path), str(image_path)


class HtmlRenderTool(MotleyTool):

    def __init__(self, work_dir: str, executable_path: str | None = None):
        """Tool for displaying html as images

        Args:
            work_dir (str): Directory for save images and html files
        """
        renderer = HtmlRender(work_dir, executable_path)
        langchain_tool = create_render_tool(renderer)
        super(HtmlRenderTool, self).__init__(langchain_tool)


class HtmlRenderInput(BaseModel):
    """Input for the HtmlRenderTool.

    Attributes:
        html (str):
    """

    html: str = Field(description="Html code for rendering to an image")


def create_render_tool(renderer: HtmlRender):
    """Create langchain tool from HtmlRender.render_image method

    Returns:
        Tool:
    """
    return Tool.from_function(
        func=renderer.render_image,
        name="html render tool",
        description="A tool for rendering html as an image",
        args_schema=HtmlRenderInput,
    )
