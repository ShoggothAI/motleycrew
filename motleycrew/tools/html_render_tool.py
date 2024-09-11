from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List

from motleycrew.common.utils import ensure_module_is_installed

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
except ImportError:
    webdriver = None
    Service = None

from langchain.tools import Tool
from langchain_core.pydantic_v1 import BaseModel, Field

from motleycrew.tools import MotleyTool
from motleycrew.common import logger


class HTMLRenderer:
    """Helper for rendering HTML code as an image."""

    def __init__(
        self,
        work_dir: str,
        chromedriver_path: str | None = None,
        headless: bool = True,
        window_size: Optional[Tuple[int, int]] = None,
    ):
        ensure_module_is_installed(
            "selenium",
            "see documentation: https://pypi.org/project/selenium/, ChromeDriver is also required",
        )

        self.work_dir = Path(work_dir).resolve()
        self.html_dir = self.work_dir / "html"
        self.images_dir = self.work_dir / "images"

        self.options = webdriver.ChromeOptions()
        if headless:
            self.options.add_argument("--headless")
        self.service = Service(executable_path=chromedriver_path)

        self.window_size = window_size

    def render_image(self, html: str, file_name: str | None = None):
        """Create a PNG image from HTML code.

        Args:
            html (str): HTML code for rendering image.
            file_name (str): File name without extension.
        Returns:
            Path to the rendered image.
        """
        logger.info("Trying to render image from HTML code")
        html_path, image_path = self.build_file_paths(file_name)
        browser = webdriver.Chrome(options=self.options, service=self.service)
        try:
            if self.window_size:
                logger.info("Setting window size to {}".format(self.window_size))
                browser.set_window_size(*self.window_size)

            url = "data:text/html;charset=utf-8,{}".format(html)
            browser.get(url)

            logger.info("Taking screenshot")
            is_created_img = browser.get_screenshot_as_file(image_path)
        finally:
            browser.close()
            browser.quit()

        if not is_created_img:
            logger.error("Failed to render image from HTML code {}".format(image_path))
            return "Failed to render image from HTML code"

        with open(html_path, "w") as f:
            f.write(html)
        logger.info("Saved the HTML code to {}".format(html_path))
        logger.info("Saved the rendered HTML screenshot to {}".format(image_path))

        return image_path

    def build_file_paths(self, file_name: str | None = None) -> Tuple[str, str]:
        """Builds paths to html and image files"""

        # check exists dirs:
        for _dir in (self.work_dir, self.html_dir, self.images_dir):
            if not _dir.exists():
                _dir.mkdir(parents=True)

        file_name = file_name or datetime.now().strftime("%Y_%m_%d__%H_%M")
        html_path = self.html_dir / "{}.html".format(file_name)
        image_path = self.images_dir / "{}.png".format(file_name)

        return str(html_path), str(image_path)


class HTMLRenderTool(MotleyTool):
    """Tool for rendering HTML as image."""

    def __init__(
        self,
        work_dir: str,
        chromedriver_path: str | None = None,
        headless: bool = True,
        window_size: Optional[Tuple[int, int]] = None,
        return_direct: bool = False,
        exceptions_to_reflect: Optional[List[Exception]] = None,
    ):
        """
        Args:
            work_dir: Directory for saving images and HTML files.
            chromedriver_path: Path to the ChromeDriver executable.
        """
        renderer = HTMLRenderer(
            work_dir=work_dir,
            chromedriver_path=chromedriver_path,
            headless=headless,
            window_size=window_size,
        )
        langchain_tool = create_render_tool(renderer)
        super().__init__(
            tool=langchain_tool,
            return_direct=return_direct,
            exceptions_to_reflect=exceptions_to_reflect,
        )


class HTMLRenderToolInput(BaseModel):
    """Input for the HTMLRenderTool."""

    html: str = Field(description="HTML code for rendering")


def create_render_tool(renderer: HTMLRenderer):
    return Tool.from_function(
        func=renderer.render_image,
        name="html_render_tool",
        description="A tool for rendering HTML code as an image",
        args_schema=HTMLRenderToolInput,
    )
