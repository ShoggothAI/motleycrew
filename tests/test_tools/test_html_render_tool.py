import os
import pytest
import shutil

from motleycrew.tools import HtmlRenderTool
from motleycrew.common.exceptions import ModuleNotInstalled


test_work_dir = "tests/html_render"


@pytest.fixture
def html_render_tool():
    try:
        tool = HtmlRenderTool(test_work_dir)
    except ModuleNotInstalled:
        tool = None

    return tool


@pytest.mark.parametrize(
    "html_code",
    [
        "<html><h1>Test html</h1></html>",
        "<h1>Test html</h1>",
    ],
)
def test_render_tool(html_render_tool, html_code):
    if html_render_tool is None:
        return

    try:
        image_path = html_render_tool.invoke(html_code)
        assert os.path.exists(image_path)
        image_dir, image_file_name = os.path.split(image_path)
        image_name = ".".join(image_file_name.split(".")[:-1])
        html_file_name = "{}.html".format(image_name)
        html_file_path = os.path.join(test_work_dir, "html", html_file_name)
        assert os.path.exists(html_file_path)
    finally:
        if os.path.exists(test_work_dir):
            shutil.rmtree(test_work_dir)
