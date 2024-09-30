import os

import pytest

from motleycrew.tools.html_render_tool import HTMLRenderTool


@pytest.mark.fat
@pytest.mark.parametrize(
    "html_code",
    [
        "<html><h1>Test html</h1></html>",
        "<h1>Test html</h1>",
    ],
)
def test_render_tool(tmpdir, html_code):
    html_render_tool = HTMLRenderTool(work_dir=str(tmpdir), window_size=(800, 600), headless=False)

    image_path = html_render_tool.invoke(html_code)
    assert os.path.exists(image_path)
    image_dir, image_file_name = os.path.split(image_path)
    image_name = ".".join(image_file_name.split(".")[:-1])
    html_file_name = "{}.html".format(image_name)
    html_file_path = os.path.join(tmpdir, "html", html_file_name)
    assert os.path.exists(html_file_path)
