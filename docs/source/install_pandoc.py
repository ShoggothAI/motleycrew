"""
Script for installing Pandoc in GitHub Actions CI environments.
"""

import os
import shutil
from pypandoc.pandoc_download import download_pandoc

pandoc_location = os.path.abspath("../../.venv/_pandoc")

with open(os.environ["GITHUB_PATH"], "a") as path:
    path.write(str(pandoc_location) + "\n")

if not shutil.which("pandoc"):
    download_pandoc(targetfolder=pandoc_location)
