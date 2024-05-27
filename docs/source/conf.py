# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.append(os.path.abspath("../.."))


project = "motleycrew"
copyright = "2024, motleycrew"
author = "motleycrew"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "nbsphinx",
    "nbsphinx_link",
]

templates_path = ["_templates", "_templates/autosummary"]
exclude_patterns = []
autosummary_generate = True
autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

nbsphinx_allow_errors = True
nbsphinx_execute = "never"

html_theme_options = {
    "display_github": True,
    "github_url": "https://github.com/ShoggothAI/motleycrew",
}
