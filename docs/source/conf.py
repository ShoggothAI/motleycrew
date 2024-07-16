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

# autodoc_default_options = {
#     "member-order": "bysource",
#     "special-members": "__init__",
# }

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "show-inheritance": True,
    "inherited-members": False,
    "undoc-members": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

nbsphinx_allow_errors = True
nbsphinx_execute = "never"

# Additional configuration for better auto-generated documentation
autosummary_generate = True  # Turn on autosummary

# Create separate .rst files for each module
autosummary_generate_overwrite = False

# Make sure that the generated files are included in the toctree
autosummary_generate_include_files = True
