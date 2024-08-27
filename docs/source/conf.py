import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Chemprop"
copyright = "2024, Chemprop developers"
author = "Chemprop developers"
release = "2.0.4"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx.ext.doctest",
    "sphinxarg.ext",
    "nbsphinx_link",
]

nbsphinx_execute = "never"
templates_path = ["_templates"]
exclude_patterns = []
autodoc_typehints = "description"

# -- AutoAPI configuration ---------------------------------------------------
nbsphinx_allow_errors = True
autoapi_dirs = ["../.."]
autoapi_ignore = ["*/tests/*", "*/cli/*"]
autoapi_file_patterns = ["*.py"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]
autoapi_keep_files = True

# -- bibtex configuration ---------------------------------------------------

bibtex_bibfiles = ["refs.bib"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
