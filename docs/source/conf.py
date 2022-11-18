# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../er-evaluation"))
sys.path.insert(0, os.path.abspath("../examples"))


# -- Project information -----------------------------------------------------

project = "ER-Evaluation"
copyright = "2022, Olivier Binette"
author = "Olivier Binette"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "m2r2",
    "sphinx_lesson",
    "sphinx.ext.viewcode",
    "sphinx_design",
]
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
nb_execution_mode = "off"

source_suffix = [".rst", ".md", ".ipynb"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    # "logo_link": "index",
    "github_url": "https://github.com/OlivierBinette/er-evaluation",
    # "twitter_url": "https://twitter.com/",
    "collapse_navigation": True,
    # "external_links": [
    #    {"name": "Learn", "url": "https://numpy.org/numpy-tutorials/"}
    #    ],
    # Add light/dark mode and documentation version switcher:
    "navbar_end": ["navbar-icon-links"],
}

html_last_updated_fmt = "%b %d, %Y"
