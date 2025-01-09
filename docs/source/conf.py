# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import jax.typing as jpt
import numpy.typing as npt

project = "msmJAX"
copyright = "2025, Florian Buchner, Johannes Schörghuber, Nico Unglert, Jesús Carrete, Georg K. H. Madsen"
author = "Florian Buchner, Johannes Schörghuber, Nico Unglert, Jesús Carrete, Georg K. H. Madsen"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
}


def custom_typehints_formatter(annotation, config):
    """Small custom formatter taking care of npt.ArrayLike"""
    # TODO: Remove this quick hack as soon as a better solution is
    # available for aliases (that works with typehints).
    if annotation == npt.ArrayLike:
        return ":py:class:`numpy.typing.ArrayLike`"
    if annotation == jpt.ArrayLike:
        return ":py:class:`jax.typing.ArrayLike`"
    return None


typehints_formatter = custom_typehints_formatter


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "pydata_sphinx_theme"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {"collapse_navigation": False, "sticky_navigation": True}
