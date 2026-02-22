# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, root_dir)
import tracker

project = 'tracker'
copyright = '2026, Jack Featherstone'
author = 'Jack Featherstone'
release = tracker.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'numpydoc',
              'sphinx.ext.autosummary',
              'sphinx_subfigure',
              'sphinx_collapse',
              'sphinx.ext.githubpages',
              'autoapi.extension']

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": True,
}

templates_path = ['_templates']
exclude_patterns = []

autoclass_content = 'both'

#autosummary_generate = True

numpydoc_class_members_toctree = False

autoapi_dirs = [f'{root_dir}/{project}']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_css_files = ['overrides.css']
