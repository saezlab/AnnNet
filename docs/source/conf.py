# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
sys.path.insert(0, os.path.abspath('../../src/graphglue'))
sys.path.insert(0, os.path.abspath('../../src/graphglue/demo'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GraphGlue'
copyright = '2025, Bottazzi Daniele'
author = 'Bottazzi Daniele'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
]

autosectionlabel_prefix_document = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

#bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'plain'
templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "sidebarwidth": '25%',
}

templates_path = ['_templates']

html_static_path = ['_static']

def skip_member(app, what, name, obj, skip, options):
    if name == "CustomArgumentParser":
        return True
    return skip

def setup(app):
    app.connect("autodoc-skip-member", skip_member)
