# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

from packaging.version import Version
import resins

PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(PATH, '_ext'))

from generate_instrument_docs import main
main()
del main

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'resins'
copyright = '2026, Rastislav Turanyi, Adam J Jackson'
author = 'Rastislav Turanyi, Adam J Jackson'
release = resins.__version__
version = Version(resins.__version__).base_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # 'sphinx.ext.napoleon',
    'numpydoc',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.autosummary',
    'inline_reference',
    'sphinx_parsed_codeblock',
]

templates_path = ['_templates']
exclude_patterns = []

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}

master_doc = 'index' # Otherwise readthedocs searches for contents.rst

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# -- Options for napoleon -------------------------------------------------

# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_attr_annotations = True

numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "obj"
