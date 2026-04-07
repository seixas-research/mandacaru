import os
import sys
from unittest.mock import MagicMock

if os.environ.get('READTHEDOCS') == 'True':
    MOCK_MODULES = ['torch', 'torch.nn', 'ase', 'ase.build', 'mace', 'numpy']
    sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)
    
sys.path.insert(0, os.path.abspath('../../src')) # root

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Carcará'
copyright = '2026, Leandro Seixas Rocha'
author = 'Leandro Seixas Rocha'
release = '2026'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Gera docs a partir das docstrings
    'sphinx.ext.napoleon',     # Suporte para estilo Google/NumPy de docstrings
    'sphinx.ext.viewcode',     # Adiciona links para o código fonte
    'myst_parser',             # Se quiser usar arquivos .md
    'sphinx_rtd_theme'        # O tema do Read the Docs
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = "Carcará"

html_theme = 'furo'                 # 'sphinx_rtd_theme', 'renku', 'shibuya'
html_static_path = ['_static']

html_theme_options = {
    "light_logo": "logo-light.png", # arquivo em docs/source/_static/
    "dark_logo": "logo-dark.png",   # arquivo em docs/source/_static/
}

html_favicon = "_static/favicon.png"

html_css_files = ['custom.css']     

copyright = 'Leandro Seixas Rocha, 2026'
html_show_sphinx = False
html_show_copyright = True
html_show_sourcelink = False