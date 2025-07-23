# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Carcará'
copyright = '2025, Leandro Seixas'
author = 'Leandro Seixas'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html

html_theme = 'alabaster'
html_static_path = ['_static']
html_title = "Carcará Documentation"
html_logo = '_static/images/logo.png'
html_favicon = '_static/images/favicon.ico'


# html_theme_options = {
#     # path is relative to _static/
#     'logo'       : 'images/logo.png',
#     'logo_name'  : True,               # show project name under logo
#     'description': 'My project subtitle'
# }

# HTML sidebar options
# html_sidebars = {
#     '**': [
#         'about.html',
#         'navigation.html',
#         'relations.html',
#         'searchbox.html',
#         'donate.html',
#     ]
# }