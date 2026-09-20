"""Sphinx configuration shared by local and Read the Docs builds."""

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "Mandacaru"
author = "Leandro Seixas Rocha"
copyright = "2026, Leandro Seixas Rocha"
release = runpy.run_path(str(ROOT / "src/mandacaru/version.py"))["__version__"]
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
templates_path = ["_templates"]
exclude_patterns = []
# Avoid duplicate attribute targets when autodoc also documents dataclass fields.
napoleon_use_ivar = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"

html_title = project
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# As in Poraquê, Furo selects colors for light, dark and automatic mode.
# It sets data-theme on <body>, not on <html>.
html_theme_options = {"light_logo": "icon.png", "dark_logo": "icon.png"}
html_favicon = "_static/favicon.png"
html_show_sphinx = False
html_show_copyright = True
html_show_sourcelink = False
