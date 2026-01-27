"""Configuration file for the RL Money Laundering Sphinx documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


project = "RL Money Laundering"
author = "OUGUENOUNE Amine"
copyright = f"{datetime.now():%Y}, {author}"

# Pull the canonical version from the distributed package if available.
try:  # pragma: no cover - best effort when package is importable locally
    from rl_money_laundering import __version__ as release
except Exception:  # pylint: disable=broad-except
    release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "api/**",
    "Thumbs.db",
    ".DS_Store",
]

html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

html_theme = "furo"
html_title = "RL Money Laundering Documentation"
html_show_sourcelink = True
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#3776ab",
        "color-brand-content": "#24292e",
        "color-admonition-title-background": "#f1f5f9",
        "color-admonition-title": "#0f172a",
    },
    "dark_css_variables": {
        "color-brand-primary": "#8fb3ff",
        "color-brand-content": "#e5e7eb",
    },
}

pygments_style = "default"
pygments_dark_style = "native"

copybutton_prompt_text = r">>> |\$ |\$\$ "
copybutton_prompt_is_regexp = True

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
]

todo_include_todos = True

# Keep headings linkable akin to PyPI docs.
myst_heading_anchors = 3

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Explicitly treat common fenced blocks as directives.
myst_fence_as_directive = [
    "mermaid",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "gymnasium": ("https://gymnasium.farama.org", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torchgeo": ("https://pytorch-geometric.readthedocs.io/en/latest", None),  # ou torchgeo si tu as une doc officielle
    "sklearn": ("https://scikit-learn.org/stable", None),
}


mermaid_version = "10.9.1"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, securityLevel:'loose', theme:'neutral'});"
nitpicky = True
nitpick_ignore = [
    ("py:class", "X"),
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "torch_geometric.data.data.Data"),
    ("py:class", "networkx.classes.digraph.DiGraph"),
    ("py:class", "typing.Dict"),
    ("py:class", "sklearn.preprocessing._label.LabelEncoder"),
    ("py:class", "sklearn.preprocessing._data.StandardScaler"),
]
