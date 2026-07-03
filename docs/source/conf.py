"""Configuration file for the Sphinx documentation builder."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "photon-mosaic"
copyright = "2026, photon-mosaic"
author = "photon-mosaic"
release = "0.1.0"

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["**.ipynb_checkpoints", "**/includes/**"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
}

html_theme = "pydata_sphinx_theme"
html_title = "photon-mosaic API"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/photon-mosaic/photon-mosaic",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "logo": {"text": project},
}

html_baseurl = "https://api.photon-mosaic.org/"
sitemap_url_scheme = "{link}"

html_static_path = ["_static"]

linkcheck_ignore = [
    "https://api.photon-mosaic.org/*",
]
