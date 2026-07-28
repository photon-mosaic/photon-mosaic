"""Configuration file for the Sphinx documentation builder."""

import sys
from pathlib import Path

import setuptools_scm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "photon-mosaic"
copyright = "2026, photon-mosaic core developers"
author = "photon-mosaic"

try:
    release = setuptools_scm.get_version(root="../..", relative_to=__file__)
    release = release.split(".dev")[0]
except LookupError:
    # if git is not initialised, still allow local build
    # with a dummy version
    release = "0.0.0"

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
# Configure the myst parser to enable cool markdown features
# See https://sphinx-design.readthedocs.io
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
# Automatically add anchors to markdown headings
myst_heading_anchors = 3


templates_path = ["_templates"]
exclude_patterns = ["**.ipynb_checkpoints", "**/includes/**"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]

html_favicon = "_static/logo.png"
html_logo = "_static/logo.png"

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
}

html_theme = "pydata_sphinx_theme"
html_title = "photon-mosaic API"

# Customize the theme
html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/photon-mosaic/photon-mosaic",  # required
            # Icon class (if "type": "fontawesome"),
            # or path to local image (if "type": "local")
            "icon": "fa-brands fa-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
            "use_edit_page_button": False,  # Ensure the edit button doesn't interfere
            "navigation_with_keys": False,  # Disable keyboard navigation between sections
            "collapse_navigation": False,  # Ensure full page loads rather than AJAX content swap
        },
        {
            # Label for this link
            "name": "Zulip (chat)",
            # URL where the link will redirect
            "url": "https://neuroinformatics.zulipchat.com/#narrow/channel/500681-photon-mosaic",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-solid fa-comments",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": f"{project} v{release}",
    },
    "footer_start": ["footer_start"],
    "footer_end": ["footer_end"],
}

html_baseurl = "https://api.photon-mosaic.org/"
sitemap_url_scheme = "{link}"

html_static_path = ["_static"]

linkcheck_ignore = [
    "https://api.photon-mosaic.org/*",
    "https://neuroinformatics.zulipchat.com/#narrow/channel/500681-photon-mosaic",
]
