"""Sphinx configuration."""

import datetime
import doctest
import os
import sys

import sphinx_rtd_theme

import chemicalx

sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "jupyter_sphinx",
    "sphinx_autodoc_typehints",
    "sphinx_automodapi.automodapi",
]

autodoc_mock_imports = ["torchdrug", "torch", "torch-scatter"]

source_suffix = ".rst"
master_doc = "index"

author = "Benedek Rozemberczki, Charles Tapley Hoyt, and Benjamin Gyori"
project = "chemicalx"
copyright = "{}, {}".format(datetime.datetime.now().year, author)

# version = chemicalx.__version__
# release = chemicalx.__version__

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
intersphinx_mapping = {"python": ("https://docs.python.org/", None)}

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": True,
}

html_logo = "_static/img/chemicalx.jpg"
html_static_path = ["_static"]
# html_context = {"css_files": ["_static/css/custom.css"]}
rst_context = {"chemicalx": chemicalx}

add_module_names = False


def setup(app):
    """Set up the app."""

    def skip(app, what, name, obj, skip, options):
        members = [
            "__init__",
            "__repr__",
            "__weakref__",
            "__dict__",
            "__module__",
        ]
        return True if name in members else skip

    app.connect("autodoc-skip-member", skip)
