import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "pleb - The EPTA Data Combination Pipeline"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autosummary_generate = False
autosummary_imported_members = False

autodoc_typehints = "description"

autodoc_default_options = {
    "undoc-members": False,
    "imported-members": False,
}

exclude_patterns = ["_build", "_autosummary"]
