# Documentation Generation Guide

This repository is documented directly from docstrings (Google-style) and type hints.

## Recommended toolchain

- **Sphinx** with autodoc + napoleon (recommended for API docs)
- Optional: **sphinx-autobuild** for live reload
- Optional: **sphinx-autodoc-typehints** for richer type rendering

## Install (example)

```bash
python -m pip install -U sphinx sphinx-autodoc-typehints sphinx-autobuild
```

## Minimal Sphinx setup

Create `docs/` (if it does not exist) and add a basic `conf.py` and `index.rst`.

### `docs/conf.py`

```python
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "EPTA Data Combination Pipeline"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autosummary_generate = True
autosummary_imported_members = True

autodoc_typehints = "description"

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Prefer module-level docstrings for summaries
autodoc_docstring_signature = True
```

### `docs/index.rst`

```rst
EPTA Data Combination Pipeline
==============================

.. toctree::
   :maxdepth: 2

   api
```

### `docs/api.rst`

```rst
API Reference
=============

.. autosummary::
   :toctree: _autosummary
   :recursive:

   pleb

.. note::
   The ``:recursive:`` option ensures submodules are discovered automatically.
   For finer control, list submodules explicitly (as in ``docs/api.rst``).
```

## Build docs locally

```bash
sphinx-build -b html docs docs/_build/html
```

For live reload while editing:

```bash
sphinx-autobuild docs docs/_build/html
```

## MkDocs alternative

If you prefer MkDocs, install `mkdocs` and `mkdocstrings[python]` and configure
`mkdocs.yml` with the `mkdocstrings` plugin to pull Google-style docstrings.

Example `mkdocs.yml` snippet:

```yaml
site_name: EPTA Data Combination Pipeline
plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: google
nav:
  - API: api.md
```

Then create an `api.md` page that includes:

```markdown
::: pleb
    options:
      show_source: false
```

## Hosting

The generated HTML in `docs/_build/html` can be served locally with any static
file server (e.g., `python -m http.server`) or uploaded to your preferred
hosting solution.
