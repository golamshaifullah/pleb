GUI (Experimental)
==================

Pleb includes a minimal Streamlit GUI for editing configs and launching runs.
This GUI is **experimental** and may change without notice.

What it provides
----------------

- **Ingest mapping editor** (load/edit/save JSON or TOML)
- **Settings editor** with a schema-driven form and raw text editor
- **Workflow editor** (load/edit/save JSON or TOML)
- **Runner** tab for pipeline/param-scan/qc-report/ingest/workflow

Install
-------

Install the optional GUI dependencies:

.. code-block:: bash

   pip install -e ".[gui]"

Run
---

From the repo root:

.. code-block:: bash

   python -m streamlit run scripts/pleb_gui.py

Notes
-----

- The GUI uses ``configs/settings/gui_schema.json`` for its settings form.
- Validation is best-effort and relies on optional ``jsonschema``.
- For production or automated runs, prefer the CLI or workflow files.
