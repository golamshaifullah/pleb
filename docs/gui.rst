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
- **Expert QC review tools** via separate Streamlit apps:
  - ``scripts/pleb_qc_review.py`` for Plotly-based review
  - ``scripts/pleb_qc_review_matplotlib.py`` for a lower-overhead Matplotlib/table workflow

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

Expert QC review:

.. code-block:: bash

   python -m streamlit run scripts/pleb_qc_review.py -- --run-dir /path/to/run

or:

.. code-block:: bash

   python -m streamlit run scripts/pleb_qc_review_matplotlib.py -- --run-dir /path/to/run

Notes
-----

- The GUI uses ``configs/schemas/gui_schema.json`` for its settings form.
- Validation is best-effort and relies on optional ``jsonschema``.
- For production or automated runs, prefer the CLI or workflow files.
