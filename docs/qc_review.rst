QC Expert Review
================

``pleb.qc_review`` is an additive review layer for PQC outputs. It does not
modify raw ``*_qc.csv`` files. Instead it writes:

- ``manual_qc_overrides.csv``: append-only expert decisions.
- ``reviewed_qc.csv``: raw QC rows plus reviewed decision columns.

Run the Streamlit reviewer
--------------------------

Install GUI extras:

.. code-block:: bash

   pip install -e ".[gui]"

Launch the reviewer:

.. code-block:: bash

   python -m streamlit run scripts/pleb_qc_review.py -- --run-dir /path/to/run

The review UI loads all ``*_qc.csv`` files below the run directory, assigns a
stable ``review_id`` to each row, plots residuals against MJD, and lets the
reviewer mark selected rows.

Manual actions
--------------

Supported actions are:

- ``mark_bad``: force the reviewed decision to ``BAD_TOA``.
- ``mark_event``: force the reviewed decision to ``EVENT``.
- ``keep``: force the reviewed decision to ``KEEP``.
- ``clear_auto_bad``: force an automatically bad row to ``KEEP``.
- ``clear_manual``: clear the latest manual override and fall back to the
automatic decision.

Non-interactive merge
---------------------

To merge an existing override CSV into raw QC output:

.. code-block:: bash

   python -m pleb.qc_review \
       --run-dir /path/to/run \
       --overrides /path/to/manual_qc_overrides.csv \
       --out /path/to/reviewed_qc.csv

Design notes
------------

Manual review uses ``review_id``/``qc_csv``/``row_index`` identity. It does not
rely on MJD-only matching. This is deliberate: duplicate or near-duplicate TOAs
are common enough that MJD-only expert overrides are unsafe.
