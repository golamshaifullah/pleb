Getting Started
===============

This page gives the fastest path to use the UX wrapper safely.

Quick start
-----------

1. Create starter config:

.. code-block:: bash

   pleb init

2. Apply a preset:

.. code-block:: bash

   pleb profile list
   pleb profile use minimal --config configs/runs/pipeline/pleb.pipeline.toml

3. Inspect resolved mapping:

.. code-block:: bash

   pleb doctor --config configs/runs/pipeline/pleb.pipeline.toml
   pleb explain --config configs/runs/pipeline/pleb.pipeline.toml

4. Run:

.. code-block:: bash

   pleb run --config configs/runs/pipeline/pleb.pipeline.toml

Golden-path journey shortcuts:

.. code-block:: bash

   pleb run detect --config configs/runs/pipeline/pleb.pipeline.toml --confirm
   pleb run apply --config configs/runs/pipeline/pleb.pipeline.toml --confirm
   pleb run publish --config configs/runs/pipeline/pleb.pipeline.toml

Plan first (no execution):

.. code-block:: bash

   pleb run --config configs/runs/pipeline/pleb.pipeline.toml --plan

5. Override a setting without editing file:

.. code-block:: bash

   pleb run --config configs/runs/pipeline/pleb.pipeline.toml --set data.jobs=8 --set run.run_pqc=true

Command reference
-----------------

``pleb init``
~~~~~~~~~~~~~

Creates starter UX config files.

.. code-block:: bash

   pleb init --config pleb.toml --force

Mode-specific starter (with verbosity):

.. code-block:: bash

   pleb init --mode pipeline --level minimal
   pleb init --mode pipeline --level balanced
   pleb init --mode pipeline --level full
   pleb init --mode ingest
   pleb init --mode workflow
   pleb init --mode qc-report

Generate one file per mode:

.. code-block:: bash

   pleb init --all-modes --outdir configs --level balanced --force

Generate a 3-pass workflow blueprint (detect -> apply -> post-clean):

.. code-block:: bash

   pleb init --workflow-template 3pass-clean --outdir configs --force

Use ``--force`` only when you explicitly want to overwrite.

``pleb profile list``
~~~~~~~~~~~~~~~~~~~~~

Lists preset names from ``configs/presets/*.toml``.

``pleb profile use <name>``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deep-merges a preset into your UX config.

.. code-block:: bash

   pleb profile use balanced --config pleb.toml

``pleb doctor``
~~~~~~~~~~~~~~~

Runs lightweight validation and prints a quick summary of resolved mode and
required keys.

``pleb explain``
~~~~~~~~~~~~~~~~

Prints section-level UX->legacy mapping summary for debugging.

``pleb run``
~~~~~~~~~~~~

Translates UX config into the current flat PLEB config and routes to the selected mode
execution.

Profiles can be applied at run time before ``--set`` overrides:

.. code-block:: bash

   pleb run --config configs/runs/pipeline/pleb.pipeline.toml --profile balanced

What files you edit vs what you do not edit
-------------------------------------------

Edit directly:

- ``pleb.toml``

Usually do not edit directly (unless advanced use):

- ``configs/runs/*``
- ``configs/workflows/*``
- ``configs/catalogs/*``
- ``configs/rules/*``

Common first-run checklist
--------------------------

- Set ``paths.home_dir`` correctly.
- Set ``paths.singularity_image`` correctly.
- Set desired branch scope under ``data``.
- Confirm ``run.mode`` (pipeline/ingest/workflow/qc-report).
- Use ``doctor`` before first run.
