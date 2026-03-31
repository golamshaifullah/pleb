Operator Quickstart
===================

This is the shortest production path.

1. Generate a golden-path scaffold:

.. code-block:: bash

   pleb init --workflow-template golden-path --outdir configs --force

2. Set paths and pulsars in:

- ``configs/project.toml``
- ``configs/policy.toml``

3. Validate:

.. code-block:: bash

   pleb doctor --config configs/runs/workflow/pleb.workflow.toml

4. Plan:

.. code-block:: bash

   pleb run --config configs/runs/workflow/pleb.workflow.toml --plan

5. Run:

.. code-block:: bash

   pleb run --config configs/runs/workflow/pleb.workflow.toml
