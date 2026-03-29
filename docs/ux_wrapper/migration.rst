Migration from Legacy Configs
=============================

This page describes safe migration from existing run/workflow TOML usage to
``pleb.toml`` UX usage.

Migration goals
---------------

- keep behavior parity,
- keep backward compatibility,
- reduce direct editing of many files.

Recommended migration plan
--------------------------

Step 1: Baseline snapshot
~~~~~~~~~~~~~~~~~~~~~~~~~

- identify currently used legacy config file(s),
- capture command invocations and run outputs used for comparison.

Step 2: Initialize UX file
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pleb init --config pleb.toml

Step 3: Copy core paths/scope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Populate:

- ``[paths]``
- ``[data]``

with values equivalent to your existing legacy profile.

Step 4: Set mode and core toggles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Populate:

- ``[run]`` section

with equivalent stage toggles.

Step 5: Move policy by group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Populate:

- ``[policy.fix]``
- ``[policy.pqc]``
- ``[policy.report]``
- ``[workflow]`` (if needed)

Step 6: Preserve advanced keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For any unmapped advanced setting, place it under ``[pipeline]`` exactly as
legacy flat key.

Step 7: Compare and validate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- run ``pleb doctor`` and ``pleb explain``,
- run pilot on small scope,
- compare branch artifacts and run outputs with legacy baseline.

Parity checklist
----------------

- same target repo/data roots,
- same branch/base branch semantics,
- same pulsar selection,
- same stage toggles,
- same policy keys (especially fix/pqc action behavior),
- same workflow file in workflow mode.

Rollback strategy
-----------------

Because execution internals are unchanged, rollback is trivial:

- run legacy command/config directly,
- keep UX wrapper changes in separate commit/branch until parity is proven.

Team adoption strategy
----------------------

- adopt UX wrapper for new users first,
- keep legacy configs for power users during transition,
- gradually codify team presets in ``configs/presets``.
