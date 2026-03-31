UX -> Legacy Mapping
====================

The UX wrapper compiles structured ``pleb.toml`` sections into the existing
legacy flat configuration dictionary used by current PLEB internals.

Why this matters
----------------

- You keep backward compatibility.
- Existing pipeline/ingest/workflow code paths remain unchanged.
- UX can evolve independently from execution internals.

Compilation order
-----------------

The adapter builds legacy config in this order:

1. ``[pipeline]`` forwarded keys
2. ``[paths]`` keys
3. ``[data]`` keys
4. ``[run]`` keys (except UX-only ``mode``/``profile``)
5. ``[workflow]`` keys (with ``file`` -> ``workflow_file``)
6. ``[policy]`` keys
7. extra unknown top-level keys

Later writes can override earlier values.

Policy nested-group prefixes
----------------------------

Nested policy sections map by prefix:

- ``[policy.fix]`` -> ``fix_*``
- ``[policy.pqc]`` -> ``pqc_*``
- ``[policy.report]`` -> ``qc_report_*``
- ``[policy.qc_report]`` -> ``qc_report_*``
- ``[policy.ingest]`` -> ``ingest_*``

Rules:

- if nested key already starts with target prefix, it is kept unchanged;
- otherwise prefix is prepended.

Examples
--------

Example 1
~~~~~~~~~

.. code-block:: toml

   [policy.fix]
   apply = true
   branch_name = "fix1"

Compiles to:

- ``fix_apply = true``
- ``fix_branch_name = "fix1"``

Example 2
~~~~~~~~~

.. code-block:: toml

   [policy.pqc]
   backend_col = "sys"
   pqc_drop_unmatched = false

Compiles to:

- ``pqc_backend_col = "sys"``
- ``pqc_drop_unmatched = false``

Example 3
~~~~~~~~~

.. code-block:: toml

   [workflow]
   file = "configs/workflows/branch_chained_fix_pqc_variants.toml"

Compiles to:

- ``workflow_file = "configs/workflows/branch_chained_fix_pqc_variants.toml"``

Conflict handling
-----------------

If the same legacy key is set in multiple sections, the later stage in the
compilation order wins.

Practical guidance:

- Keep stable baseline in ``[paths]``, ``[data]``, ``[run]``.
- Keep advanced one-off keys in ``[pipeline]`` only when necessary.
- Avoid setting same key in many places.

Introspection
-------------

Use:

.. code-block:: bash

   pleb explain --config pleb.toml

to inspect section-level mapping summary.

For a full key-level crosswalk, keep :doc:`../full_settings_catalog` as source
of truth for legacy flat keys.
