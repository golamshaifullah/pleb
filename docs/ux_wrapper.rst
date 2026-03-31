UX Wrapper (Single-File Interface)
==================================

This chapter is the landing page for the UX wrapper around PLEB configuration.

Goal
----

Give users a simpler entrypoint:

- one user-facing file (``pleb.toml``),
- a small command set (``init/run/profile/doctor/explain``),
- compatibility with existing PLEB internals.

What this chapter set covers
----------------------------

- command-level usage,
- UX config schema,
- UX->legacy mapping behavior,
- presets,
- mode routing,
- troubleshooting,
- migration from legacy run files.

Reading order
-------------

1. :doc:`ux_wrapper/getting_started`
2. :doc:`ux_wrapper/schema`
3. :doc:`ux_wrapper/mapping`
4. :doc:`ux_wrapper/modes`
5. :doc:`ux_wrapper/troubleshooting`

UX deep pages
-------------

.. toctree::
   :maxdepth: 1

   ux_wrapper/getting_started
   ux_wrapper/schema
   ux_wrapper/mapping
   ux_wrapper/presets
   ux_wrapper/modes
   ux_wrapper/troubleshooting
   ux_wrapper/migration

Related docs
------------

- Config system guide: :doc:`configuration`
- Config layout map: :doc:`config_layout`
- Deep dive architecture: :doc:`pleb_deep_dive`
- Full key catalog (legacy flat keys): :doc:`full_settings_catalog`
- CLI reference: :doc:`cli`
