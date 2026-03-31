Developer Internals
===================

This chapter documents implementation-level surfaces.

Key modules
-----------

- UX layer: ``pleb/ux/*``
- Workflow engine: ``pleb/workflow.py``
- Pipeline run coordination: ``pleb/pipeline.py``
- Config models: ``pleb/config.py``

Contract notes
--------------

- Workflow schema version is explicit: ``workflow_version = 1``.
- Advanced optional stages (whitenoise, compare-public) are implemented as optional workflow steps.
- The UX key catalog is centralized in ``pleb/ux/key_catalog.py``.
