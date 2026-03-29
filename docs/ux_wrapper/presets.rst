Presets
=======

Presets are reusable UX config fragments stored in:

- ``configs/presets/*.toml``

Current built-ins
-----------------

- ``minimal``
- ``balanced``

List presets
------------

.. code-block:: bash

   pleb profile list

Apply preset
------------

.. code-block:: bash

   pleb profile use minimal --config pleb.toml
   pleb profile use balanced --config pleb.toml

Merge semantics
---------------

Preset application uses deep-merge:

- nested dictionaries are merged recursively,
- scalar/list values in preset override existing values at same path,
- unrelated existing keys are preserved.

Recommended usage pattern
-------------------------

1. ``pleb init``
2. ``pleb profile use minimal``
3. edit paths/branches
4. run pilot
5. ``pleb profile use balanced`` for stricter run profile

Authoring custom presets
------------------------

Create new files under ``configs/presets/`` and keep them focused by intent:

- ``qc_conservative.toml``
- ``fix_comment_only.toml``
- ``workflow_detect_apply.toml``

Use comments at top describing expected use-case and safety level.
