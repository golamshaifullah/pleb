Ingest Mapping
==============

The ingest mode builds a canonical pulsar layout from arbitrary folders using
an explicit JSON mapping file. Backend names are **never auto-parsed**. Every
backend is defined in the mapping file using the PETA naming convention:

``TEL.BACKEND.CENFREQ``

Canonical output layout
-----------------------

For each pulsar, ingest writes:

- ``Jxxxx+xxxx/Jxxxx+xxxx.par``
- ``Jxxxx+xxxx/Jxxxx+xxxx_all.tim`` (includes each backend tim)
- ``Jxxxx+xxxx/tims/TEL.BACKEND.CENFREQ.tim``
- ``Jxxxx+xxxx/tmplts/<original_template_name>``

Running ingest
--------------

You can run ingest as a subcommand or as a config-driven mode:

.. code-block:: bash

   pleb ingest --mapping configs/ingest_mapping.example.json --output-dir /data/epta/EPTA

Or via config:

.. code-block:: toml

   ingest_mapping_file = "configs/ingest_mapping.example.json"
   ingest_output_dir = "/data/epta/EPTA"

Then:

.. code-block:: bash

   pleb --config pipeline.toml

JSON schema
-----------

Schema file: ``configs/ingest_mapping.schema.json``.

.. code-block:: json

   {
     "$schema": "https://json-schema.org/draft/2020-12/schema",
     "title": "pleb ingest mapping",
     "type": "object",
     "properties": {
       "sources": { "type": "array", "items": { "type": "string" } },
       "par_roots": { "type": "array", "items": { "type": "string" } },
       "template_roots": { "type": "array", "items": { "type": "string" } },
       "ignore_backends": { "type": "array", "items": { "type": "string" } },
       "pulsar_aliases": { "type": "object", "additionalProperties": { "type": "string" } },
       "backends": {
         "type": "object",
         "additionalProperties": {
           "type": "object",
           "properties": {
             "root": { "type": "string" },
             "ignore": { "type": "boolean" },
             "tim_glob": { "type": "string" },
             "ignore_suffixes": { "type": "array", "items": { "type": "string" } }
           },
           "required": ["root"]
         }
       }
     }
   }

Example mapping
---------------

.. code-block:: json

   {
     "sources": [
       "/data/epta/raw",
       "/data/epta/legacy"
     ],
     "par_roots": [
       "/data/epta/raw/parfiles",
       "/data/epta/legacy/par"
     ],
     "template_roots": [
       "/data/epta/raw/templates"
     ],
     "ignore_backends": [
       "NRT.OLD.CHECK"
     ],
     "pulsar_aliases": {
       "B1937+21": "J1939+2134",
       "B1855+09": "J1857+0943"
     },
     "backends": {
       "EFF.P200.1360": {
         "root": "/data/epta/raw/tim/EFF/P200/1360",
         "tim_glob": "*.tim",
         "ignore_suffixes": ["_all.tim"]
       },
       "NRT.NUPPI.1480": {
         "root": "/data/epta/legacy/NRT/NUPPI/1480",
         "tim_glob": "*.tim"
       },
       "JBO.ROACH.1520": {
         "root": "/data/epta/raw/JBO/ROACH/1520",
         "ignore": false
       }
     }
   }

How to populate the mapping
---------------------------

1. **List data roots** (optional): ``sources`` is informative only.
2. **Define par roots**: directories where ``*.par`` are stored.
3. **Define template roots**: directories containing profile templates.
4. **Add pulsar aliases**: map every B-name to its canonical J-name.
5. **Define backends**: one entry per backend using the PETA naming convention.
   Each backend entry must include a ``root`` path that contains the tim files
   for that backend. No automatic parsing or guessing is performed.
6. **Ignore lists**: if a backend is listed in ``ignore_backends`` it is
   skipped entirely.

Strict mapping rules
--------------------

- Backend names come **only** from the mapping file keys.
- If a tim file is found but its pulsar name cannot be resolved to a J-name
  (via explicit mapping), ingest fails.
- If multiple par files map to the same pulsar, ingest fails.
