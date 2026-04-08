Ingest: Build The Canonical Pulsar Tree
=======================================

This page explains how to get from a messy source-data layout to the canonical
dataset tree that ``pleb`` expects.

Ingest comes first. If ingest is wrong, every later stage is downstream of a
bad filesystem model.


What Ingest Does
----------------

Ingest reads an explicit mapping file and writes a standard per-pulsar layout.

It does not guess backend names. Backend names come from the mapping file.

According to :doc:`../ingest`, ingest writes:

- ``Jxxxx+xxxx/Jxxxx+xxxx.par``
- ``Jxxxx+xxxx/Jxxxx+xxxx_all.tim``
- ``Jxxxx+xxxx/tims/TEL.BACKEND.CENFREQ.tim``
- ``Jxxxx+xxxx/tmplts/...``


Two Files Are Required
----------------------

For a normal ingest setup, create:

1. a mapping JSON under ``configs/catalogs/ingest/``,
2. an ingest run profile under ``configs/runs/ingest/``.


The Mapping JSON
----------------

The mapping file describes where source files live and how backend names are
assigned.

Example:
``configs/catalogs/ingest/single_pulsar_mapping.json``

Tracked repository example:
``configs/catalogs/ingest/single_pulsar_mapping.example.json``

.. code-block:: json

   {
     "sources": [
       "/data/raw_release"
     ],
     "par_roots": [
       "/data/raw_release/par"
     ],
     "template_roots": [
       "/data/raw_release/templates"
     ],
     "pulsar_aliases": {
       "B1907-3744": "J1909-3744"
     },
     "backends": {
       "EFF.P200.1360": {
         "root": "/data/raw_release/tim/EFF/P200/1360",
         "tim_glob": "*.tim",
         "ignore_suffixes": ["_all.tim"]
       },
       "NRT.NUPPI.1480": {
         "root": "/data/raw_release/tim/NRT/NUPPI/1480",
         "tim_glob": "*.tim"
       }
     }
   }


How To Read The Mapping Keys
----------------------------

``sources``
  Informational root list. This helps the student document where material came
  from, but does not define backend identity.

``par_roots``
  Directories where ingest looks for parfiles.

``template_roots``
  Directories where template files live.

``pulsar_aliases``
  Explicit alias map, usually B-name to J-name. If a name cannot be resolved
  cleanly, ingest should not guess.

``backends``
  The most important part of the mapping. Each key is the canonical backend
  name that will be used later. Each backend entry must point to a real source
  directory.

``tim_glob``
  The pattern used to find source tim files within that backend root.

``ignore_suffixes``
  A way to skip known files such as pre-existing aggregate ``_all.tim`` files.


Why The Backend Key Matters
---------------------------

The backend key is not just a label. It becomes part of the later dataset
structure and later QC grouping logic.

If the mapping uses inconsistent backend names, the student will later get:

- broken or misleading jump logic,
- bad system grouping,
- confusing PQC backend splits.

Rule:

The ingest mapping defines the canonical backend identity used by downstream
stages.


The Ingest Run Profile
----------------------

Example:
``configs/runs/ingest/single_pulsar_ingest.toml``

Tracked repository example:
``configs/runs/ingest/single_pulsar_ingest.example.toml``

.. code-block:: toml

   ingest_mapping_file = "configs/catalogs/ingest/single_pulsar_mapping.json"
   ingest_output_dir = "/data/canonical/EPTA-DR3/epta-dr3-data"

   ingest_verify = true
   ingest_commit_branch_name = "raw_ingest"
   ingest_commit_base_branch = "ingest"
   ingest_commit_message = "Ingest: single-pulsar import"

This mirrors the role of ``configs/runs/ingest/ingest_epta_data.toml``.


How To Explain Each Ingest Run Key
----------------------------------

``ingest_mapping_file``
  Path to the JSON mapping file.

``ingest_output_dir``
  Where the canonical pulsar tree is written.
  This is the directory later referenced indirectly by ``home_dir`` and
  ``dataset_name`` in pipeline-style run profiles.

``ingest_verify``
  Turn on ingest checks so mapping or alias errors fail early.

``ingest_commit_branch_name``
  Branch that receives the ingested dataset state.

``ingest_commit_base_branch``
  Existing branch ingest starts from.

``ingest_commit_message``
  Git commit message for the ingest mutation.


Where The Data Ends Up
----------------------

Suppose:

- ``home_dir = "/data/canonical"``
- ``dataset_name = "EPTA-DR3/epta-dr3-data"``

Then the canonical dataset root later used by pipeline profiles is:

.. code-block:: text

   /data/canonical/EPTA-DR3/epta-dr3-data

Inside that tree, it should be possible to locate:

.. code-block:: text

   /data/canonical/EPTA-DR3/epta-dr3-data/J1909-3744/

and inside it:

.. code-block:: text

   J1909-3744.par
   J1909-3744_all.tim
   tims/


How To Run Ingest
-----------------

Run:

.. code-block:: bash

   pleb ingest --config configs/runs/ingest/single_pulsar_ingest.toml

After it finishes, inspect the output tree before doing anything else.


What Ingest Does Not Do
-----------------------

Ingest standardizes the file layout. It does not by itself:

- run tempo2,
- insert jumps,
- infer the final QC grouping policy,
- decide which TOAs are suspicious,
- apply FixDataset mutations beyond the ingest-specific commit step.

Those tasks happen later.


What To Check After Ingest
--------------------------

For one pulsar, verify all of the following manually:

1. the pulsar directory exists,
2. the pulsar parfile exists,
3. the ``Jxxxx_all.tim`` include file exists,
4. backend tim files exist under ``tims/``,
5. backend filenames match the intended canonical names,
6. there is only one parfile for the pulsar,
7. aliases were resolved to the intended J-name.

If any of these checks fail, do not proceed to FixDataset.


Common Ingest Mistakes
----------------------

- mapping the wrong source directory to a backend key,
- forgetting a B-name to J-name alias,
- accidentally ingesting an existing aggregate ``_all.tim``,
- using inconsistent backend keys for conceptually identical systems,
- using a mapping file that was never reviewed against the source
  release tree.


Why This Stage Matters
----------------------

Do not treat ingest as a clerical stage.

Ingest is the stage where raw source structure is translated into the naming
and grouping model that later
operations depend on.


Detailed Mapping Guidance
-------------------------

For more depth on mapping structure and failure conditions, see :doc:`../ingest`
and the JSON schema at
``configs/schemas/ingest_mapping.schema.json``.

When reviewing a mapping file, check three things in order:

1. backend names are canonical and stable,
2. pulsar aliases are complete enough to resolve all names,
3. source roots do not accidentally include derived or aggregate files that
   should not be re-ingested.


How Ingest Connects To The Next Stage
-------------------------------------

After a successful ingest run, the next stage is usually a FixDataset pass that
starts from the ingest branch. In practice, that means:

- ingest writes the canonical tree,
- ingest commits it to a branch such as ``raw_ingest``,
- the Step-1 FixDataset profile then sets
  ``fix_base_branch = "raw_ingest"``.

This is why the ingest branch name matters even in a single-pulsar workflow.


Related Documentation
---------------------

- ingest mode overview and schema: :doc:`../ingest`
- CLI entry point details: :doc:`../cli`
- mode selection and compatibility notes: :doc:`../running_modes`
