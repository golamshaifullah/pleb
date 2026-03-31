tempo2, PQC, and Reporting
===========================

This chapter covers runtime expectations and output interpretation for the
analysis stages.

tempo2 run coordination
--------------------

PLEB can run tempo2 directly or through a configured container image depending
on runtime configuration.

Check:

- executable/image availability,
- bind paths for containerized runs,
- expected par/tim paths resolved by stage.

PQC integration (forwarded settings only)
-----------------------------------

PLEB forwards ``pqc_*`` settings and does not re-implement detector logic.

Use PLEB for:

- stage coordination,
- branch/result routing,
- post-QC apply policy.

Use PQC docs for:

- detector assumptions,
- outlier/event model semantics,
- detector-specific thresholds.

PQC docs:
https://golamshaifullah.github.io/pqc/index.html

Reporting outputs
-----------------

PLEB reporting can include:

- QC summary tables,
- plots,
- compact PDF,
- per-backend action CSV lists,
- optional cross-pulsar coincidence report.

Compact report usage
--------------------

Use compact report when you need a reviewer-oriented summary instead of raw
output-file browsing.

Align compact report outlier columns with your apply strategy so report and
action policy are consistent.

Cross-pulsar coincidence
------------------------

Optional post-QC stage that clusters temporally coincident anomalies across
pulsars using configured window/minimum thresholds.

Use as triage signal; do not treat as standalone detector proof.
