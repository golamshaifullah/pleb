# Pleb + PQC Singularity

For the DR3 timing/analysis images and interactive workbooks, use the maintained
pair under `pleb_singularity/`: `pleb_dr3_upstream.def` followed by
`discovery_dr3_upstream.def`. See [the dependency audit and build instructions](DEPENDENCIES.md)
for included notebook/GUI/storage features, optional profiles and validation limits.
The generic workflow below builds a different, older image family.

This definition builds a container on top of a configurable `psrpta.sif`
and installs both `pleb` and `pqc`.

Build from the repo root (adjust paths if your layout differs):

```sh
BASE_IMAGE=/path/to/psrpta.sif \
  singularity/build_pleb_sif.sh
```

If your sources live elsewhere:

```sh
BASE_IMAGE=/path/to/psrpta.sif \
PLEB_SRC=/path/to/pleb \
PQC_SRC=/path/to/pqc \
  singularity/build_pleb_sif.sh
```

Run:

```sh
singularity exec pleb.sif pleb --help
```

Default bundled layout inside the container:

- `/opt/pleb` (pleb source)
- `/opt/pqc` (pqc source)
- `/opt/pleb_data/configs/settings`
- `/opt/pleb_data/configs/workflows`
- `/opt/pleb_data/results`
- `/opt/pleb_data/logs`

Useful environment vars:

- `PLEB_DATA=/opt/pleb_data`
- `PLEB_CONFIGS=/opt/pleb_data/configs`

Build note:

- The base image uses Python 3.9, while pleb declares `>=3.10`. The build
  sets `PIP_IGNORE_REQUIRES_PYTHON=1` to install anyway. If you want a strict
  install, rebuild on a base image with Python 3.10+.
- The container installs all extras: `pleb[plot,gui,qc]`.

## Reproducible Tempo2 + Discovery image

`pleb_discovery.def.in` extends the validated production image without
rebuilding Tempo2. This is intentional: it preserves the exact Tempo2 binary,
runtime observatory database, plugins, clocks, and libstempo ABI that produced
the accepted residuals.

Build it with:

```sh
singularity/build_pleb_discovery_sif.sh
```

The build script verifies the SHA-256 checksum of the base image before doing
anything. To use a deliberately replaced base image, supply both values:

```sh
BASE_IMAGE=/path/to/pleb_tempo2.sif \
EXPECTED_BASE_SHA256=$(sha256sum /path/to/pleb_tempo2.sif | awk '{print $1}') \
OUT_SIF=pleb_tempo2_discovery.sif \
  singularity/build_pleb_discovery_sif.sh
```

The derived image pins Enterprise, Enterprise Extensions, and Discovery to the
commits used by the working analysis environment. Its test also checks the
embedded Tempo2 and libstempo commit locks.
