# Pleb + PQC Singularity

This definition builds a container on top of a configurable `psrpta.sif`
and installs both `pleb` and `pqc`.

Build from the repo root (adjust paths if your layout differs):

```sh
BASE_IMAGE=/path/to/psrpta.sif \
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
