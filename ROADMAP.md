# Roadmap

This roadmap uses three buckets: **Now**, **Next**, **Later**.

## Now (Scope Freeze: current release line)

The current release line is scope-frozen for user-facing surface area.

Allowed:
- bug fixes,
- stability/reproducibility improvements,
- tests,
- documentation/organization updates.

Not allowed in this line:
- new user-facing config surface,
- new run modes,
- breaking workflow/config changes.

## Next

- **N-pass compiler DSL** (planned next major initiative):
  - declarative pass graph (N passes),
  - compile to workflow/run configs,
  - validation + dry-run plan output,
  - backward-compatible bridge from existing templates.

## Later

- richer project scaffolding around `project.toml` + `policy.toml`,
- stronger acceptance gates between detect/apply/publish journeys,
- reference dataset + golden-output checks in CI.
