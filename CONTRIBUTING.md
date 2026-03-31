# Contributing

## Scope Policy (Current Release Line)

This project is currently in a **scope freeze** for user-facing surface area.

### In scope

- bug fixes,
- stability/reproducibility work,
- tests and CI hardening,
- documentation improvements and reorganization.

### Out of scope (for this release line)

- new user-facing config keys/modes/templates,
- breaking config/workflow behavior changes,
- large feature additions without roadmap approval.

## Next-Major Work

The approved next-major initiative is:

- **N-pass compiler DSL** (tracked as roadmap item, not part of current frozen scope).

Implement this on dedicated feature branches and target the next milestone.

## Pull Request Guidance

For each PR, include:

1. scope statement (`bugfix/docs/test/infra` vs `next-major`),
2. behavior impact summary,
3. tests added/updated,
4. docs updates (if user-facing behavior changed).
