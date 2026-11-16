# Duplicate TOAs are commented by default

Within-file deduplication now defaults to `fix_dedupe_action = "comment"`.
Every rejected copy remains in its original position as:

```text
C DUPLICATE_TOA original_filename frequency mjd uncertainty observatory ...
```

The original TOA and its flag/value pairs remain available. The first active
matching row survives; existing comments never act as survivors. Repeating the
pass leaves already-commented copies unchanged. The fix report records the
matching rule, duplicate and retained input line numbers, and both row hashes.
`duplicates` counts exclusions; `commented` and `removed` distinguish actions.
Dry-run counts describe planned actions and do not imply that files were written.

The TOML settings are:

```toml
fix_dedupe_action = "comment"
fix_dedupe_comment_prefix = "C DUPLICATE_TOA"
fix_dedupe_same_obs_same_bw_globs = ["WSRT.*.tim", "JBO.ROACH*.tim"]
```

The ROACH heuristic is enabled only by explicit configuration. The ordinary
default remains WSRT-only. Physical deletion remains available only through
an explicit `fix_dedupe_action = "delete"` setting.

`configs/runs/fixdataset/epta_dr3_duplicate_comments.toml` provides a separate
cleanup pass. Its dedicated exact-overlap catalog retains the correctly named
S60.4857 file and comments identical S60.1420 copies. Cross-file overlap
handling already uses comments. The profile preserves PAR files and INCLUDE
wrappers. It does not add a fuzzy cross-file J1713 rule or alter which processing
version survives the ROACH heuristic; those choices remain separate policies.

Use a disposable Git repository containing the reconstructed dataset, and
override the profile's `home_dir`, `dataset_name` and `fix_base_branch` to match
it. Copy the dedicated overlap catalog into that repository at its declared
relative path, or pass an absolute `fix_overlap_exact_catalog_path` override.

The existing SIF images contain the earlier PLEB code. Use the updated source
in their runtime (with an explicitly recorded source overlay) or build a new
image from a commit containing these changes. No existing SIF, frozen portable
reconstruction archive or release dataset is changed by this source update.
