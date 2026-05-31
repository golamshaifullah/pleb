# Optimize Scoring

PLEB optimization ranks candidate PQC settings with a scalar objective score. The
optimizer samples a candidate setting, runs PQC, computes summary metrics from the
QC outputs, and keeps the candidate with the highest score.

The production single-pulsar objective is defined in
`configs/optimize/objectives/single_pulsar_variant_consensus_production.toml`.
It currently uses `maximize = true` and `score_offset = 0.0`, so larger scores
are better.

## Fully Expanded Production Score

The score is the weighted sum below:

```text
score =
    2.00 * residual_cleanliness
  + 1.00 * residual_whiteness
  + 0.75 * event_coherence
  + 0.75 * stability
  + 0.50 * event_stability
  - 1.25 * bad_fraction
  - 0.25 * event_fraction
  - 1.00 * overfragmentation_penalty
  - 0.75 * backend_inconsistency_penalty
  - 0.25 * parameter_complexity_penalty
  + 1.25 * variant_bad_support_among_bad
  + 0.15 * variant_support_mean_present
```

Equivalently:

```text
score =
    2.00 / (1 + MAD(clean residuals))
  + 1.00 / (1 + abs(lag-1 autocorrelation of clean residuals))
  + 0.75 * event_coherence
  + 0.75 / (1 + std(bad_fraction across folds))
  + 0.50 / (1 + std(event_fraction across folds))
  - 1.25 * (n_bad / n_toas)
  - 0.25 * (n_event_members / n_toas)
  - 1.00 * (n_singleton_events / n_events)
  - 0.75 * normalized_entropy(bad_TOA_backend_distribution)
  - 0.25 * parameter_complexity_penalty
  + 1.25 * mean(variant_bad_support among bad TOAs)
  + 0.15 * mean(variant_support_present)
```

Metrics missing from a candidate output are treated as zero for scoring.

## Positive Terms

The residual terms reward a candidate when the TOAs left after QC have compact
residuals and low lag-1 correlation. In practice, this favors settings that
remove real bad points without leaving an obviously structured residual tail.

The event terms reward coherent, reproducible event detection. `event_coherence`
is higher when event-member TOAs are concentrated in a common backend/system, and
the stability terms are higher when bad/event fractions remain similar across
evaluation folds.

The variant-support terms reward agreement across PQC variants. A bad TOA is
more credible when multiple variants support it, and the mean support term gives
a small bonus when the candidate is not relying on a single fragile detection
path.

## Negative Terms

`bad_fraction` and `event_fraction` penalize broad selections. They stop the
optimizer from winning simply by marking too many TOAs as bad or event-affected.

`overfragmentation_penalty` penalizes event detections that break into many
single-TOA events. A physical or instrumental event should usually have some
local structure, not a large collection of isolated one-point labels.

`backend_inconsistency_penalty` penalizes bad TOAs spread evenly across many
backends. The intent is to prefer localized backend/system problems over diffuse
flagging that looks like a global over-selection.

`parameter_complexity_penalty` discourages overly complex candidate settings. If
two settings score similarly, the simpler setting should generally be preferred.

## Legacy Hard-Constraint Scores

Older optimize objectives also included post-apply fit terms and hard
constraints. In that form, PLEB first computed the weighted raw score and then
subtracted a large penalty for each hard-constraint violation:

```text
score =
    raw_weighted_score
  - 1.0e12 * N_constraint_violations
```

For the previous production-style objective, the expanded post-apply form was:

```text
score =
    2.00 * residual_cleanliness
  + 1.00 * residual_whiteness
  + 0.75 * event_coherence
  + 0.75 * stability
  + 0.50 * event_stability
  - 1.25 * bad_fraction
  - 0.25 * event_fraction
  - 1.00 * overfragmentation_penalty
  - 0.75 * backend_inconsistency_penalty
  - 0.25 * parameter_complexity_penalty
  + 1.25 * variant_bad_support_among_bad
  + 0.15 * variant_support_mean_present
  + 8.00 * post_apply_fit_quality
  + 2.50 * post_apply_wrms_quality
  + 3.00 * post_apply_backend_alignment
  - 1.0e12 * N_constraint_violations
```

The post-apply terms were:

```text
post_apply_fit_quality      = 1 / (1 + max(post_apply_redchisq, 0))
post_apply_wrms_quality     = 1 / (1 + max(post_apply_wrms_us, 0))
post_apply_backend_alignment = 1 / (1 + max(max_abs_backend_offset_us, 0))
```

A score near `-1e12` means the candidate violated one hard constraint. A score
near `-2e12` means it violated two hard constraints. These are not normal
low-quality scores; they mean the raw score was overridden by infeasibility.

The hard constraints used in that objective were:

```text
max_backend_bad_fraction <= 0.9
min_backend_n_clean      >= 5
post_apply_redchisq      <= 100
```

## Interpretation

Positive or small single-digit scores are ordinary comparable optimize scores.
Huge negative scores such as `-999999999995` indicate hard-constraint penalties,
not a trial crash. Failed trials are reported separately with `status = failed`
and no usable score.

The optimize score is a ranking heuristic, not a scientific truth label. A high
score means the candidate matched the objective's preference for clean residuals,
coherent events, cross-variant support, restrained flagging, and simple settings.
It does not prove that every selected TOA is bad or that every event label is
astrophysical.
