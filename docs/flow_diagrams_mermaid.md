# pleb Flow Diagrams (Mermaid)

## Pipeline Mode

```mermaid
flowchart TD
    A[Load config + --set overrides] --> B[Resolve paths branches pulsars]
    B --> C{run_fix_dataset?}
    C -->|yes| D[FixDataset stage]
    C -->|no| E
    D --> E{run_tempo2?}
    E -->|yes| F[tempo2 stage]
    E -->|no| G
    F --> G{run_pqc?}
    G -->|yes| H[PQC stage]
    G -->|no| I
    H --> I{qc_report?}
    I -->|yes| J[QC report stage]
    I -->|no| K[Write outputs]
    J --> K[Write outputs]
```

## Ingest Mode

```mermaid
flowchart TD
    A[Read ingest mapping] --> B[Discover source par tim tmplts]
    B --> C[Copy to standard pulsar layout]
    C --> D{ingest_verify or lockfile checks?}
    D -->|yes| E[Validate expected files + lock state]
    D -->|no| F
    E --> F{ingest_commit_branch?}
    F -->|yes| G[Commit to ingest branch]
    F -->|no| H[Done]
    G --> H[Done]
```

## Param-Scan Mode

```mermaid
flowchart TD
    A[Load param-scan config] --> B[Run baseline fit metrics]
    B --> C[Test candidate params/models]
    C --> D[Rank by delta-chi2 and diagnostics]
    D --> E[Write scan outputs]
```

## QC-Report Mode

```mermaid
flowchart TD
    A[Find existing *_qc.csv] --> B[Diagnostics + summary plots]
    B --> C{compact_pdf?}
    C -->|yes| D[Build compact PDF]
    C -->|no| E[Write qc_report dir]
    D --> E[Write qc_report dir]
```

## Workflow Mode (Serial + Parallel Groups)

```mermaid
flowchart TD
    A[Load workflow + base config] --> B[Apply global set overrides]
    B --> C{Top-level steps or groups}
    C --> D[Run group 1]
    D --> E[Barrier wait for group 1 completion]
    E --> F[Run group 2]
    F --> G[Barrier wait for group 2 completion]
    G --> H{loops?}
    H -->|yes| I[Run loop iterations with mode serial or parallel]
    H -->|no| J[Done]
    I --> J[Done]
```

## PQC Usage: Detection Only

```mermaid
flowchart TD
    A[run_pqc=true] --> B[Generate per-pulsar *_qc.csv]
    B --> C[Write qc_summary.tsv]
    C --> D[No tim/par edits]
```

## PQC Usage: Detect Then Apply Comments

```mermaid
flowchart TD
    subgraph StageA[Stage A: Detect]
      A1[Pipeline run with run_pqc=true] --> A2[Write *_qc.csv]
    end
    subgraph StageB[Stage B: Apply]
      B1[FixDataset apply] --> B2[Load QC CSV per pulsar]
      B2 --> B3[Match TOAs by MJD backend timfile]
      B3 --> B4[Comment or delete per fix_qc settings]
      B4 --> B5[Commit on fix branch]
    end
    A2 --> B1
```

## PQC Detection Sequence (Outliers + Events)

```mermaid
flowchart TD
    A[Parse tim metadata INCLUDE recursion] --> B[Load libstempo arrays]
    B --> C[Merge timing arrays + tim metadata]
    C --> D[Ensure backend keys sys group]
    D --> E[Feature extraction orbital/solar/freq/altaz]
    E --> F{Structure detrending/tests enabled?}
    F -->|yes| G[Detrend residuals by feature bins]
    F -->|no| H
    G --> H[Outlier detectors]
    H --> H1[OU bad-measurement]
    H1 --> H2[Robust MAD outliers backend/global]
    H2 --> H3[Optional hard sigma gate]
    H3 --> I[Event detectors]
    I --> I1[Transients]
    I1 --> I2[Exp dips]
    I2 --> I3[Step + DM-step]
    I3 --> I4[Solar + Eclipse]
    I4 --> I5[Gaussian bump + Glitch]
    I5 --> J[Event-aware reconciliation]
    J --> J1[Remove event members from bad_point]
    J1 --> J2[Compute event_member]
    J2 --> J3[Set outlier_any compatibility field]
    J3 --> K[Write *_qc.csv]
```
