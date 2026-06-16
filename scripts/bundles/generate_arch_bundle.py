#!/usr/bin/env python3
"""Generate minimal PLEB campaign bundles for scheduler/workstation targets.

The generated bundle always contains the same minimal campaign payload:
``data_source``, ``configs``, ``pleb``, ``scripts/build_release_from_pleb_outputs.py``,
and a git-backed ``repo_state_seed``.  Architecture-specific files are added at
the bundle root:

- HTCondor: DAG renderer/submit/monitor helpers.
- PBS: shared-filesystem qsub launchers.
- Slurm: shared-filesystem sbatch launchers.
- workstation: local multi-process Singularity runner.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import time
from datetime import datetime
from pathlib import Path
from textwrap import dedent

ARCHES = ("htcondor", "pbs", "slurm", "workstation")
EMPTY = "__PLEB_EMPTY__"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def chmod_x(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_executable(path: Path, text: str) -> None:
    path.write_text(dedent(text).lstrip(), encoding="utf-8")
    chmod_x(path)


def write_text(path: Path, text: str) -> None:
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def ensure_clean_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise SystemExit(f"Refusing to overwrite existing bundle dir: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def prune_release_scripts(scripts_dir: Path) -> None:
    keep = {"build_release_from_pleb_outputs.py"}
    if not scripts_dir.exists():
        return
    for item in scripts_dir.iterdir():
        if item.name not in keep:
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def patch_stage_wrapper(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '[optimize.fixed_overrides]\nhome_dir = "."',
        '[optimize.fixed_overrides]\nhome_dir = ".."',
    )
    old = 'if command -v apptainer >/dev/null 2>&1; then\n    OUTER_RUNTIME="apptainer"'
    new = (
        'if [[ "${PLEB_FORCE_SINGULARITY:-0}" == "1" ]] '
        "&& command -v singularity >/dev/null 2>&1; then\n"
        '    OUTER_RUNTIME="singularity"\n'
        "elif command -v apptainer >/dev/null 2>&1; then\n"
        '    OUTER_RUNTIME="apptainer"'
    )
    if old in text and "PLEB_FORCE_SINGULARITY" not in text:
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def commit_seed_if_dirty(seed: Path, message: str) -> None:
    run(["git", "-C", str(seed), "config", "gc.auto", "0"])
    status = subprocess.check_output(
        ["git", "-C", str(seed), "status", "--short"], text=True
    )
    if status.strip():
        run(["git", "-C", str(seed), "add", "-A"])
        run(["git", "-C", str(seed), "commit", "-m", message])
    wait_for_git_pack_quiescence(seed)


def wait_for_git_pack_quiescence(seed: Path) -> None:
    pack_dir = seed / ".git" / "objects" / "pack"
    if not pack_dir.exists():
        return
    for _ in range(200):
        if not list(pack_dir.glob("tmp_pack_*")):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Git pack maintenance did not quiesce under {pack_dir}")


def prepare_campaign(
    repo: Path, bundle: Path, n_trials: int, sif_path: str | None
) -> None:
    campaign = bundle / "campaign"
    campaign.mkdir()
    shutil.copytree(
        repo / "condor_full_epta_dr_optimize" / "scaffold", campaign / "scaffold"
    )
    env = os.environ.copy()
    env.update(
        {
            "PLEB_REPO": str(repo),
            "DATA_SOURCE_SRC": str(repo / "data_source"),
            "CONFIGS_SRC": str(repo / "configs"),
            "PLEB_SRC": str(repo / "pleb"),
            "SCRIPTS_SRC": str(repo / "scripts"),
            "DEFAULT_N_TRIALS": str(n_trials),
        }
    )
    if sif_path:
        env["PLEB_OUTER_SIF"] = sif_path
    run(
        [
            str(repo / "condor_full_epta_dr_optimize" / "prepare_full_bundle.sh"),
            str(campaign),
        ],
        cwd=repo,
        env=env,
    )

    # Keep the runtime script payload minimal.  The release builder is the only
    # top-level script needed inside campaign states.
    prune_release_scripts(campaign / "scripts")
    prune_release_scripts(campaign / "repo_state_seed" / "scripts")

    for wrapper in (
        campaign / "scaffold" / "scripts" / "run_campaign_stage.sh",
        campaign / "repo_state_seed" / "scaffold" / "scripts" / "run_campaign_stage.sh",
    ):
        patch_stage_wrapper(wrapper)

    commit_seed_if_dirty(
        campaign / "repo_state_seed",
        "Minimize bundle scripts and enable Singularity override",
    )


def common_readme(arch: str, name: str, n_trials: int) -> str:
    return f"""
    PLEB {arch} bundle

    Bundle name:
      {name}

    Flow:
      ingest -> step1_all -> step2_all -> pulsar jobs -> merge_final -> release

    Per-pulsar optimization:
      n_trials={n_trials}

    Final release artifact:
      campaign/release_epta_dr3_pleb.tar.gz

    This bundle is intentionally minimal.  It does not include old results,
    previous bundles, pycache trees, or unrelated helper scripts.
    """


def htcondor_scripts(bundle: Path) -> None:
    write_executable(
        bundle / "fix_paths_and_render.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail

        B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        C="$B/campaign"
        DATASET="$C/EPTA-DR3/epta-dr3-data"
        CONTROL="$DATASET/_campaign"

        mkdir -p "$CONTROL/config" "$CONTROL/dag" "$CONTROL/condor_logs"
        printf '%s\n' "$C" > "$C/.campaign_repo_root"
        printf '%s\n' "$DATASET" > "$C/.campaign_dataset_root"
        printf '%s\n' "$CONTROL" > "$C/.campaign_control_root"
        printf '%s\n' "$C" > "$CONTROL/config/repo_root.txt"
        printf '%s\n' "$DATASET" > "$CONTROL/config/dataset_root.txt"
        printf '%s\n' "EPTA-DR3/epta-dr3-data" > "$CONTROL/config/dataset_rel.txt"

        if [[ -n "${PLEB_OUTER_SIF:-}" ]]; then
          printf '%s\n' "$PLEB_OUTER_SIF" > "$CONTROL/config/sif_path.txt"
        elif [[ -f "$HOME/containers/pleb_tempo2.sif" ]]; then
          printf '%s\n' "$HOME/containers/pleb_tempo2.sif" > "$CONTROL/config/sif_path.txt"
        elif [[ ! -s "$CONTROL/config/sif_path.txt" ]]; then
          echo "No pleb_tempo2.sif found. Set PLEB_OUTER_SIF=/path/to/pleb_tempo2.sif and rerun." >&2
          exit 2
        fi

        rm -f "$CONTROL/dag/pleb_campaign.dag" "$CONTROL/dag/pleb_campaign.dag".* "$CONTROL/dag/pleb_campaign.dag".rescue* 2>/dev/null || true
        "$C/scaffold/scripts/render_campaign_dag.sh" "$C" "$CONTROL/dag/pleb_campaign.dag"
        echo "DAG: $CONTROL/dag/pleb_campaign.dag"
        """,
    )
    write_executable(
        bundle / "preflight_bundle.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail

        B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        C="$B/campaign"
        DATASET="$C/EPTA-DR3/epta-dr3-data"
        CONTROL="$DATASET/_campaign"
        CFG="$CONTROL/config"
        DAGDIR="$CONTROL/dag"
        RUNNER="$C/scaffold/scripts/run_campaign_stage.sh"
        SUBMIT="$C/scaffold/launchers/pleb_campaign.htcondor.submit"

        fail() { echo "FAIL: $*" >&2; exit 1; }
        warn() { echo "WARN: $*" >&2; }
        ok() { echo "OK: $*"; }
        read_cfg() {
          local name="$1"
          [[ -f "$CFG/$name" ]] || fail "missing campaign config: $CFG/$name"
          tr -d '\\r\\n' < "$CFG/$name"
        }

        [[ -d "$C" ]] || fail "missing campaign directory: $C"
        [[ -d "$DATASET" ]] || fail "missing dataset directory: $DATASET"
        [[ -f "$RUNNER" ]] || fail "missing stage runner: $RUNNER"
        [[ -f "$SUBMIT" ]] || fail "missing HTCondor submit file: $SUBMIT"

        final_stage="$(read_cfg final_stage.txt)"
        step2_jobs="$(read_cfg step2_jobs.txt)"
        step2_chunks="$(read_cfg step2_chunks.txt)"
        [[ "$final_stage" == "step5_apply_comments" ]] || fail "final_stage=$final_stage, expected step5_apply_comments"
        [[ "$step2_chunks" =~ ^[0-9]+$ ]] || fail "step2_chunks is not numeric: $step2_chunks"
        [[ "$step2_jobs" =~ ^[0-9]+$ ]] || fail "step2_jobs is not numeric: $step2_jobs"
        (( step2_chunks >= 1 )) || fail "step2_chunks must be >= 1"
        (( step2_jobs >= 1 )) || fail "step2_jobs must be >= 1"
        ok "final stage is comment-only: $final_stage"
        ok "Step2 split: chunks=$step2_chunks request_cpus=$step2_jobs"

        if grep -R -E 'fix_qc_action = "delete"|step6_apply_delete' \
          "$C/scaffold" "$C/configs/htcondor_campaign" "$CFG" >/tmp/pleb_preflight_delete_refs.$$ 2>/dev/null; then
          cat /tmp/pleb_preflight_delete_refs.$$
          rm -f /tmp/pleb_preflight_delete_refs.$$
          fail "delete-stage reference found in scaffold/config"
        fi
        rm -f /tmp/pleb_preflight_delete_refs.$$
        ok "no delete-stage references in scaffold/config"

        grep -q 'RESULTS_STASH' "$RUNNER" || fail "merge-step2 result-stash fix missing"
        grep -q 'git -C "$WORKTREE_DIR" clean -fdx' "$RUNNER" || fail "merge-step2 dirty-worktree clean missing"
        ok "merge-step2 dirty-worktree protection is present"

        grep -q 'request_cpus = $(request_cpus:1)' "$SUBMIT" || fail "submit file does not use DAG request_cpus macro"
        ok "HTCondor submit file uses per-node request_cpus"

        if (( step2_chunks > 1 )); then
          for idx in $(seq 1 "$step2_chunks"); do
            n="$(printf '%02d' "$idx")"
            cfg="$C/configs/htcondor_campaign/epta_dr_optimize/step2_detect_variants_chunk${n}.toml"
            seed_cfg="$C/repo_state_seed/configs/htcondor_campaign/epta_dr_optimize/step2_detect_variants_chunk${n}.toml"
            [[ -f "$cfg" ]] || fail "missing chunk config: $cfg"
            [[ -f "$seed_cfg" ]] || fail "missing seed chunk config: $seed_cfg"
            grep -q "jobs = $step2_jobs" "$cfg" || fail "$cfg does not contain jobs = $step2_jobs"
            grep -q "step2_detect_variants_all_chunk${n}" "$cfg" || fail "$cfg has wrong chunk branch"
            grep -q "wf_step2_detect_variants_chunk${n}" "$cfg" || fail "$cfg has wrong chunk outdir"
          done
          ok "Step2 chunk configs are distinct"
        fi

        if [[ -n "${PLEB_OUTER_SIF:-}" ]]; then
          [[ -f "$PLEB_OUTER_SIF" ]] || fail "PLEB_OUTER_SIF does not exist: $PLEB_OUTER_SIF"
        elif [[ -s "$CFG/sif_path.txt" ]]; then
          sif="$(read_cfg sif_path.txt)"
          [[ -f "$sif" ]] || fail "configured SIF does not exist: $sif"
        elif [[ -f "$HOME/containers/pleb_tempo2.sif" ]]; then
          :
        else
          fail "no SIF found; set PLEB_OUTER_SIF=/absolute/path/to/pleb_tempo2.sif"
        fi
        ok "container path is available"

        "$B/fix_paths_and_render.sh" >/tmp/pleb_preflight_render.$$.log
        DAG="$DAGDIR/pleb_campaign.dag"
        [[ -f "$DAG" ]] || fail "DAG was not rendered: $DAG"

        python3 - "$DAG" "$step2_jobs" "$step2_chunks" <<'PY'
import re
import sys
from collections import Counter
from pathlib import Path

dag = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
step2_jobs = sys.argv[2]
step2_chunks = int(sys.argv[3])

def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)

nodes = re.findall(r"^JOB\\s+(\\S+)\\s+", dag, re.M)
vars_ = dict(re.findall(r'^VARS\\s+(\\S+).*?request_cpus="([^"]+)"', dag, re.M))
full = [node for node in nodes if node.startswith("FULL_")]

if len(full) != 60:
    fail(f"expected 60 pulsar jobs, found {len(full)}")
if "MERGE_STEP2" not in nodes:
    fail("DAG missing MERGE_STEP2")
if step2_chunks > 1 and "STEP2_ALL" in nodes:
    fail("DAG still contains STEP2_ALL despite split Step2")

expected_step2 = [f"STEP2_CHUNK{i:02d}" for i in range(1, step2_chunks + 1)]
if step2_chunks == 1:
    expected_step2 = ["STEP2_ALL"]
for node in expected_step2:
    if node not in nodes:
        fail(f"DAG missing {node}")
    if vars_.get(node) != step2_jobs:
        fail(f"{node} request_cpus={vars_.get(node)}, expected {step2_jobs}")
if vars_.get("MERGE_STEP2") != "1":
    fail(f"MERGE_STEP2 request_cpus={vars_.get('MERGE_STEP2')}, expected 1")
if "step6_apply_delete" in dag or 'fix_qc_action = "delete"' in dag:
    fail("DAG contains delete-stage text")

print("OK: DAG shape and CPU requests are correct")
print(f"OK: pulsar jobs: {len(full)}")
print(f"OK: request_cpus counts: {dict(Counter(vars_.values()))}")
PY

        if command -v condor_status >/dev/null 2>&1; then
          slots="$(
            condor_status -af Name Cpus State Activity 2>/dev/null |
              awk -v need="$step2_jobs" '$2 >= need { n++ } END { print n+0 }'
          )"
          if (( slots < step2_chunks )); then
            condor_status -af Name Cpus State Activity 2>/dev/null || true
            fail "only $slots slots have >= $step2_jobs cores; need $step2_chunks"
          fi
          ok "HTCondor has $slots slots with >= $step2_jobs cores"
        else
          warn "condor_status not found; skipped cluster capacity check"
        fi

        df -h "$B" "$C" 2>/dev/null || true
        ok "preflight passed"
        """,
    )
    write_executable(
        bundle / "launch_htcondor.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail

        B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        C="$B/campaign"
        DAG="$C/EPTA-DR3/epta-dr3-data/_campaign/dag/pleb_campaign.dag"
        if [[ "${PLEB_SKIP_PREFLIGHT:-0}" != "1" ]]; then
          "$B/preflight_bundle.sh"
        else
          "$B/fix_paths_and_render.sh"
        fi
        cd "$C"
        condor_submit_dag "$DAG"
        """,
    )
    write_executable(
        bundle / "monitor_htcondor.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        C="$B/campaign"
        DAG="$C/EPTA-DR3/epta-dr3-data/_campaign/dag/pleb_campaign.dag"
        [[ -f "$DAG" ]] || { echo "DAG not found: $DAG" >&2; exit 2; }
        condor_q -af ClusterId ProcId JobStatus DAGNodeName RemoteHost HoldReason 2>/dev/null || true
        echo
        ls -1t "$C"/EPTA-DR3/epta-dr3-data/_campaign/dag/*.dagman.out 2>/dev/null | head -1 | xargs -r tail -40
        """,
    )
    write_text(
        bundle / "README_HTCONDOR.txt",
        """
        Run:
          PLEB_OUTER_SIF=/path/to/pleb_tempo2.sif ./launch_htcondor.sh

        Preflight only:
          PLEB_OUTER_SIF=/path/to/pleb_tempo2.sif ./preflight_bundle.sh

        Skip launch preflight only if you have already run it:
          PLEB_SKIP_PREFLIGHT=1 PLEB_OUTER_SIF=/path/to/pleb_tempo2.sif ./launch_htcondor.sh

        Fill all available slots:
          Leave PLEB_DAG_MAXJOBS unset, or set it to the number of slots before
          rendering/submitting:
            PLEB_DAG_MAXJOBS=$(condor_status -af Name | wc -l) PLEB_OUTER_SIF=/path/to/pleb_tempo2.sif ./launch_htcondor.sh

        Monitor:
          ./monitor_htcondor.sh

        HTCondor uses file transfer.  The SIF is transferred per job by DAGMan.
        """,
    )


def shared_stage_script() -> str:
    return """
    #!/usr/bin/env bash
    set -euo pipefail

    B="${PLEB_BUNDLE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
    C="$B/campaign"
    CONTROL="$C/EPTA-DR3/epta-dr3-data/_campaign"
    MANIFEST="$CONTROL/manifests/pulsars.tsv"
    SIF="${PLEB_SIF:-${PLEB_OUTER_SIF:-}}"
    RUN_ROOT="${PLEB_RUN_ROOT:-$C/.shared_runs}"
    LOG_DIR="$CONTROL/shared_logs"
    STATUS_DIR="$CONTROL/shared_status"
    FINAL_STAGE="${PLEB_FINAL_STAGE:-step6_apply_delete}"
    EMPTY="__PLEB_EMPTY__"

    stage="${1:?stage}"
    config_rel="${2:-$EMPTY}"
    pulsar="${3:-$EMPTY}"
    n_trials="${4:-$EMPTY}"
    state_in_rel="${5:?state_in_rel}"
    state_out_rel="${6:-$EMPTY}"
    result_rel="${7:-$EMPTY}"
    merge_inputs="${8:-$EMPTY}"
    job_name="${9:-$stage}"

    if [[ -z "$SIF" ]]; then
      if [[ -f "$B/pleb_tempo2.sif" ]]; then SIF="$B/pleb_tempo2.sif"
      elif [[ -f "$HOME/containers/pleb_tempo2.sif" ]]; then SIF="$HOME/containers/pleb_tempo2.sif"
      else echo "Set PLEB_SIF=/path/to/pleb_tempo2.sif" >&2; exit 2
      fi
    fi
    [[ -f "$SIF" ]] || { echo "Singularity image not found: $SIF" >&2; exit 2; }

    mkdir -p "$CONTROL/config" "$LOG_DIR" "$STATUS_DIR" "$RUN_ROOT"
    printf '%s\n' "$C" > "$C/.campaign_repo_root"
    printf '%s\n' "$C/EPTA-DR3/epta-dr3-data" > "$C/.campaign_dataset_root"
    printf '%s\n' "$CONTROL" > "$C/.campaign_control_root"
    printf '%s\n' "$SIF" > "$CONTROL/config/sif_path.txt"

    slugify() {
      local value="$1"
      value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
      value="${value//+/p}"
      value="${value//-/m}"
      value="${value//./_}"
      printf '%s' "$value" | sed 's/[^a-z0-9_]/_/g; s/_\\+/_/g; s/^_//; s/_$//'
    }

    if [[ "$merge_inputs" == "__PLEB_ALL_PULSARS__" ]]; then
      merge_inputs="$(
        awk 'NF && $1 !~ /^#/ {
          p=$1; gsub(/\\+/, "p", p); gsub(/-/, "m", p); p=tolower(p); gsub(/[^a-z0-9_]/, "_", p);
          printf "%spulsar_%s.tar.gz", sep, p; sep=","
        }' "$MANIFEST"
      )"
    fi

    job_dir="$RUN_ROOT/$job_name"
    status_file="$STATUS_DIR/$job_name.status"
    state_name="$(basename "$state_in_rel")"
    rm -rf "$job_dir"
    mkdir -p "$job_dir"

    status_write() {
      local status="$1"
      local rc="${2:-}"
      {
        printf 'job=%s\n' "$job_name"
        printf 'stage=%s\n' "$stage"
        printf 'pulsar=%s\n' "$pulsar"
        printf 'status=%s\n' "$status"
        [[ -n "$rc" ]] && printf 'exit_code=%s\n' "$rc"
        printf 'time=%s\n' "$(date -Is)"
        printf 'stdout=%s\n' "$LOG_DIR/$job_name.out"
        printf 'stderr=%s\n' "$LOG_DIR/$job_name.err"
      } > "$status_file"
    }

    status_write STAGING
    cp -a "$C/scaffold/scripts/run_campaign_stage.sh" "$job_dir/run_campaign_stage.sh"
    ln -s "$SIF" "$job_dir/$(basename "$SIF")"
    cp -a "$C/$state_in_rel" "$job_dir/$state_name"

    if [[ -n "$merge_inputs" && "$merge_inputs" != "$EMPTY" ]]; then
      IFS=',' read -r -a inputs <<< "$merge_inputs"
      for item in "${inputs[@]}"; do
        [[ -z "$item" ]] && continue
        cp -a "$C/$item" "$job_dir/$(basename "$item")"
      done
    fi

    status_write RUNNING
    set +e
    (
      cd "$job_dir"
      PLEB_FORCE_SINGULARITY=1 bash ./run_campaign_stage.sh \
        "$stage" "$config_rel" "$pulsar" "$n_trials" "$FINAL_STAGE" \
        "$state_name" "$state_out_rel" "$result_rel" "$merge_inputs" "$(basename "$SIF")"
    ) >"$LOG_DIR/$job_name.out" 2>"$LOG_DIR/$job_name.err"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      status_write FAILED "$rc"
      exit "$rc"
    fi

    if [[ -n "$state_out_rel" && "$state_out_rel" != "$EMPTY" ]]; then
      rm -rf "$C/$state_out_rel"
      mv "$job_dir/$state_out_rel" "$C/$state_out_rel"
    fi
    if [[ -n "$result_rel" && "$result_rel" != "$EMPTY" ]]; then
      mv "$job_dir/$result_rel" "$C/$result_rel"
    fi
    status_write DONE
    """


def monitor_script() -> str:
    return """
    #!/usr/bin/env bash
    set -euo pipefail
    B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    C="$B/campaign"
    CONTROL="$C/EPTA-DR3/epta-dr3-data/_campaign"
    MANIFEST="$CONTROL/manifests/pulsars.tsv"
    STATUS_DIR="${PLEB_STATUS_DIR:-$CONTROL/shared_status}"
    LOG_DIR="${PLEB_LOG_DIR:-$CONTROL/shared_logs}"
    INTERVAL="${PLEB_MONITOR_INTERVAL:-0}"

    slugify() {
      local value="$1"
      value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
      value="${value//+/p}"; value="${value//-/m}"; value="${value//./_}"
      printf '%s' "$value" | sed 's/[^a-z0-9_]/_/g; s/_\\+/_/g; s/^_//; s/_$//'
    }
    val() { [[ -f "$1" ]] && awk -F= -v k="$2" '$1==k{sub(/^[^=]*=/,""); print; exit}' "$1"; }
    infer() {
      local job="$1" artifact="$2" s
      s="$(val "$STATUS_DIR/$job.status" status || true)"
      [[ -n "$s" ]] && { printf '%s' "$s"; return; }
      [[ -n "$artifact" && -e "$artifact" ]] && { printf DONE; return; }
      printf PENDING
    }
    lastlog() {
      local job="$1"
      if [[ -s "$LOG_DIR/$job.err" ]]; then tail -1 "$LOG_DIR/$job.err" | tr '\\t' ' ' | cut -c1-90
      elif [[ -s "$LOG_DIR/$job.out" ]]; then tail -1 "$LOG_DIR/$job.out" | tr '\\t' ' ' | cut -c1-90
      fi
    }
    print_once() {
      mkdir -p "$STATUS_DIR" "$LOG_DIR"
      printf 'PLEB bundle status at %s\n\n' "$(date -Is)"
      printf '%-28s %-14s %-10s %-22s %s\n' JOB STAGE STATUS PULSAR LAST_LOG
      printf '%-28s %-14s %-10s %-22s %s\n' ---------------------------- -------------- ---------- ---------------------- ----------------
      rows=(
        "ingest|ingest|$(infer ingest "$C/repo_state_raw_ingest")|-"
        "step1_all|pipeline|$(infer step1_all "$C/repo_state_step1")|-"
        "step2_all|pipeline|$(infer step2_all "$C/repo_state_step2")|-"
      )
      if [[ -f "$MANIFEST" ]]; then
        while read -r pulsar _; do
          [[ -z "${pulsar:-}" || "$pulsar" == \\#* ]] && continue
          slug="$(slugify "$pulsar")"
          rows+=("pulsar_${slug}|pulsar-full|$(infer "pulsar_${slug}" "$C/pulsar_${slug}.tar.gz")|$pulsar")
        done < "$MANIFEST"
      fi
      rows+=("merge_final|merge-pulsars|$(infer merge_final "$C/repo_state_final")|-")
      rows+=("release|release|$(infer release "$C/release_epta_dr3_pleb.tar.gz")|-")
      done_n=0; run_n=0; fail_n=0; pend_n=0; total=0
      for row in "${rows[@]}"; do
        IFS='|' read -r job stage status pulsar <<< "$row"
        total=$((total+1))
        case "$status" in DONE) done_n=$((done_n+1));; RUNNING|STAGING) run_n=$((run_n+1));; FAILED) fail_n=$((fail_n+1));; *) pend_n=$((pend_n+1));; esac
        printf '%-28s %-14s %-10s %-22s %s\n' "$job" "$stage" "$status" "$pulsar" "$(lastlog "$job")"
      done
      printf '\nTOTAL=%d DONE=%d ACTIVE=%d FAILED=%d PENDING=%d\n' "$total" "$done_n" "$run_n" "$fail_n" "$pend_n"
    }
    if [[ "$INTERVAL" != 0 ]]; then while true; do clear 2>/dev/null || true; print_once; sleep "$INTERVAL"; done; else print_once; fi
    """


def workstation_scripts(bundle: Path) -> None:
    write_executable(bundle / "run_shared_stage.sh", shared_stage_script())
    write_executable(bundle / "monitor_jobs.sh", monitor_script())
    write_executable(
        bundle / "run_workstation_singularity.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        C="$B/campaign"
        MANIFEST="$C/EPTA-DR3/epta-dr3-data/_campaign/manifests/pulsars.tsv"
        JOBS="${PLEB_WORKSTATION_JOBS:-$(command -v nproc >/dev/null && nproc || echo 1)}"
        export PLEB_BUNDLE_ROOT="$B"
        export PLEB_STATUS_DIR="$C/EPTA-DR3/epta-dr3-data/_campaign/shared_status"
        export PLEB_LOG_DIR="$C/EPTA-DR3/epta-dr3-data/_campaign/shared_logs"

        slugify() {
          local value="$1"
          value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
          value="${value//+/p}"; value="${value//-/m}"; value="${value//./_}"
          printf '%s' "$value" | sed 's/[^a-z0-9_]/_/g; s/_\\+/_/g; s/^_//; s/_$//'
        }
        run_pulsar() {
          local pulsar="$1" n_trials="$2" slug
          slug="$(slugify "$pulsar")"
          bash "$B/run_shared_stage.sh" pulsar-full __PLEB_EMPTY__ "$pulsar" "$n_trials" repo_state_step2 __PLEB_EMPTY__ "pulsar_${slug}.tar.gz" __PLEB_EMPTY__ "pulsar_${slug}"
        }
        export -f slugify run_pulsar
        export B

        bash "$B/run_shared_stage.sh" ingest configs/htcondor_campaign/epta_dr_optimize/ingest_from_raw_sources.toml __PLEB_EMPTY__ __PLEB_EMPTY__ repo_state_seed repo_state_raw_ingest __PLEB_EMPTY__ __PLEB_EMPTY__ ingest
        bash "$B/run_shared_stage.sh" pipeline configs/htcondor_campaign/epta_dr_optimize/step1_fix_all.toml __PLEB_EMPTY__ __PLEB_EMPTY__ repo_state_raw_ingest repo_state_step1 __PLEB_EMPTY__ __PLEB_EMPTY__ step1_all
        bash "$B/run_shared_stage.sh" pipeline configs/htcondor_campaign/epta_dr_optimize/step2_detect_variants_all.toml __PLEB_EMPTY__ __PLEB_EMPTY__ repo_state_step1 repo_state_step2 __PLEB_EMPTY__ __PLEB_EMPTY__ step2_all
        awk 'NF && $1 !~ /^#/ {print $1, $2}' "$MANIFEST" | xargs -n 2 -P "$JOBS" bash -c 'run_pulsar "$0" "$1"'
        bash "$B/run_shared_stage.sh" merge-pulsars __PLEB_EMPTY__ __PLEB_EMPTY__ __PLEB_EMPTY__ repo_state_step2 repo_state_final __PLEB_EMPTY__ __PLEB_ALL_PULSARS__ merge_final
        bash "$B/run_shared_stage.sh" release __PLEB_EMPTY__ __PLEB_EMPTY__ __PLEB_EMPTY__ repo_state_final __PLEB_EMPTY__ release_epta_dr3_pleb.tar.gz __PLEB_EMPTY__ release
        echo "Release: $C/release_epta_dr3_pleb.tar.gz"
        """,
    )
    write_text(
        bundle / "README_WORKSTATION.txt",
        """
        Run:
          PLEB_SIF=/path/to/pleb_tempo2.sif PLEB_WORKSTATION_JOBS=8 bash ./run_workstation_singularity.sh

        Monitor once:
          bash ./monitor_jobs.sh

        Live monitor:
          PLEB_MONITOR_INTERVAL=10 bash ./monitor_jobs.sh
        """,
    )


def slurm_scripts(bundle: Path) -> None:
    write_executable(bundle / "run_shared_stage.sh", shared_stage_script())
    write_executable(bundle / "monitor_jobs.sh", monitor_script())
    write_executable(
        bundle / "slurm_stage.sbatch",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        B="${PLEB_BUNDLE_ROOT:?PLEB_BUNDLE_ROOT required}"
        bash "$B/run_shared_stage.sh" "$PLEB_STAGE" "$PLEB_CONFIG_REL" "$PLEB_PULSAR" "$PLEB_N_TRIALS" "$PLEB_STATE_IN" "$PLEB_STATE_OUT" "$PLEB_RESULT_REL" "$PLEB_MERGE_INPUTS" "$PLEB_JOB_NAME"
        """,
    )
    write_executable(
        bundle / "slurm_pulsar_array.sbatch",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        B="${PLEB_BUNDLE_ROOT:?PLEB_BUNDLE_ROOT required}"
        MANIFEST="$B/campaign/EPTA-DR3/epta-dr3-data/_campaign/manifests/pulsars.tsv"
        line="$(awk 'NF && $1 !~ /^#/ {print $1, $2}' "$MANIFEST" | sed -n "${SLURM_ARRAY_TASK_ID}p")"
        pulsar="$(awk '{print $1}' <<< "$line")"
        n_trials="$(awk '{print $2}' <<< "$line")"
        slug="$(printf '%s' "$pulsar" | tr '[:upper:]' '[:lower:]' | sed 's/+ /p/g; s/+ /p/g')"
        slug="${pulsar,,}"; slug="${slug//+/p}"; slug="${slug//-/m}"; slug="${slug//./_}"; slug="$(printf '%s' "$slug" | sed 's/[^a-z0-9_]/_/g')"
        bash "$B/run_shared_stage.sh" pulsar-full __PLEB_EMPTY__ "$pulsar" "$n_trials" repo_state_step2 __PLEB_EMPTY__ "pulsar_${slug}.tar.gz" __PLEB_EMPTY__ "pulsar_${slug}"
        """,
    )
    write_executable(
        bundle / "submit_slurm.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        MANIFEST="$B/campaign/EPTA-DR3/epta-dr3-data/_campaign/manifests/pulsars.tsv"
        export PLEB_BUNDLE_ROOT="$B"
        cpus="${PLEB_SLURM_CPUS:-1}"
        mem="${PLEB_SLURM_MEM:-8G}"
        time="${PLEB_SLURM_TIME:-24:00:00}"
        limit="${PLEB_SLURM_ARRAY_LIMIT:-${PLEB_WORKSTATION_JOBS:-4}}"
        count="$(awk 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$MANIFEST")"
        common=(--cpus-per-task="$cpus" --mem="$mem" --time="$time" --export=ALL,PLEB_BUNDLE_ROOT="$B")
        jid1="$(sbatch --parsable "${common[@]}" --job-name=pleb_ingest --export=ALL,PLEB_BUNDLE_ROOT="$B",PLEB_STAGE=ingest,PLEB_CONFIG_REL=configs/htcondor_campaign/epta_dr_optimize/ingest_from_raw_sources.toml,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_seed,PLEB_STATE_OUT=repo_state_raw_ingest,PLEB_RESULT_REL=__PLEB_EMPTY__,PLEB_MERGE_INPUTS=__PLEB_EMPTY__,PLEB_JOB_NAME=ingest "$B/slurm_stage.sbatch")"
        jid2="$(sbatch --parsable "${common[@]}" --dependency=afterok:"$jid1" --job-name=pleb_step1 --export=ALL,PLEB_BUNDLE_ROOT="$B",PLEB_STAGE=pipeline,PLEB_CONFIG_REL=configs/htcondor_campaign/epta_dr_optimize/step1_fix_all.toml,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_raw_ingest,PLEB_STATE_OUT=repo_state_step1,PLEB_RESULT_REL=__PLEB_EMPTY__,PLEB_MERGE_INPUTS=__PLEB_EMPTY__,PLEB_JOB_NAME=step1_all "$B/slurm_stage.sbatch")"
        jid3="$(sbatch --parsable "${common[@]}" --dependency=afterok:"$jid2" --job-name=pleb_step2 --export=ALL,PLEB_BUNDLE_ROOT="$B",PLEB_STAGE=pipeline,PLEB_CONFIG_REL=configs/htcondor_campaign/epta_dr_optimize/step2_detect_variants_all.toml,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_step1,PLEB_STATE_OUT=repo_state_step2,PLEB_RESULT_REL=__PLEB_EMPTY__,PLEB_MERGE_INPUTS=__PLEB_EMPTY__,PLEB_JOB_NAME=step2_all "$B/slurm_stage.sbatch")"
        jid4="$(sbatch --parsable "${common[@]}" --dependency=afterok:"$jid3" --array=1-"$count"%"$limit" --job-name=pleb_pulsars "$B/slurm_pulsar_array.sbatch")"
        jid5="$(sbatch --parsable "${common[@]}" --dependency=afterok:"$jid4" --job-name=pleb_merge --export=ALL,PLEB_BUNDLE_ROOT="$B",PLEB_STAGE=merge-pulsars,PLEB_CONFIG_REL=__PLEB_EMPTY__,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_step2,PLEB_STATE_OUT=repo_state_final,PLEB_RESULT_REL=__PLEB_EMPTY__,PLEB_MERGE_INPUTS=__PLEB_ALL_PULSARS__,PLEB_JOB_NAME=merge_final "$B/slurm_stage.sbatch")"
        jid6="$(sbatch --parsable "${common[@]}" --dependency=afterok:"$jid5" --job-name=pleb_release --export=ALL,PLEB_BUNDLE_ROOT="$B",PLEB_STAGE=release,PLEB_CONFIG_REL=__PLEB_EMPTY__,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_final,PLEB_STATE_OUT=__PLEB_EMPTY__,PLEB_RESULT_REL=release_epta_dr3_pleb.tar.gz,PLEB_MERGE_INPUTS=__PLEB_EMPTY__,PLEB_JOB_NAME=release "$B/slurm_stage.sbatch")"
        printf 'Submitted Slurm chain: %s %s %s %s %s %s\n' "$jid1" "$jid2" "$jid3" "$jid4" "$jid5" "$jid6"
        """,
    )
    write_text(
        bundle / "README_SLURM.txt",
        """
        Run:
          PLEB_SIF=/path/to/pleb_tempo2.sif PLEB_SLURM_ARRAY_LIMIT=8 bash ./submit_slurm.sh

        Monitor:
          bash ./monitor_jobs.sh
          squeue -u "$USER"
        """,
    )


def pbs_scripts(bundle: Path) -> None:
    write_executable(bundle / "run_shared_stage.sh", shared_stage_script())
    write_executable(bundle / "monitor_jobs.sh", monitor_script())
    write_executable(
        bundle / "pbs_stage.pbs",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        B="${PLEB_BUNDLE_ROOT:?PLEB_BUNDLE_ROOT required}"
        bash "$B/run_shared_stage.sh" "$PLEB_STAGE" "$PLEB_CONFIG_REL" "$PLEB_PULSAR" "$PLEB_N_TRIALS" "$PLEB_STATE_IN" "$PLEB_STATE_OUT" "$PLEB_RESULT_REL" "$PLEB_MERGE_INPUTS" "$PLEB_JOB_NAME"
        """,
    )
    write_executable(
        bundle / "pbs_pulsar_array.pbs",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        B="${PLEB_BUNDLE_ROOT:?PLEB_BUNDLE_ROOT required}"
        MANIFEST="$B/campaign/EPTA-DR3/epta-dr3-data/_campaign/manifests/pulsars.tsv"
        idx="${PBS_ARRAY_INDEX:-${PBS_ARRAYID:-1}}"
        line="$(awk 'NF && $1 !~ /^#/ {print $1, $2}' "$MANIFEST" | sed -n "${idx}p")"
        pulsar="$(awk '{print $1}' <<< "$line")"
        n_trials="$(awk '{print $2}' <<< "$line")"
        slug="${pulsar,,}"; slug="${slug//+/p}"; slug="${slug//-/m}"; slug="${slug//./_}"; slug="$(printf '%s' "$slug" | sed 's/[^a-z0-9_]/_/g')"
        bash "$B/run_shared_stage.sh" pulsar-full __PLEB_EMPTY__ "$pulsar" "$n_trials" repo_state_step2 __PLEB_EMPTY__ "pulsar_${slug}.tar.gz" __PLEB_EMPTY__ "pulsar_${slug}"
        """,
    )
    write_executable(
        bundle / "submit_pbs.sh",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        MANIFEST="$B/campaign/EPTA-DR3/epta-dr3-data/_campaign/manifests/pulsars.tsv"
        count="$(awk 'NF && $1 !~ /^#/ {n++} END {print n+0}' "$MANIFEST")"
        ncpus="${PLEB_PBS_NCPUS:-1}"
        mem="${PLEB_PBS_MEM:-8gb}"
        walltime="${PLEB_PBS_WALLTIME:-24:00:00}"
        limit="${PLEB_PBS_ARRAY_LIMIT:-${PLEB_WORKSTATION_JOBS:-4}}"
        res="select=1:ncpus=${ncpus}:mem=${mem},walltime=${walltime}"
        qvars_base="PLEB_BUNDLE_ROOT=$B"
        jid1="$(qsub -N pleb_ingest -l "$res" -v "$qvars_base,PLEB_STAGE=ingest,PLEB_CONFIG_REL=configs/htcondor_campaign/epta_dr_optimize/ingest_from_raw_sources.toml,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_seed,PLEB_STATE_OUT=repo_state_raw_ingest,PLEB_RESULT_REL=__PLEB_EMPTY__,PLEB_MERGE_INPUTS=__PLEB_EMPTY__,PLEB_JOB_NAME=ingest" "$B/pbs_stage.pbs")"
        jid2="$(qsub -N pleb_step1 -l "$res" -W depend=afterok:"$jid1" -v "$qvars_base,PLEB_STAGE=pipeline,PLEB_CONFIG_REL=configs/htcondor_campaign/epta_dr_optimize/step1_fix_all.toml,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_raw_ingest,PLEB_STATE_OUT=repo_state_step1,PLEB_RESULT_REL=__PLEB_EMPTY__,PLEB_MERGE_INPUTS=__PLEB_EMPTY__,PLEB_JOB_NAME=step1_all" "$B/pbs_stage.pbs")"
        jid3="$(qsub -N pleb_step2 -l "$res" -W depend=afterok:"$jid2" -v "$qvars_base,PLEB_STAGE=pipeline,PLEB_CONFIG_REL=configs/htcondor_campaign/epta_dr_optimize/step2_detect_variants_all.toml,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_step1,PLEB_STATE_OUT=repo_state_step2,PLEB_RESULT_REL=__PLEB_EMPTY__,PLEB_MERGE_INPUTS=__PLEB_EMPTY__,PLEB_JOB_NAME=step2_all" "$B/pbs_stage.pbs")"
        jid4="$(qsub -N pleb_pulsars -l "$res" -W depend=afterok:"$jid3" -J 1-"$count"%"$limit" -v "$qvars_base" "$B/pbs_pulsar_array.pbs")"
        jid5="$(qsub -N pleb_merge -l "$res" -W depend=afterok:"$jid4" -v "$qvars_base,PLEB_STAGE=merge-pulsars,PLEB_CONFIG_REL=__PLEB_EMPTY__,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_step2,PLEB_STATE_OUT=repo_state_final,PLEB_RESULT_REL=__PLEB_EMPTY__,PLEB_MERGE_INPUTS=__PLEB_ALL_PULSARS__,PLEB_JOB_NAME=merge_final" "$B/pbs_stage.pbs")"
        jid6="$(qsub -N pleb_release -l "$res" -W depend=afterok:"$jid5" -v "$qvars_base,PLEB_STAGE=release,PLEB_CONFIG_REL=__PLEB_EMPTY__,PLEB_PULSAR=__PLEB_EMPTY__,PLEB_N_TRIALS=__PLEB_EMPTY__,PLEB_STATE_IN=repo_state_final,PLEB_STATE_OUT=__PLEB_EMPTY__,PLEB_RESULT_REL=release_epta_dr3_pleb.tar.gz,PLEB_MERGE_INPUTS=__PLEB_EMPTY__,PLEB_JOB_NAME=release" "$B/pbs_stage.pbs")"
        printf 'Submitted PBS chain: %s %s %s %s %s %s\n' "$jid1" "$jid2" "$jid3" "$jid4" "$jid5" "$jid6"
        """,
    )
    write_text(
        bundle / "README_PBS.txt",
        """
        Run:
          PLEB_SIF=/path/to/pleb_tempo2.sif PLEB_PBS_ARRAY_LIMIT=8 bash ./submit_pbs.sh

        Monitor:
          bash ./monitor_jobs.sh
          qstat -u "$USER"

        This uses PBS Pro array syntax (-J).  If your site uses Torque (-t),
        adjust submit_pbs.sh accordingly.
        """,
    )


def add_arch_files(arch: str, bundle: Path, name: str, n_trials: int) -> None:
    write_text(bundle / "README_RUN.txt", common_readme(arch, name, n_trials))
    if arch == "htcondor":
        htcondor_scripts(bundle)
    elif arch == "pbs":
        pbs_scripts(bundle)
    elif arch == "slurm":
        slurm_scripts(bundle)
    elif arch == "workstation":
        workstation_scripts(bundle)
    else:
        raise AssertionError(arch)


def tar_bundle(bundle: Path, out_dir: Path) -> Path:
    tar_path = out_dir / f"{bundle.name}.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    run(
        [
            "tar",
            "--exclude=*/.git/objects/pack/tmp_pack_*",
            "-czf",
            str(tar_path),
            "-C",
            str(out_dir),
            bundle.name,
        ]
    )
    run(["gzip", "-t", str(tar_path)])
    return tar_path


def build_one(args: argparse.Namespace, arch: str) -> Path:
    repo = args.repo.resolve()
    date = datetime.now().strftime("%Y%m%d")
    name = args.name or f"epta_ingest_to_release_{arch}_ntrials{args.n_trials}_{date}"
    if args.arch == "all" and args.name:
        name = f"{args.name}_{arch}"
    bundle = args.out_dir.resolve() / name
    ensure_clean_dir(bundle, force=args.force)
    prepare_campaign(repo, bundle, args.n_trials, args.sif_path)
    add_arch_files(arch, bundle, name, args.n_trials)
    tar_path = tar_bundle(bundle, args.out_dir.resolve())
    print(f"{arch}: {tar_path}")
    return tar_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=(*ARCHES, "all"), required=True)
    p.add_argument("--repo", type=Path, default=repo_root_from_script())
    p.add_argument("--out-dir", type=Path, default=repo_root_from_script() / "bundles")
    p.add_argument(
        "--name",
        default=None,
        help="Bundle directory name; with --arch all, used as a prefix.",
    )
    p.add_argument("--n-trials", type=int, default=6)
    p.add_argument(
        "--sif-path",
        default=None,
        help="Optional default SIF path written into generated config.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing generated bundle directory.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arches = ARCHES if args.arch == "all" else (args.arch,)
    for arch in arches:
        build_one(args, arch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
