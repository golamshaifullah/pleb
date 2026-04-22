#!/usr/bin/env bash
set -euo pipefail

# Show the top-level pleb workflow plus any active per-pulsar PQC workers.
# Optionally pass a run label substring, e.g.:
#   scripts/show_active_pqc_workers.sh step2_pqc_balanced_detect_v1_3

run_label="${1:-}"
shell_pid="$$"
parent_pid="${PPID:-}"

perl_script='
  next if /show_active_pqc_workers\.sh/;
  next if /\brg\b/;
  if (/^\s*(\d+)\s+(\d+)\s+(.*)$/) {
    my ($pid, $elapsed, $cmd) = ($1, $2, $3);
    if ($cmd =~ /run_pqc_for_parfile/ && $cmd =~ /\.pqc_(J\d{4}[+-]\d{4})\.json/) {
      printf "%s\tpid=%s\telapsed=%s\n", $1, $pid, $elapsed;
    } elsif ($cmd =~ m{/bin/pleb workflow --file }) {
      printf "WORKFLOW\tpid=%s\telapsed=%s\n", $pid, $elapsed;
    }
  }
'

output="$(ps -eo pid,etimes,cmd --cols 500 | perl -ne "${perl_script}")"
output="$(printf '%s\n' "${output}" | grep -vE "^.*pid=(${shell_pid}|${parent_pid})\b" || true)"

if [[ -n "${run_label}" ]]; then
  output="$(printf '%s\n' "${output}" | grep -E "^WORKFLOW|${run_label}|J[0-9]{4}[+-][0-9]{4}" || true)"
fi

if [[ -z "${output}" ]]; then
  echo "No visible pleb workflow or PQC worker processes."
  exit 0
fi

printf '%s\n' "${output}"
