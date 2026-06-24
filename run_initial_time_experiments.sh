#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CONFIG_FILE="${INITIAL_TIME_CONFIG:-${ROOT_DIR}/configs/initial_time_experiment.yaml}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-86400}"

usage() {
    cat <<'EOF'
Usage: ./run_initial_time_experiments.sh [initial_time_experiment args...]

Runs the initial-planning-time sweep using:
  experiments/initial_time_experiment.py
  configs/initial_time_experiment.yaml

Defaults added by this wrapper:
  --fill-missing-replays --workspace-replay --no-show

Examples:
  ./run_initial_time_experiments.sh
  ./run_initial_time_experiments.sh --planner-name aoest --num-runs 3
  ./run_initial_time_experiments.sh --control-durations 0.5 1.0 --planning-times 4 8

Environment:
  PYTHON_BIN              Python executable (default: /usr/bin/python3)
  INITIAL_TIME_CONFIG     Config path override
  RUN_TIMEOUT_SECONDS     Whole-run timeout (default: 86400)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

echo "Initial-time experiment"
echo "  script:  ${ROOT_DIR}/experiments/initial_time_experiment.py"
echo "  config:  ${CONFIG_FILE}"
echo "  timeout: ${RUN_TIMEOUT_SECONDS}s"
echo

cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/experiments/initial_time_experiment.py" \
    --config "${CONFIG_FILE}" \
    --fill-missing-replays \
    --workspace-replay \
    --no-show \
    "$@"
)

timeout "${RUN_TIMEOUT_SECONDS}" "${cmd[@]}"

missing_file="$(mktemp "${TMPDIR:-/tmp}/aura_initial_missing.XXXXXX.tsv")"
trap 'rm -f "${missing_file}"' EXIT

"${cmd[@]}" --dry-run-missing --missing-replays-file "${missing_file}" >/dev/null
missing_count=0
if [[ -f "${missing_file}" ]]; then
    while IFS= read -r _line; do
        missing_count=$((missing_count + 1))
    done < "${missing_file}"
fi

if (( missing_count > 0 )); then
    echo "[ERROR] Initial-time run finished with ${missing_count} missing replay slot(s)."
    exit 1
fi

echo "Initial-time replay grid complete."
