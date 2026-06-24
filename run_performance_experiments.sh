#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CONFIG_FILE="${WALL_TIME_CONFIG:-${ROOT_DIR}/configs/performance_double_integrator.yaml}"
TRIAL_TIMEOUT_SECONDS="${TRIAL_TIMEOUT_SECONDS:-3600}"
NUM_RUNS="${NUM_RUNS:-10}"
RUN_NUMBER=""
METHOD="${METHOD:-both}"
PLANNERS=("aorrt" "aoest" "sststar")
SKIP_EXISTING=1
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./run_performance_experiments.sh [options] [wall_time_experiment args...]

Runs the double-integrator wall-time comparison:
  experiments/wall_time_experiment.py
  configs/performance_double_integrator.yaml

Options handled by this wrapper:
  --config FILE          Config path
  --planner-name NAME    aorrt, aoest, sststar, or all (default: all)
  --method METHOD        aura, replanning, or both (default: both)
  --num-runs N           Run numbers 1..N (default: 10)
  --run-number N         Run one numbered repeat
  --overwrite            Do not add --skip-existing
  -h, --help             Show this help

Other options are passed through to wall_time_experiment.py.

Environment:
  PYTHON_BIN             Python executable (default: /usr/bin/python3)
  WALL_TIME_CONFIG       Config path override
  TRIAL_TIMEOUT_SECONDS  Timeout per planner batch (default: 3600)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --planner-name)
            if [[ "$2" == "all" ]]; then
                PLANNERS=("aorrt" "aoest" "sststar")
            else
                PLANNERS=("$2")
            fi
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            RUN_NUMBER=""
            shift 2
            ;;
        --run-number)
            RUN_NUMBER="$2"
            shift 2
            ;;
        --overwrite)
            SKIP_EXISTING=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "Wall-time experiment"
echo "  script:   ${ROOT_DIR}/experiments/wall_time_experiment.py"
echo "  config:   ${CONFIG_FILE}"
echo "  planners: ${PLANNERS[*]}"
echo "  method:   ${METHOD}"
if [[ -n "${RUN_NUMBER}" ]]; then
    echo "  run:      ${RUN_NUMBER}"
else
    echo "  runs:     1..${NUM_RUNS}"
fi
echo "  timeout:  ${TRIAL_TIMEOUT_SECONDS}s per planner"
echo

failures=0
for planner in "${PLANNERS[@]}"; do
    cmd=(
        "${PYTHON_BIN}" "${ROOT_DIR}/experiments/wall_time_experiment.py"
        --config "${CONFIG_FILE}"
        --planner-name "${planner}"
        --method "${METHOD}"
    )
    if [[ -n "${RUN_NUMBER}" ]]; then
        cmd+=(--run-number "${RUN_NUMBER}")
    else
        cmd+=(--num-runs "${NUM_RUNS}")
    fi
    if [[ "${SKIP_EXISTING}" -eq 1 ]]; then
        cmd+=(--skip-existing)
    fi
    cmd+=("${EXTRA_ARGS[@]}")

    echo "== ${planner} =="
    if ! timeout "${TRIAL_TIMEOUT_SECONDS}" "${cmd[@]}"; then
        failures=$((failures + 1))
        echo "[ERROR] ${planner} failed"
    fi
    echo
done

if (( failures > 0 )); then
    echo "Completed with ${failures} failed planner batch(es)."
    exit 1
fi

echo "All requested wall-time runs completed."
