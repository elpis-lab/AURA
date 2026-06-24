#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
TRIAL_TIMEOUT_SECONDS="${TRIAL_TIMEOUT_SECONDS:-3600}"
SYSTEM_NAME="${ERROR_SYSTEM_NAME:-kinematic_car}"
ENVIRONMENT_NAME="${ERROR_ENVIRONMENT_NAME:-gaussian}"
NUM_CONTROLS="${ERROR_NUM_CONTROLS:-5}"
NUM_TRIALS="${ERROR_NUM_TRIALS:-5}"
SEED="${ERROR_SEED:-42}"
EXTRA_ARGS=()
POSITIONAL_COUNT=0

usage() {
    cat <<'EOF'
Usage: ./run_error_experiments.sh [system_name] [environment_name] [options]

Runs the naive-vs-optimized tracking-error experiment:
  experiments/error_experiment.py

Defaults:
  system_name=kinematic_car
  environment_name=gaussian
  --num-controls 5 --num-trials 5 --seed 42

Examples:
  ./run_error_experiments.sh
  ./run_error_experiments.sh double_integrator gaussian --num-controls 8
  ./run_error_experiments.sh pushing_object mujoco --duration 2.0

Wrapper options:
  --system-name NAME       kinematic_car, double_integrator, or pushing_object
  --environment-name NAME  gaussian or mujoco
  --num-controls N         Number of controls
  --num-trials N           Number of repeated trials
  --seed N                 Base random seed
  -h, --help               Show this help

Other options are passed through to error_experiment.py.

Environment:
  PYTHON_BIN               Python executable (default: /usr/bin/python3)
  TRIAL_TIMEOUT_SECONDS    Timeout for the experiment process (default: 3600)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --system-name|--system)
            SYSTEM_NAME="$2"
            shift 2
            ;;
        --environment-name|--environment|--simulator-mode)
            ENVIRONMENT_NAME="$2"
            shift 2
            ;;
        --num-controls)
            NUM_CONTROLS="$2"
            shift 2
            ;;
        --num-trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
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
        --*)
            EXTRA_ARGS+=("$1")
            shift
            if [[ $# -gt 0 && "$1" != --* ]]; then
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
        *)
            if [[ "${POSITIONAL_COUNT}" -eq 0 ]]; then
                SYSTEM_NAME="$1"
            elif [[ "${POSITIONAL_COUNT}" -eq 1 ]]; then
                ENVIRONMENT_NAME="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            POSITIONAL_COUNT=$((POSITIONAL_COUNT + 1))
            shift
            ;;
    esac
done

export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "Error experiment"
echo "  script:      ${ROOT_DIR}/experiments/error_experiment.py"
echo "  system:      ${SYSTEM_NAME}"
echo "  environment: ${ENVIRONMENT_NAME}"
echo "  controls:    ${NUM_CONTROLS}"
echo "  trials:      ${NUM_TRIALS}"
echo "  seed:        ${SEED}"
echo "  timeout:     ${TRIAL_TIMEOUT_SECONDS}s"
echo

exec timeout "${TRIAL_TIMEOUT_SECONDS}" \
    "${PYTHON_BIN}" "${ROOT_DIR}/experiments/error_experiment.py" \
    "${SYSTEM_NAME}" \
    "${ENVIRONMENT_NAME}" \
    --num-controls "${NUM_CONTROLS}" \
    --num-trials "${NUM_TRIALS}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"
