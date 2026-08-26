#!/usr/bin/env bash
# Run the 100-trial fixed-reference deviation experiment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AURA_PYTHON_BIN:-python3.10}"
python_paths="${repo_root}/.deps/ompl/lib/python3.10/site-packages:${repo_root}/.deps/python"
library_paths="${repo_root}/.deps/ompl/lib"
torch_env="${AURA_TORCH_ENV:-${HOME}/pytorch-gpu}"
if [[ -d "${torch_env}/lib/python3.10/site-packages" ]]; then
    python_paths="${python_paths}:${torch_env}/lib/python3.10/site-packages"
    library_paths="${library_paths}:${torch_env}/lib/python3.10/site-packages/nvidia/cudnn/lib"
fi
export PYTHONPATH="${python_paths}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${library_paths}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${python_bin}" "${repo_root}/experiment/deviation_error.py" \
    --system all \
    --environment all \
    --num-trials 100 \
    --num-controls 10 \
    --seed 42 \
    --output-dir "${repo_root}/results/error_experiment" \
    "$@"
