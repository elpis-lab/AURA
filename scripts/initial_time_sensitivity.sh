#!/usr/bin/env bash
# Run the configured initial-time sweep and open its average/median surfaces.
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

"${python_bin}" "${repo_root}/experiment/initial_time_sensitivity.py" \
    --config "${repo_root}/configs/experiments/initial_time_sensitivity.yaml" "$@"
"${python_bin}" "${repo_root}/scripts/plot_initial_time.py"
