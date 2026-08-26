#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ompl_source="${AURA_OMPL_SOURCE:-${HOME}/Documents/ompl}"
ompl_build="${AURA_OMPL_BUILD:-${ompl_source}/build-aura}"
python_bin="${AURA_PYTHON_BIN:-python3.10}"
jobs="${AURA_BUILD_JOBS:-$(nproc)}"

for planner_dir in aorrt aoest sststar; do
    if [[ ! -d "${ompl_source}/src/ompl/control/planners/${planner_dir}" ]]; then
        echo "Missing custom planner source: ${planner_dir}" >&2
        exit 2
    fi
done

cmake -S "${ompl_source}" -B "${ompl_build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DOMPL_BUILD_PYTHON_BINDINGS=ON \
    -DPython_EXECUTABLE="${python_bin}" \
    -DCMAKE_INSTALL_PREFIX="${repo_dir}/.deps/ompl"
cmake --build "${ompl_build}" -j"${jobs}"
cmake --install "${ompl_build}"

PYTHONPATH="${repo_dir}/.deps/ompl/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}" \
"${python_bin}" -c \
    'from ompl import control as oc; assert all(hasattr(oc, n) for n in ("AORRT", "AOEST", "SSTStar")); print("custom OMPL planners ready")'
