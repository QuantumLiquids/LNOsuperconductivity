#!/bin/bash
# Reproduce the one-core GCC/AOCL build tested on cent. Run within Slurm.
set -euo pipefail
readonly LNO_DEPLOY="${1:-$HOME/LNOsuperconductivity/ipeps_u1_20261105}"
export LNO_DEPLOY
export AOCL_ROOT="$HOME/aocl/aocl-linux-gcc-5.1.0/gcc"
export LD_LIBRARY_PATH="$AOCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset CPLUS_INCLUDE_PATH C_INCLUDE_PATH
readonly LNO_CMAKE="$HOME/cmake-3.28.0-rc5-linux-x86_64/bin/cmake"
mkdir -p "$LNO_DEPLOY/logs"
python3 - <<'PY'
import json
import os
import pathlib

deployment = pathlib.Path(os.environ["LNO_DEPLOY"])
cpus = sorted(os.sched_getaffinity(0))
cores = []
for cpu in cpus:
    topology = pathlib.Path("/sys/devices/system/cpu") / f"cpu{cpu}" / "topology"
    cores.append({"cpu": cpu, "core": (topology / "core_id").read_text().strip(),
                  "socket": (topology / "physical_package_id").read_text().strip()})
(deployment / "logs/build_affinity.json").write_text(json.dumps({
    "cpus": cpus, "topology": cores, "slurm_job": os.environ.get("SLURM_JOB_ID"),
    "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
    "slurm_threads_per_core": os.environ.get("SLURM_THREADS_PER_CORE")}, indent=2) + "\n")
assert len(cpus) == 1, (cpus, "Expected one hardware-thread CPU for serial build")
PY

if [[ -f "$LNO_DEPLOY/build/Makefile" ]] && make -C "$LNO_DEPLOY/build" -j1 \
    ipeps_bilayer_simple_update_u1 ipeps_bilayer_validate_model_u1; then
  :
else
  "$LNO_CMAKE" -S "$LNO_DEPLOY/source" -B "$LNO_DEPLOY/build" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DMPI_C_COMPILER=/usr/bin/mpicc -DMPI_CXX_COMPILER=/usr/bin/mpicxx \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_PREFIX_PATH="$HOME/.local;$AOCL_ROOT" \
    -DTENSOR_HEADER_PATH="$LNO_DEPLOY/dependencies/TensorToolkit/include" \
    -DMPS_HEADER_PATH="$LNO_DEPLOY/dependencies/UltraDMRG/include" \
    -DPEPS_HEADER_PATH="$LNO_DEPLOY/dependencies/PEPS/include" \
    -Dhptt_INCLUDE_DIR="$HOME/.local/include" -Dhptt_LIBRARY="$HOME/.local/lib/libhptt.a" \
    -DHP_NUMERIC_USE_AOCL=ON -DHP_NUMERIC_USE_MKL=OFF -DHP_NUMERIC_USE_OPENBLAS=OFF \
    -DQLMPS_USE_GPU=OFF
  make -C "$LNO_DEPLOY/build" -j1 \
    ipeps_bilayer_simple_update_u1 ipeps_bilayer_validate_model_u1
fi
"$LNO_DEPLOY/build/ipeps_bilayer_validate_model_u1"
sha256sum "$LNO_DEPLOY/build/ipeps_bilayer_simple_update_u1" \
  "$LNO_DEPLOY/build/ipeps_bilayer_validate_model_u1" > "$LNO_DEPLOY/logs/executables.sha256"
ldd "$LNO_DEPLOY/build/ipeps_bilayer_simple_update_u1" > "$LNO_DEPLOY/logs/ldd_u1.txt"
date -Is > "$LNO_DEPLOY/logs/build_verified_at.txt"
