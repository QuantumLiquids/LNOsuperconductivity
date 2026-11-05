#!/bin/bash
#SBATCH --job-name=lno-ipeps-u1
#SBATCH --partition=cent
#SBATCH --nodelist=cent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail
deployment="${1:?deployment directory}"
manifest="${2:?campaign manifest}"
export AOCL_ROOT="${AOCL_ROOT:-$HOME/aocl/aocl-linux-gcc-5.1.0/gcc}"
export LD_LIBRARY_PATH="$AOCL_ROOT/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
export BLIS_NUM_THREADS=1 GOTO_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# One bound hardware thread on the allocated physical core; no MPI ranks.
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=threads \
  python3 "$deployment/source/src_ipeps_bilayer_two_orbital/cluster/cent_campaign.py" \
  run "$deployment" "$manifest" "$SLURM_ARRAY_TASK_ID"
