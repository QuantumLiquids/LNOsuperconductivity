#!/bin/bash
# Submit with an existing deployment and log directory:
# sbatch --chdir="$DEPLOYMENT" --output="$DEPLOYMENT/logs/build-%j.log" \
#   source/src_ipeps_bilayer_two_orbital/cluster/build_cent.slurm "$DEPLOYMENT"
#SBATCH --job-name=lno-u1-build
#SBATCH --partition=cent
#SBATCH --nodelist=cent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=build-%j.log
set -euo pipefail
readonly LNO_DEPLOY="${1:-$HOME/LNOsuperconductivity/ipeps_u1_20261105}"
mkdir -p "$LNO_DEPLOY/logs"
trap 'rc=$?; echo "$rc" > "$LNO_DEPLOY/logs/build_exit_code.txt"; date -Is > "$LNO_DEPLOY/logs/build_finished_at.txt"' EXIT
date -Is > "$LNO_DEPLOY/logs/build_started_at.txt"
srun --ntasks=1 --cpus-per-task=1 --cpu-bind=threads \
  bash "$LNO_DEPLOY/source/src_ipeps_bilayer_two_orbital/cluster/build_cent.sh" "$LNO_DEPLOY"
