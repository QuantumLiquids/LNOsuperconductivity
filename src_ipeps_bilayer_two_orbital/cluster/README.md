# cent: N-only bilayer iPEPS screening

The requested deployment is isolated under
`~/LNOsuperconductivity/ipeps_u1_20261105/`, with `source/`, matching dependency
header snapshots, `build/`, `jobs/`, `logs/` and `runs/`. Shared checkouts and
unrelated jobs are not modified. The original code target conserves N and Sz;
all calculations in this campaign use `ipeps_bilayer_simple_update_u1`, which
conserves only N. Checkpoint symmetry is `fU1QN`, not `fU1U1QN`.

## Build

Use the tested GCC 11.4 / AOCL 5.1 toolchain and home CMake 3.28; the system
CMake 3.22 does not support FindBLAS's AOCL vendor. The build uses the copied
TensorToolkit, UltraDMRG and PEPS headers, plus existing HPTT under `~/.local`.
The shared PEPS checkout has a different CTMRG version, so do not substitute
its headers silently. `build_cent.sh` and `build_cent.slurm` preserve the
tested build recipe. Build one core at a time and run the `_u1` validator.

The initial verified build was Slurm job 168427, exit 0. It bound exactly one
hardware thread and passed the Hamiltonian, fermionic-sign, transverse-spin,
checkpoint-symmetry and analytic gate tests. Runtime uses one libgomp and
AOCL BLIS/FLAME. Every compute task also sets all BLAS/OpenMP thread counts 1.

## Campaign

From the deployment directory:

```sh
python3 source/src_ipeps_bilayer_two_orbital/cluster/cent_campaign.py \
  prepare jobs/campaign_manifest.json
sbatch --chdir="$PWD" --array=0-31%3 \
  source/src_ipeps_bilayer_two_orbital/cluster/cent_array.slurm \
  "$PWD" "$PWD/jobs/campaign_manifest.json"
```

`prepare` refuses to overwrite an existing manifest. Review it before changing
the protocol or point selection. The initial array is 168429. First three
tasks compare D=2,3,4 at the original table parameters for the negative
diagonal. Subsequent tasks add `(0,pi)` and the positive diagonal, then repeat
at the midpoint and candidate. The midpoint linearly interpolates only
Jx1,Jx2,Jxz1,Jxz2 between the table and the exploratory candidate; hopping,
Hund coupling, V, onsite splitting, filling and zero interlayer terms stay
fixed. Additional candidate D3 tasks cover FM, both quarter-axis stripes and
the other axial domain. An aligned-layer FM control at the table explicitly
permits a nonzero total Sz in N-only tensors.

All cold starts use a 4x4 rung cell, Dmin 1, N=48 and one thread. The preliminary
schedule is 10 sweeps at tau 0.1 then 10 at tau 0.03, total imaginary time 1.3.
Chi is fixed at 8, CTMRG tolerance 1e-5, no Chi sweep. Intermediate checkpoints
are saved every 5 sweeps without CTMRG; a final contraction is always required.
If final CTMRG exhausts 200 sweeps, the wrapper retries that unchanged final
checkpoint at the same Chi/tolerance with a 500-sweep cap. Failures remain
failures when the retry does not converge; missing results are never assigned
a high energy. All such retries are explicitly recorded.

Each array task requests one physical core with `--hint=nomultithread`,
`--ntasks=1`, `--cpus-per-task=1`, `--mem=4G`. `srun --cpu-bind=threads` binds
one hardware thread, which the runner verifies before computing. Do not use
multiple MPI ranks for this executable. On cent's core-based Slurm accounting,
the allocated core may be billed as two logical CPUs because its SMT sibling
is also reserved; the process still has one-CPU affinity and one thread.

Start with throttle 3, check measured RSS/current scheduler memory, then raise
only this array's throttle if appropriate. The initial free allocatable memory
was about 37 GiB; memory reservations limit concurrency before the idle cores.
Each case writes separate inputs, logs, checkpoints, binary/manifest hashes,
affinity, measured Sx/Sz Fourier amplitudes and peak child RSS. A completed
case is reused only when both its input and binary hashes match. Incomplete
cases are retained; rerun them under a fresh campaign name.

## Readout

```sh
python3 source/src_ipeps_bilayer_two_orbital/cluster/cent_campaign.py \
  collect "$PWD" "$PWD/jobs/campaign_manifest.json" "$PWD/logs/campaign_results.json"
```

This writes JSON and CSV, including missing/running/failed cases. For every
point and D it reports both diagonal energies relative to the measured
`(0,pi)` branch. Compare the **same point, initialization protocol, time and
Chi** across D. The phase labels use both layers' Sx/Sz Fourier powers and
flag different layer wavevectors. This is a collinear-start screen; it does
not optimize every spiral at the same q or establish a phase boundary.

After recompiling on another machine, use fresh gate caches. Native checkpoints
have a different schema between one- and two-U1 targets and cannot be mixed.
The Python scan helper's `--resume` also checks absolute cache/input paths;
a fresh remote run root is generally clearer than copying a local scan tree.
