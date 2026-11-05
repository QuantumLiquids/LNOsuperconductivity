# Local projected-bilayer iPEPS validation, 2026-11-05

The implementation and an initial D=2 screen run on the local macOS CPU.
These results establish a working calculation, not a converged ground state.

## Setup

- Supplied projected Hamiltonian and table: `params/physics_table.json`.
- 4x4 rung cell, 32 Ni, physical rung dimension 81, fU1U1 (N, 2Sz).
- Exact cell sector N=48, total Sz=0; n=1.5/Ni.
- Dmin=1, Dmax=2, truncation 1e-12, two numerical threads.
- LayerAlignment=-1; zero interlayer hopping and exchange from the table.
- Tau/steps: 0.05/20, 0.02/20, 0.005/20 (total imaginary time 1.5).
- CTMRG Chi 16, tolerance 1e-8, at least 4 sweeps, cap 200.
- Unchanged final checkpoints remeasured at Chi 32, tolerance 1e-10.
- The first five starts share `ChargePattern=stripe_x, ChargeOffset=0`;
  the sixth rotates the charge seed to `stripe_y`.
- `params/screen_d2.json` reproduces these evolution settings.

## Short-run results

Energy is the expectation of the physical H per Ni in the supplied table's
units, with EpsilonZ=0 and EpsilonX=0.558. It includes onsite and exchange
terms; Mu=0. The last 20-sweep change is **not** a convergence error bar.

| Initial spin | Initial charge | E/Ni after 60 sweeps | Change in last 20 | Seconds |
|---|---|---:|---:|---:|
| diagonal_plus | stripe_x | 1.181997800 | -0.003206 | 68.6 |
| diagonal_minus | stripe_x | 1.180283322 | -0.003124 | 58.3 |
| stripe_x | stripe_x | 1.205865176 | -0.002332 | 45.6 |
| stripe_y | stripe_x | 1.159560457 | -0.004015 | 58.0 |
| neel | stripe_x | 1.182527028 | -0.001792 | 54.7 |
| stripe_x | stripe_y | 1.156675948 | -0.004656 | 84.0 |

The diagonal-plus branch retains a top-layer z-orbital Fourier amplitude
0.346224 at (pi/2,pi/2), compared with its seeded value 1/(2 sqrt(2)).
This is a local-Sz Fourier amplitude, not a two-point structure factor.
It shows that a diagonal magnetic branch is representable and survives this
short evolution. It does not establish spontaneous order or minimum energy.

The initial-charge dependence is substantial: the same stripe_x magnetic
seed reaches E/Ni 1.205865 with x-oriented charge stripes and 1.156676 with
y-oriented charge stripes. The latter is lower than both diagonal branches.
Therefore the present screen **does not reproduce a lowest-energy diagonal
phase**, and a ranking based only on the first three magnetic starts would
be misleading. Multiple charge orientations/phases, longer projection and
larger D remain necessary. The supplied near-isotropic couplings also differ
from the earlier Kondo model.

All six final states have D=2 on every bond and retain the exact cell sector.
The largest measured density deviation from 1.5/Ni is below 5e-12.
Chi 16 versus Chi 32 changes E/Ni by less than 1e-12 for these snapshots.
This check alone is not a bound on contraction or variational errors. The
energy still changes by roughly 0.002–0.005/Ni during the final 20 sweeps,
so imaginary-time convergence is not established.

The blocked-rung bond dimension is shared between the two layers. A product
of two independent single-layer D=2 states generally requires rung D=4;
this D=2 screen can allocate entanglement unevenly. At the table's zero
interlayer couplings, the relative layer spin registry is exactly degenerate.

## Verification

- Both C++20 executables build with Homebrew Clang 21.1.4 and Apple SDK libc++.
- Independent validator passes: all 81 basis states and 6561 two-rung input
  columns for both bond types, analytic Hund multiplets, literal J S.S,
  explicit Jordan-Wigner signs, Hermiticity, N/Sz conservation, graded
  operator encoding, and 48 spin/charge/phase seed combinations.
- All five initial two-sweep smoke runs reach D=2 and keep N=48,Sz=0.
- Checkpoint restart and a tau=0 identity sweep preserve energy to 1e-10.
- Independent --measure reproduces checkpoint energy.
- Existing output directories are rejected without changing any file;
  Boolean numeric parameters are rejected before output creation.
- The 60-sweep rotated-charge run used peak resident memory
  1,202,962,432 bytes (about 1.12 GiB), measured by child getrusage.
- Fixed an observed duplicate-OpenMP startup abort by linking Homebrew's
  standalone libomp, also used by OpenBLAS, instead of LLVM's bundled copy.

## Artifacts and reproduction

`local_screen_20261105.csv` contains full-precision results and checkpoint
paths relative to the repository. Raw tensors and histories remain under
`run/bilayer_competitors_d2_screen_20261105/`,
`run/bilayer_additional_d2_screen_20261105/`, and
`run/bilayer_stripe_x_charge_y_d2_screen_20261105/`; these run directories
are ignored by Git. Each has its exact physics/algorithm inputs. The first
root contains `spin_energy_final.png` for the two diagonal branches and
rotated-charge stripe_x branch.

To repeat the full magnetic/charge screen, choose a new output directory:

```sh
python3 src_ipeps_bilayer_two_orbital/compare_seeds.py \
  src_ipeps_bilayer_two_orbital/params/physics_table.json \
  src_ipeps_bilayer_two_orbital/params/screen_d2.json \
  run/bilayer_new_screen --charge-patterns stripe_x stripe_y
```

Add `--charge-offsets 0 1 2 3` for all relative translations, or
`--prepare-only` to create inputs. `explore_d2.json` provides a longer
schedule; its completion still needs energy/order convergence checks.

`local_screen_20261105_provenance.json` records dependency HEADs, locally
modified tracked files, and application source hashes. PEPS and UltraDMRG
had pre-existing edits outside the infinite-update paths; no dependency
checkout was modified by this task.
