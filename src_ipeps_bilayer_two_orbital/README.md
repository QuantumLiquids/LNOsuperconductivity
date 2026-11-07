# Projected two-orbital bilayer fermionic iPEPS

This implements the supplied follow-up Hamiltonian, with **no double occupancy
within each orbital**. It is distinct from the existing Kondo model: the z
orbital retains charge dynamics, including interorbital hopping. One PEPS site
is a complete vertical rung: top/bottom layers, z/x orbitals, with physical
dimension `3^4 = 81`. Both layers share the spatial PEPS bond dimension D.

The first intended use is an exploratory D=2 simple-update calculation on a
local CPU. A finite CTMRG environment measures energies and order parameters.
Neither a short smoke run nor convergence of simple-update bond weights
establishes a thermodynamic ground state.

## Model and conventions

The rung modes have the fixed Fock ordering
`(top z up, top z down, top x up, top x down,
bottom z up, bottom z down, bottom x up, bottom x down)`.
The basis is grouped into `(N, 2Sz)` sectors (`fU1U1QN`). Fermion parity is
computed from N. Jordan-Wigner signs inside a rung and between the two ordered
endpoint rungs are included before conversion to the library's graded tensor
operator convention. Spatial fermion signs are handled by the PEPS library.

For each Ni site, the onsite term is

```
epsilon_z n_z + epsilon_x n_x + V n_z n_x
    - 2 JH (S_z . S_x + n_z n_x / 4) - Mu (n_z + n_x).
```

The rung also includes `-TzPerp sum_sigma (z_top^dag z_bottom + h.c.)`
and `JzPerp S_top,z . S_bottom,z`. Each planar bond in each layer has
`-Tx (x_i^dag x_j + h.c.)`,
`-Txz (x_i^dag z_j + z_i^dag x_j + h.c.)`,
`Jx S_i,x . S_j,x`, and
`Jxz (S_i,x . S_j,z + S_i,z . S_j,x)`.
The spin-exchange terms do **not** include an extra `-n_i n_j/4`.
Onsite/rung terms are distributed equally over four adjacent planar bonds.

In particular, the Hund convention here has coefficient `-2 JH`, and its
density term combines with V to give `V - JH/2`. It must not be replaced by
the earlier Kondo code's `JK = -JH` convention.

`params/physics_table.json` transcribes the supplied parameter table in its
original energy units. The common energy zero is `EpsilonZ=0`,
`EpsilonX=0.558`. The parenthesized U=5 is not an additional Hubbard term in
this projected Hilbert space. No parameters are inferred from U.

On 2026-11-07 the collaborator confirmed opposite cross-orbital hopping signs
along x and y. The canonical table now explicitly sets `TxzYSign=-1`:
the Hamiltonian uses `-Txz` on horizontal bonds and `+Txz` on vertical bonds,
for both x-to-z/z-to-x channels and both layers. All amplitudes and exchanges
retain their tabulated values. An overall reversal of both cross-orbital
signs is an orbital gauge choice; their relative directional sign matters.
Missing `TxzYSign` retains +1 for historical input compatibility.
`physics_table_same_sign_legacy.json` and the old
`physics_diagonal_candidate.json` explicitly retain +1. **The 2026-11-05
energy searches and candidate below belong to that earlier same-sign model.**

For an east bond starting at `(row,col)`, type 1 is used if
`(row+col)%2 == 0`; for a south bond type 1 is used if the parity is 1.
All other bonds use type 2. Thus types 1/2 are zigzag chain/bridge bonds,
not simply horizontal/vertical bonds. Directional signs multiply these
amplitudes independently of the checkerboard type. The two cross-orbital
channels on any one bond share that bond's directional sign.

The supplied table sets both interlayer couplings to zero. The two layers
therefore decouple exactly; their relative spin orientation cannot be
selected energetically at those parameters. The blocked-rung representation
also shares a D=2 variational budget between the two layers, which is more
restrictive than two independent single-layer D=2 calculations.

## Unit cells and seeds

Coordinates are `x=col, y=row`, with wavevectors in the original square
lattice coordinates. Seeds are only initial conditions; evolution does not
pin the requested magnetic order. Every seed starts with both z orbitals
occupied and one x electron per rung. On the common 4x4 cell its occupied
layer follows `floor(x/2)%2`, identical for all magnetic seeds. This keeps
all AF-layer competitors in the same total Sz=0 sector; a checkerboard
layer charge pattern would artificially polarize the Neel seed's x spins.
Smaller validation cells use a balanced alternative charge seed.
`ChargePattern` can be `auto` (the above default), `stripe_x`, or `stripe_y`;
`ChargeOffset` (0..3) shifts its phase. Explicit x/y charge stripes require
period four in that direction. Use both orientations and offsets for multiple
starts: at small D and short projection time, the initial charge pattern can
substantially change the energy ranking of magnetic seeds.
The cell has three electrons/rung, or 1.5/Ni. Charge and Sz
are conserved by the tensor symmetry; orbital occupation can evolve.

| InitState | top-layer spin sign | wavevector |
|---|---|---|
| `diagonal_plus` | `++--` as `(x+y)%4` increases | `(pi/2, pi/2)` |
| `diagonal_minus` | `++--` as `(x-y)%4` increases | `(pi/2, -pi/2)` |
| `stripe_x` | `(-1)^x` | `(pi, 0)` |
| `stripe_y` | `(-1)^y` | `(0, pi)` |
| `stripe_pi2_x` | `++--` along x | `(pi/2, 0)` |
| `stripe_pi2_y` | `++--` along y | `(0, pi/2)` |
| `neel` | `(-1)^(x+y)` | `(pi, pi)` |
| `fm` | constant | `(0, 0)` |

`LayerAlignment` is +1 or -1. It sets the initial relative layer spin sign.
`SpinOffset` (0..3, default 0) shifts the `++--` diagonal spin pattern before
choosing the spin direction. Offsets 0 and 1 are distinct relative to the
checkerboard exchange pattern, even though their magnetic wavevector agrees.
Offsets 2 and 3 additionally reverse all spins. For axial/Neel seeds the offset
acts modulo 2. Scan these relative spin phases as well as charge configurations.
For an antialigned bilayer, layer-resolved magnetic Fourier amplitudes are
essential because the layer sum can cancel. A Fourier amplitude of local Sz
is a symmetry-breaking diagnostic, not a two-point structure factor.

For the quarter-axis seeds, perpendicular charge stripes keep the balanced
Sz=0 sector: `stripe_pi2_x` with `ChargePattern=stripe_y`, and vice versa.
Parallel charge stripes with relative `SpinOffset=1` are a second balanced
start. The two-U1 target rejects polarized quarter-axis starts; the N-only
target permits them. Auto selects a perpendicular charge seed on a 4x4 cell.
All ten independent q/-q Fourier pairs of a real 4x4 spin texture are now
measured, including the two quarter axes and mixed `(pi/2,pi)` momenta.

For a single diagonal `++--` pattern, translations obey `dx+dy=0 (mod 4)`.
The abstract minimum is a skew cell of area 4, e.g. `(1,-1),(2,2)`.
To accommodate both `(pi/2,pi/2)` and `(pi,0)` in the same unrestricted
cell requires at least area 8, e.g. `(2,2),(2,-2)`. The present library
supports rectangular cells; the smallest common rectangular choice is
**4x4 rungs (32 Ni sites)**. A 2x2 or 4x2 rectangular cell excludes the
literal diagonal order. Both diagonal directions are included because the
checkerboard zigzags single out a direction, and the old finite-system
stripe initializer uses a different diagonal convention.

## Build and run

Requires current TensorToolkit, UltraDMRG and PEPS headers, including
`InfiniteSimpleUpdateState`, fermionic CTMRG, and local imaginary-time gates.
Build from the repository's configured `build/` directory:

```sh
make -j4 ipeps_bilayer_simple_update ipeps_bilayer_validate_model
./ipeps_bilayer_validate_model
```

If this checkout has no build directory, configure using the repository's
macOS recipe, with the actual dependency paths (on this machine under
`/Users/wanghaoxin/GitHub/`, not the stale `~/PEPS` aliases). Set
`TENSOR_HEADER_PATH`, `MPS_HEADER_PATH`, and `PEPS_HEADER_PATH` explicitly
to each dependency's `include` directory if CMake does not find them.

From the repository root:

```sh
build/ipeps_bilayer_simple_update \
  src_ipeps_bilayer_two_orbital/params/physics_table.json \
  src_ipeps_bilayer_two_orbital/params/smoke_d2.json \
  run/bilayer_smoke
```

The output directory must be new. `smoke_d2.json` performs only two sweeps;
`screen_d2.json` reproduces the 60-sweep local screen, while
`explore_d2.json` supplies a longer decreasing-tau schedule. Start with the
smoke run to measure local wall time before launching several seeds.
The implementation caches four type/orientation gates per tau (`H1,H2,V1,V2`)
instead of diagonalizing a separate copy for every translated bond. Cache
format 2 records this ordering and the directional sign; earlier two-gate
caches are rejected. Use fresh runs/caches after the sign correction.

All JSON files use a `CaseParams` wrapper. Model couplings belong in the
physics file; cell, initialization, evolution and CTMRG controls belong in
the algorithm file. Restart a saved Gamma/lambda checkpoint by adding its
path as a fourth argument. Change the output directory for every run.

Prepare or run the competing seeds sequentially with:

```sh
python3 src_ipeps_bilayer_two_orbital/compare_seeds.py \
  src_ipeps_bilayer_two_orbital/params/physics_table.json \
  src_ipeps_bilayer_two_orbital/params/smoke_d2.json \
  run/bilayer_competitors
```

This writes separate logs/checkpoints and `comparison.json`/`comparison.csv`.
Add `--prepare-only` to write inputs without starting the calculations;
replace the algorithm with `explore_d2.json` for a longer screen. The table
labels initial seeds, not inferred final phases.
To scan both charge orientations, add `--charge-patterns stripe_x stripe_y`;
add `--charge-offsets 0 1 2 3` to scan relative translations as well. Each
combination gets its own directory. Use `--prepare-only` to review the number
of calculations first.

Remeasure one unchanged checkpoint at larger Chi by editing a copy of the
algorithm file and running:

```sh
build/ipeps_bilayer_simple_update --measure \
  src_ipeps_bilayer_two_orbital/params/physics_table.json \
  algorithm_chi32.json path/to/checkpoint result_chi32.json
```

## Comparing competing states

For initial parameter discovery, keep a common small Chi instead of performing
a Chi sweep at every point. `scan_exchange.py` and
`params/exchange_scan_d2.json` implement a coarse fixed-Chi=8 scan of explicit
exchange parameter sets with fixed DFT hopping, onsite terms and Hund JH.
The manifest separates uniform scaling of antiferromagnetic exchanges from
changing the Jxz1/Jxz2 ratio, including a fixed-sum ratio scan. It includes
both diagonal directions and relative spin phases, axial stripes, Neel and FM.

```sh
python3 src_ipeps_bilayer_two_orbital/scan_exchange.py \
  src_ipeps_bilayer_two_orbital/params/physics_table.json \
  src_ipeps_bilayer_two_orbital/params/exchange_scan_d2.json \
  run/exchange_scan
```

Use `--points name1 name2` for selected points and `--resume` to reuse completed
runs with identical inputs. Final layer-resolved Fourier peaks determine the
reported phase groups; seed names alone do not. Negative
`delta_plus_minus_axial` or `delta_minus_minus_axial` means the respective
diagonal branch has lower energy among the tested final states at that point.
Neel/FM remain additional competitors. Complete longer projection and D/Chi
checks only for promising candidates.

`GateCacheDir` optionally reuses the same serialized physical gates between
starts or D values. Cache identity includes the exact physics input, basis,
time step and executable build. Use a new cache after changing the binary.
Runs sharing a cache must execute sequentially; separate parameter points
use separate directories. The exchange scan manages this automatically.

Use identical 4x4 geometry, D, tau schedule, couplings and CTMRG controls
for each seed. Inspect the **final** layer/orbital-resolved Sz pattern and
Fourier amplitudes: an initializer's name does not identify its final phase.
Also compare density and the final virtual bond dimensions. A Dmax=2 request
can retain fewer states when the discarded-weight criterion permits it.

Converge the imaginary-time projection and reduce tau. Then remeasure the
candidate checkpoints at increasing Chi and tighter CTMRG tolerances. Compare
physical `H` energies per Ni (half the per-rung energy); a nonzero Mu is
removed using the measured density. A CTMRG sweep cap without convergence
is reported as a failure. Finite-Chi CTMRG energy is still approximate and
is not certified as a variational upper bound.

Try multiple charge/spin seeds and larger D after the first exploration.
D=2 can have strong charge-sector and layer allocation biases, particularly
for an 81-state rung. If energy differences are comparable to Chi/tau
changes, the ordering is unresolved. The weak anisotropy in the supplied
table need not reproduce the earlier model's diagonal stripe.

## Validation

`ipeps_bilayer_validate_model` checks the projected Fock basis, analytic
onsite Hund multiplets, exchange conventions, hopping signs, conservation,
Hermiticity, graded operator encoding, and seed Fourier wavevectors. Small
runtime checks should additionally verify D growth, finite hopping energy,
checkpoint/restart, measured filling and CTMRG convergence before long runs.

See `Note/local_validation_20261105.md` for the first local measurements and
their limits. These short runs do not identify a lowest-energy phase.

## Exchange-search candidate (2026-11-05)

The fixed-Chi=8 search and a longer common projection found a useful candidate
with Jx=(0.01625,0.015825), Jxz=(0.20,0.0067), and all original hopping,
Hund JH=1, V=3, onsite splitting and interlayer terms unchanged.
At D=2 and total imaginary time 5.15, the negative diagonal has
E/Ni=1.104196330 versus the best tested axial stripe 1.110755777.
The positive diagonal, other axial stripe, Neel and FM samples are higher.
The last time-step stage changes the diagonal-minus-axial gap by about 5.5e-5/Ni.
This is a parameter-screening candidate, with a much larger Jxz anisotropy
than the original table; it is not an experimental calibration.

The user accepts either diagonal direction during this search. The present
axes still distinguish `(pi/2,pi/2)` and `(pi/2,-pi/2)`; align axes, strong
zigzag bonds and momentum definitions with the follow-up paper before using
its wavevector convention. `SpinOffset` must be tracked with the bond pattern.

Ready-to-run inputs are `params/physics_diagonal_candidate.json` and
`params/candidate_d2.json`. For the candidate's negative-diagonal start:

```sh
build/ipeps_bilayer_simple_update \
  src_ipeps_bilayer_two_orbital/params/physics_diagonal_candidate.json \
  src_ipeps_bilayer_two_orbital/params/candidate_d2.json \
  run/diagonal_candidate_new
```

See `Note/exchange_search_20261105.md` for complete couplings, parameter
comparisons, continuation tests and scope. `Note/d_controls_20261105.md`
contains baseline D=2,3,4 controls; `Note/candidate_d_controls_20261105.md`
contains the matched D=2,3 candidate comparison. Larger D lowers both branch
energies in these short tests, but its effect on their difference depends on
parameters. No new Chi sweep was performed during the parameter search.

## N-only tensors and cent deployment

`ipeps_bilayer_simple_update_u1` and `ipeps_bilayer_validate_model_u1` compile
with `LNO_IPEPS_SPIN_U1=0`. Their physical and virtual indices use `fU1QN(N)`;
neither a spin quantum number nor an exact total-Sz sector is imposed.
The exact sector metadata reports Sz as null, and measured total spin is
kept separately. Fermion parity remains fixed by N modulo 2. Native two-U1
checkpoints have a different symmetry schema and cannot be loaded directly.

The gate builder uses the Hamiltonian's intrinsic Sz conservation to
exponentiate small auxiliary blocks, then copies every coefficient into
the N-only gate. This operator optimization does not reintroduce a spin
constraint on PEPS tensors or truncation. Its analytic Hund/identity tests
pass for both modes. N-only diagnostics include Sx and Sz; phase grouping
uses the sum of their Fourier powers on both layers, and flags differing
layer wavevectors. Real coefficients imply zero Sy expectation. The present
starts are collinear; a general spiral at a specified q is not optimized.

The `cluster/` directory contains a one-physical-core cent Slurm workflow,
an isolated build template and a prepare/run/collect campaign helper.
See `cluster/README.md` for exact commands. It keeps all hopping at the
supplied values and compares the original table, a midpoint in AF exchanges,
and the exploratory candidate at D=2,3,4 using a common short protocol.
The reference gap is explicitly relative to the measured `(0,pi)` branch.
FM and quarter-axis guards are included at candidate D=3.
The completed campaign readout and its limits are recorded in
`Note/cent_u1_d234_20261105.md`; these N-only cold-start data use a different
projection history from the earlier local two-U1 controls.

`cluster/prepare_cent_followup.py` prepares frozen-state Chi tests, matched
checkpoint continuations, and a constrained exchange scan from the original
campaign readout. `cluster/cent_followup.py` runs/collects these explicit
manifests; `plot_followup.py` plots their measured branch gaps. The design
and exact allowed exchanges are in `Note/constrained_exchange_design_20261105.md`.
Its completed historical same-sign readout is archived in
`Note/cent_followup_same_sign_20261105.md`. For the corrected opposite-sign
model, `cluster/prepare_opposite_txz.py` prepares fresh original-table starts
and selects matched D4 cold starts from their measured D3 branch minima.
It also prepares common-Chi fixed-state measurements and a constrained
exchange scan. See [the opposite-sign run record](Note/opposite_txz_20261107.md)
for the inputs, energy comparisons and checkpoint provenance.

`ipeps_bilayer_components_u1` is a separate measurement-only target. It uses
the same converged CTMRG environment and bond density matrices to split the
energy into ten coupling-weighted Hamiltonian pieces, including both Hund
terms. Their sum is checked against the original total estimator. Its CLI is
the same `--measure physics algorithm checkpoint result.json` form. Build it
explicitly; ongoing calculations can retain the original executable.
`plot_energy_components.py` plots signed differences from those measurements
and labels the hopping sign convention in the figure and provenance.

`MeasureInitial=false` and `MeasureStageEnds=false` allow a coarse screen to
measure only after the final evolution step. `CheckpointEvery` retains states
without running CTMRG at every checkpoint. A failed final contraction keeps
the evolved state; the cent wrapper can retry the same Chi and tolerance
with a larger iteration cap, and records that recovery separately.
