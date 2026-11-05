# Exchange-parameter search, 2026-11-05

A fixed-Chi D=2 search found a negative-diagonal candidate below all sampled
axial, Neel and FM branches. The literal positive-diagonal branch remains
higher in the current square-lattice coordinates. The candidate survives
additional common imaginary-time projection and a smaller final time step.
It is a starting point for detailed convergence work, not a ground-state
identification or an experimental parameter fit.

The user subsequently confirmed that either diagonal direction is acceptable
for discovery; the coordinate convention will be aligned with the new paper
later. The original axes and both Fourier labels remain explicit here.

## Physical parameter origin and search policy

The collaborator reports that the tight-binding parameters come from DFT,
while exchange couplings were scanned on the scale t^2/U. These statements
do not make the table a unique inference from experiment. This search keeps
all hopping, V=3, Hund JH=1, EpsilonX-EpsilonZ=0.558 and zero interlayer
couplings fixed. Uniform exchange scaling refers to antiferromagnetic Jx,
Jxz and JzPerp; it does not scale the ferromagnetic Hund coupling.

At the user's request, every new screening run uses common Chi=8 and
CTMRG tolerance 1e-6, without a Chi sweep. The cell is 4x4 rungs, D=2,
N=48 and Sz=0. Competing seeds include both diagonals, both axial stripes,
Neel and FM. The first five complete parameter points have 8 starts each,
including two diagonal spin phases and selected charge orientations. The
0.20 candidate has 6 selected starts. These 46 cold starts are not an
exhaustive search over all charge phases or magnetic textures.

Cold starts use tau 0.1 for 25 sweeps and tau 0.03 for 15 sweeps, total time 2.95.
The manifest `params/exchange_scan_d2.json` records explicit couplings and
`params/exchange_candidate_d2.json` records the additional candidate. Of the
first manifest's 8 planned points, 5 were completed, one only partially run,
and two left unrun after prioritizing stronger anisotropy. The partial point
is excluded from the comparison. All completed checkpoints were retained.

## Coarse results

Define delta_plus=E(pi/2,pi/2)-min(E(pi,0),E(0,pi)) and delta_minus analogously
for(pi/2,-pi/2). Energies and gaps are per Ni in the supplied table units.
Groups are identified by measured final Fourier peaks, rather than seed labels.

| Point | Jxz1 | Jxz2 | delta_plus | delta_minus |
|---|---:|---:|---:|---:|
| baseline | 0.038500 | 0.026800 | +0.031926 | +0.030623 |
| scale025 | 0.009625 | 0.006700 | +0.031645 | +0.030352 |
| ratio10_mean | 0.059364 | 0.005936 | +0.031828 | +0.024494 |
| ratio10_scale025 | 0.014841 | 0.001484 | +0.031602 | +0.030541 |
| strong_anisotropy | 0.154000 | 0.006700 | +0.029482 | -0.000149 |
| jxz1_020_jxz2_00067 | 0.200000 | 0.006700 | +0.031572 | -0.010048 |

`scale025` scales every AF exchange by 0.25. `ratio10_mean` changes the Jxz
ratio to 10 while preserving Jxz1+Jxz2 and leaving Jx unchanged.
`ratio10_scale025` combines that ratio with uniform AF scaling 0.25.
The last two points instead have Jx1=0.01625 and Jx2=0.015825, Jxz2=0.0067,
and Jxz1=0.154 or 0.20. They therefore enhance interorbital exchange
anisotropy much more strongly than the original table.

The shallow apparent crossing at Jxz1=0.154 is not reliable. Continuing
its negative-diagonal and best axial checkpoints by 40 equal tau 0.03 sweeps
changes their energies to 1.117813873571 and 1.114781758604, giving a positive
gap +0.003032114967. This overturned the coarse negative gap -0.000149087449.
Raw continuations are in `run/bilayer_anisotropy_continuation_20261105/`.

## Candidate that survives longer projection

The candidate couplings are:

```
Tx1=0.403, Tx2=0.398, Txz1=0.211, Txz2=0.190
Jx1=0.01625, Jx2=0.015825
Jxz1=0.20, Jxz2=0.0067
JH=1, V=3, EpsilonX-EpsilonZ=0.558
TzPerp=JzPerp=0, n=1.5/Ni
```

Here Jxz1/Jxz2=29.850746, versus 1.436567 in the original table; Jxz1 is
about 5.19 times its original value. This is an exploratory parameter change,
not evidence that the original DFT-derived system has this exchange ratio.
The moderate fixed-sum ratio changes tested above did not produce a crossing.

All six candidate branches continue from their cold checkpoints with
60 sweeps at tau 0.03 and 40 sweeps at tau 0.01. Total accumulated imaginary
time is 5.15, with the same D=2 and Chi=8 throughout.

| Initial branch | Final E/Ni | Dominant q on both layers |
|---|---:|---|
| diagonal_minus | 1.104196330 | pi2_minus_pi2 |
| stripe_x | 1.110755777 | pi_0 |
| stripe_y | 1.119946529 | 0_pi |
| neel | 1.139947128 | pi_pi |
| diagonal_plus | 1.143471363 | pi2_pi2 |
| fm | 1.172771835 | 0_0 |

The final negative-diagonal minus axial gap is -0.006559447092/Ni. Before the
last 40 tau 0.01 sweeps it was-0.006504510117; it changes by only -5.49e-5/Ni
in that last stage. Absolute energies still decrease by approximately
0.0013--0.0017/Ni in the last stage, so full projection convergence has not
been established. The observed sign is a useful coarse candidate, not a
certified error-bounded ground-state comparison.

Both layers retain the intended Fourier peak in every refined branch.
Every virtual bond has dimension 2 and all states preserve exact N=48,Sz=0.
The maximum measured filling error in these refined states is2.22e-16/Ni;
separate layer densities and Fourier amplitudes are in the refined CSV.

## Coordinate and spin-phase convention

The solver uses x=col and y=row. Its strong zigzag bonds are east at even
row+col and south at odd row+col. At this convention, increasing Jxz1/Jxz2
classically favors the negative diagonal, with one specific ++-- phase.
For a product seed with one x electron per rung,
E_Jxz/Ni=(Jxz2-Jxz1)/4 for diagonal_minus SpinOffset 0; SpinOffset 1 reverses
this sign. The literal positive diagonal has canceling Jxz contributions.
The independent model validator checks this coefficient and phase dependence.

The older single-layer initializer called `stripe_pi2pi2` uses
`y_chain=row-floor((row+col)/2)` and alternating signs in y_chain. Its actual
Fourier peak is also on the negative diagonal under x=col,y=row. Thus its
name is not sufficient to establish the coordinate convention in the paper.
Under the explicit relabeling x_paper=x_code,y_paper=-y_code, the negative
diagonal is called (pi/2,pi/2). The hopping/exchange geometry must be relabeled
with the coordinates; this does not demonstrate that the literal positive
diagonal wins in the current fixed axes. Both directions are reported above.

## Increasing D

`d_controls_20261105.md` and its CSV report matched baseline D=2,3,4 controls
from identical old checkpoints, with equal extra imaginary time 0.06 and
fixed Chi 8. The positive-diagonal minus axial gaps are respectively
0.026061619551,0.026493701404,0.026221520950/Ni. Both branches gain energy
from larger D, but the relative gap does not improve in that short test.
This cannot predict the D trend at other parameter points.

The candidate also has a separate matched D=2 versus D=3 short test, starting
from its coarse checkpoints and adding exactly the same time 0.06. Its
negative-diagonal-minus-axial gap is -0.009737640664 at D=2 and
-0.010112801894 at D=3, a change of -0.000375161231/Ni. Thus this candidate's
short test slightly favors the diagonal as D increases, whereas the original
baseline test slightly favors the axial branch. Neither is a D extrapolation.
`candidate_d_controls_20261105.md` and its CSV preserve these data. They use
different starting snapshots and accumulated time from the longer refined
D=2 values above, so those energies must not be mixed into one D trend.

The shared rung D=2 is more restrictive than two independent single-layer
D=2 states, whose product generally requires rung D=4. Larger D is therefore
worth testing, but gives no general preference for a particular magnetic
wavevector. In the actual longer D4 attempts, local sweeps cost tens of
seconds and finite-Chi 8 CTMRG measurements around one minute. At the table's
zero interlayer couplings, a future single-layer d=9 calculation can exploit
exact layer factorization more cheaply; finite interlayer couplings would
require restoring the bilayer treatment.

## Files and validation

- `exchange_search_cold_20261105.csv`: full precision six-point coarse readout.
- `exchange_candidate_refined_20261105.csv`: all six refined states.
- Raw coarse runs: `run/bilayer_exchange_scan_d2_20261105/` and
  `run/bilayer_exchange_candidate_d2_20261105/`.
- Raw refinement: `run/bilayer_exchange_candidate_refined_20261105/`.
- `params/refine_d2.json` records the continuation schedule. The raw run
  algorithm JSON files include each source checkpoint and cache location.
- `params/physics_diagonal_candidate.json` and `params/candidate_d2.json`
  provide the complete candidate Hamiltonian and merged time 5.15 schedule
  for a fresh negative-diagonal run.

The C++ model validator passes the full two-rung Hamiltonian checks plus
192 spin/charge/phase seed combinations and the analytic zigzag-phase energy.
The new gate cache is keyed by physics, basis, tau and executable build;
subsequent seeds load the same immutable gate tensors. Resume validation
reuses completed runs without new simulation. A stop-file check launches no
solver, and an actual invalid solver input propagates a failing scan exit
status. The original broad scan was reprioritized only after its active
case completed; its record is in `reprioritized.json`. No older checkpoints
were overwritten. No detailed Chi scan was performed in this search.

The ready-to-run merged 140-sweep candidate schedule was also run afresh with
the immutable gate cache. It reproduces the staged/restarted negative-diagonal
energy 1.1041963301971733/Ni to 1e-11. Application, executable and parameter
hashes are recorded in `exchange_search_20261105_provenance.json`.

The checked figure and its source audit are saved under
`run/bilayer_exchange_results_20261105/exchange_and_d_controls_checked.png`
and the same-stem JSON. The standalone helper is `plot_exchange_scan.py`.
