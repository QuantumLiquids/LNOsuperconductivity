# Baseline D controls, 2026-11-05

These are short, paired simple-update controls for the original supplied
parameter table. They address whether increasing the shared bilayer rung D
already favors the literal `(pi/2,pi/2)` branch. They do not establish a
converged phase or predict the D dependence at other J parameters.

## Matched protocol

All six runs use the same 4x4 rung geometry, N=48 electrons, total Sz=0,
original table couplings, and layer alignment -1. Each D starts from the same
previous D=2 checkpoint for its magnetic branch:

- Diagonal-plus: `run/bilayer_competitors_d2_screen_20261105/diagonal_plus/checkpoint_step00000060`.
- Axial `(pi,0)`: `run/bilayer_stripe_x_charge_y_d2_screen_20261105/checkpoint_step00000060`.

The diagonal branch has the old x-oriented charge seed; the axial branch
has the old y-oriented charge seed. They were not reinitialized when D was
changed. Each continuation applies exactly two sweeps at tau=0.03, adding
imaginary time 0.06. Dmin=1, truncation tolerance 1e-12, inverse cutoff 1e-12,
and one numerical thread are shared. All bonds reach the listed D.

CTMRG is fixed at Chi=8, tolerance 1e-6, min 2 and max 100 sweeps. No Chi sweep
was performed. The different actual CTMRG iteration counts are convergence
outcomes, not different requested controls. This is deliberately an
exploratory comparison under the user's coarse-screening policy.

## Results

Energies are per Ni in the original table units. A positive gap means the
axial branch is lower. The last column compares that gap with its D=2 value.

| D | Diagonal-plus E/Ni | Axial E/Ni | Diagonal minus axial | Gap change from D=2 |
|---|---:|---:|---:|---:|
| 2 | 1.180189526799 | 1.154127907249 | 0.026061619551 | +0.000000000000 |
| 3 | 1.172201153450 | 1.145707452046 | 0.026493701404 | +0.000432081854 |
| 4 | 1.168275367621 | 1.142053846671 | 0.026221520950 | +0.000159901399 |

Increasing D lowers both measured energies. In this short baseline test,
D=3 and D=4 slightly increase the diagonal-minus-axial gap, by about
4.32e-4 and 1.60e-4/Ni respectively. Thus these data do not support a claim
that increasing D automatically favors the diagonal branch. This conclusion
is limited to the original parameters and a very short matched continuation;
there is no general monotonic rule for the energy difference, and simple
update does not guarantee variational optimization.

Both layers retain their respective target spin Fourier peaks in every run:
`pi2_pi2` for the diagonal branch and `pi_0` for the axial branch. The CSV
contains their separate top/bottom z-orbital amplitudes, layer densities,
full-precision energies and checkpoint paths. Every run preserves the exact
N=48,Sz=0 tensor sector. The maximum measured deviation from 1.5/Ni is
1.901e-10. The finite-Chi density estimates are not exact sector constraints.

## Local costs

Wall times include initial/final CTMRG measurements and vary with concurrent
local work. Gates were cached; the first D4 diagonal probe also prepared its
gates. The short timings are not estimates of converged production cost.

| D | Branch | Total seconds | Two sweep seconds | Final CTMRG seconds | CTMRG sweeps |
|---|---|---:|---|---:|---:|
| 2 | diagonal_plus | 7.34 | 0.42, 0.45 | 3.15 | 4 |
| 2 | stripe_x | 6.98 | 0.48, 0.47 | 2.98 | 4 |
| 3 | diagonal_plus | 17.55 | 1.20, 3.15 | 9.97 | 4 |
| 3 | stripe_x | 22.92 | 1.47, 5.64 | 12.04 | 4 |
| 4 | diagonal_plus | 22.08 | 0.70, 3.83 | 9.25 | 4 |
| 4 | stripe_x | 112.33 | 3.92, 22.87 | 82.12 | 6 |

A longer D4 attempt illustrates the cost after virtual sectors fill: local
sweeps reached 20--29 seconds and fixed-Chi 8 CTMRG measurements took roughly
57--82 seconds. D2 local sweeps were much cheaper. Gate preparation in the initial probe
used peak resident memory 1,222,230,016 bytes (about 1.14 GiB). The longer
D4 axial job loaded cached gates and peaked at 256,655,360 bytes (about
245 MiB); CPU time, rather than state storage, was the immediate limit.
These are child getrusage measurements. A product of two
independent single-layer D=2 states generically requires rung D=4; rung D=2
shares its small virtual space between both decoupled layers. This motivates
checking larger D, but does not determine which magnetic branch gains more.

## Longer controls and cost caps

The D2 continuations completed 40 sweeps at tau 0.03. A D4 diagonal attempt
was stopped after a valid step 20 measurement because it was slow. It has
E/Ni=1.085910255624 and measured n/Ni=
1.499998887036 at that checkpoint.

The corresponding D4 axial attempt saved valid measurements at steps 5,10,15.
Its local step 20 took 40.85 seconds and triggered the requested 30-second
sweep cap before the final measurement finished. Its latest valid measured
checkpoint is step 15, with E/Ni=1.070982475418.
The directory `stripe_x_d4_steps20/checkpoint_step00000020` is incomplete and
must not be loaded or treated as a result.

These longer D4 snapshots have unequal continuation times, so they cannot
be used together to infer a D-dependent energy gap. The six matched short
runs above are the usable D comparison. The stopped diagonal checkpoint's
n/Ni deviation is approximately 1.1e-6; this was reported without an extra
Chi study. No further long D4 or D3 grid was launched.

## Artifacts and provenance

All raw controls are under `run/bilayer_d_control_20261105/`, ignored by Git.
`short_d_control_results.json` records completed short jobs and costs;
`axis_d4_control_result.json` records the capped long axial job. The controller
scripts and their algorithm JSON files preserve commands and input settings. No simulation process remains active for this subtask.

| D | Branch | Short run directory |
|---|---|---|
| 2 | diagonal_plus | `run/bilayer_d_control_20261105/diagonal_plus_d2_steps2` |
| 2 | stripe_x | `run/bilayer_d_control_20261105/stripe_x_d2_steps2` |
| 3 | diagonal_plus | `run/bilayer_d_control_20261105/diagonal_plus_d3_steps2` |
| 3 | stripe_x | `run/bilayer_d_control_20261105/stripe_x_d3_steps2` |
| 4 | diagonal_plus | `run/bilayer_d_control_20261105/probe_d4` |
| 4 | stripe_x | `run/bilayer_d_control_20261105/stripe_x_d4_steps2` |

The gate cache records build `Nov  5 2026 16:58:38` and exact physics JSON.
Executable SHA256: `a8c87fec30539ea91b51d1cb8803b933b7c38c2b77c16a4ad1591b97cdb7928f`.
Physics JSON SHA256: `5c7425a1c09538b0398210f3ce95301df4ec2089c4dde78df78e7d689cd79546`.
Application source and dependency working copies may have local edits;
HEADs identify their baseline only:

- LNOsuperconductivity: `d5741e0bee1099f075582456f52b7dbc37343089`.
- PEPS: `9cd3837e1eb78fc805ceebdeb0cc24b0b822500a`.
- TensorToolkit: `2744eeb10fb33109e570c7d4997fce62e7ae9b6e`.
- UltraDMRG: `7ed427afb840f19fd064e4fab77561b0e11c7138`.

## Corrected analytic Jxz normalization

An independent Jxz-only CTM measurement of the original D1 diagonal-minus
product seed gives E/Ni=-0.0029250000000000, equal to
`(Jxz2-Jxz1)/4`, not `/2`. Each rung has one x electron and two Ni sites.
The raw operator-only result is
`run/bilayer_d_control_20261105/analytic_jxz_product.json`.
This factor correction does not change the directional conclusion: under
the code's current zigzag convention, enhancing Jxz1 relative to Jxz2
classically favors the literal negative diagonal for SpinOffset 0.
