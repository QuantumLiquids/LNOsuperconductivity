# Archived same-sign follow-up, initiated 2026-11-05

> Historical model: `TxzYSign=+1`. On 2026-11-07 the collaborator confirmed
> opposite x/y cross-orbital hopping signs. These results remain useful for
> diagnosing the earlier calculation; they are not results for their DMRG
> Hamiltonian. The corrected model uses fresh states and a separate deployment.

In the same-sign model, both a longer projection and the allowed exchange
scan leave the best sampled axial branch below both diagonal branches at D4
and D3, respectively. A decomposition of the original-table D3 pair finds
that the x-orbital hopping term contributes most of its axial advantage.

## Time and environment checks

The 12 D3/D4 continuations restart the original T=1.3 states and add 20
sweeps at tau=0.03 and 20 at tau=0.01, reaching T=2.1. Both diagonals and
(0,pi) are measured at the original table and the historical strong-Jxz
candidate. These continuation comparisons use Chi=8; the final CTM tolerance
is 1e-6, versus 1e-5 in the original screen. All 12 complete.

| Point | D | Negative diagonal minus (0,pi), T=1.3 | T=2.1 |
|---|---:|---:|---:|
| table | 3 | +0.030403810 | +0.033869524 |
| table | 4 | +0.045049392 | +0.048467296 |
| historical candidate | 3 | -0.011679463 | -0.006750788 |
| historical candidate | 4 | +0.016171453 | +0.02341601 |

Frozen D4 T=1.3 checkpoints were also remeasured at Chi=8,12,16 with
identical tolerance 1e-6. At the original table, the negative-diagonal gap
changes by -5.45e-10/Ni between Chi8 and Chi16. At the historical candidate,
Chi12 and Chi16 agree within 1.33e-10/Ni, with gap +0.0161710883/Ni.
The candidate axial Chi8 measurement did not converge in 200 or 500
iterations; its strict-tolerance Chi8 gap remains missing. The higher-Chi
measurements preserve the axial advantage for these particular frozen states.

## Constrained exchange scan

The eight points and seeds are specified in
`constrained_exchange_design_20261105.md`. All obey
`min(Jx1,Jx2)>Jxz1`, `Jxz2/Jxz1>0.25`; hopping magnitudes, onsite terms and
zero interlayer couplings remain fixed. Each point has eight cold D3 starts
with T=2.95. The common final readout is Chi=16. After several Chi8 axial
contractions failed, the 25 already evolved states were retained for Chi16
measurement and 39 unstarted cases were launched with Chi16 directly. No
state's projection schedule was changed, and no Chi8 and Chi16 energies
were mixed to find a branch minimum.

| Point | Minimum diagonal minus minimum axial, per Ni |
|---|---:|
| original table | +0.033694068 |
| table, AF scale 0.5 | +0.033946622 |
| interpolation 0.05 | +0.031731968 |
| interpolation 0.05, AF scale 0.5 | +0.03296553 |
| interpolation 0.10 | +0.02976642 |
| interpolation 0.10, AF scale 0.5 | +0.03198357 |
| boundary ratio 0.30 | +0.027600704 |
| boundary ratio 0.30, AF scale 0.5 | +0.030902539 |

All 64 measured states retain actual D=3 and exact N=48 with no spin U(1).
The maximum measured filling deviation is 1.71e-9/Ni. The best diagonal is
the negative orientation at each point, while the best axial state is
(0,pi). Half-strength AF exchanges do not improve these sampled gaps.
The much stronger historical candidate violates the revised exchange bounds
and is included only as a convergence control above.

## Why the sampled axial branch is lower

For the same-sign original table at D3, T=2.95, Chi16, the best negative
diagonal and (0,pi) energies are 1.1079552743865 and 1.0742612064038/Ni.
Using a common CTM contraction procedure for each fixed state, the ten-term
decomposition gives these differences, diagonal minus axial:

| Contribution | Difference/Ni |
|---|---:|
| x hopping | +0.027431458 |
| xz hopping | +0.000415107 |
| x exchange | +0.002244428 |
| xz exchange | -0.002443674 |
| interorbital density V | +0.003889288 |
| Hund | +0.001449413 |
| orbital splitting | +0.000708049 |
| interlayer hopping, interlayer exchange, chemical potential | 0 |
| total | +0.033694068 |

The component sums reproduce each full energy to 4.44e-16/Ni. The largest
difference is x hopping (about 81% of the gap); the two planar exchange
contributions nearly cancel and slightly favor the diagonal. This describes
these states, not the derivative of a phase boundary with respect to J, and
it must not be transferred to the opposite-sign Hamiltonian.

## Provenance

`cent_followup_same_sign_20261105.json` is the compact final archive. Complete
collection JSON/CSV and manifests remain under local ignored
`run/cent_ipeps_followup_20261105/` and remote
`~/LNOsuperconductivity/ipeps_u1_20261105/`. No old solver was overwritten.
The original solver binary is `2e17419c…c737f2` (source `b2cefd8`); the
measurement-only component binary is `386a3e59…68ee0d` (source `45cd939`).
These N-only simple-update screens compare sampled collinear states at finite
D, time and Chi; they do not establish the thermodynamic ground state.
