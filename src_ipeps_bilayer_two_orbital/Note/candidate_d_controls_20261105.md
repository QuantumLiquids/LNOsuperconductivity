# Short D controls at the exchange candidate, 2026-11-05

This is a paired D=2 versus D=3 test at the candidate parameters
Jx1=0.01625, Jx2=0.015825, Jxz1=0.2, Jxz2=0.0067. Other couplings match
the supplied table, including JH=1 and zero interlayer hopping/exchange.
The literal target in this test is q=(pi/2,-pi/2), not the positive diagonal.

Each D starts from the same coarse D=2 checkpoint for its branch under
`run/bilayer_exchange_candidate_d2_20261105/jxz1_020_jxz2_00067/`:

- `diagonal_minus_spin0_charge_stripe_x_shift0/checkpoint_step00000040`.
- `stripe_x_spin0_charge_stripe_y_shift0/checkpoint_step00000040`.

Those cold runs had total imaginary time 2.95. Each paired continuation adds
exactly two sweeps at tau 0.03, or 0.06 extra time. Fixed Chi 8, tolerance 1e-6,
min 2/max 100 CTMRG sweeps and one numerical thread are shared. No Chi sweep
was performed. Dmin=1 and truncation tolerance 1e-12 are unchanged.
The immutable tau 0.03 gates were copied into a private cache to avoid any
interaction with the main continuation's cache writes.

| D | Negative-diagonal E/Ni | Axial E/Ni | Diagonal minus axial |
|---|---:|---:|---:|
| 2 | 1.114931559995 | 1.124669200659 | -0.009737640664 |
| 3 | 1.105962605521 | 1.116075407416 | -0.010112801894 |

Both sampled branches decrease in measured energy. D=3 changes the gap by
-0.000375161231/Ni relative to D=2, slightly increasing the negative-diagonal
advantage in this very short candidate test. In contrast, the matched short
controls at the original couplings slightly favored the axial branch when
D increased. Thus the observed D dependence is parameter dependent; the
results do not establish a general rule that larger D favors diagonal order.

All bonds reach the specified D. Both layers keep q=(pi/2,-pi/2) in the
diagonal branch and q=(pi,0) in the axial branch. The exact tensor sector is
N=48,Sz=0 in all four runs. The maximum measured density deviation from
1.5/Ni is 1.165e-09. Full-precision energies, layer densities, separate
layer Fourier amplitudes and checkpoint paths are in the accompanying CSV.

## Costs and artifacts

| D | Branch | Total seconds | Two sweep seconds | Final CTMRG seconds |
|---|---|---:|---|---:|
| 2 | diagonal_minus | 1.88 | 0.12, 0.11 | 0.75 |
| 2 | stripe_x | 2.80 | 0.25, 0.23 | 1.26 |
| 3 | diagonal_minus | 5.30 | 0.31, 0.83 | 3.35 |
| 3 | stripe_x | 5.25 | 0.40, 0.98 | 2.95 |

No 10-second D3 local-sweep cap was triggered. Raw data, input JSON, the
controller script and immutable cached gates are under
`run/bilayer_candidate_d_control_20261105/`. All four jobs completed and no
simulation process remains active. This short test is separate from the
main candidate continuation to total time 5.15; those longer-time energies
must not be compared directly with this D-control table.

Executable SHA256:
`a8c87fec30539ea91b51d1cb8803b933b7c38c2b77c16a4ad1591b97cdb7928f`.
Physics JSON SHA256:
`1819ee117facdcb4309d5bf05d3f33c989d4ecf72fc6b2a01c3a8fc78f6a1361`.
Gate cache metadata records build
`Nov  5 2026 16:58:38`.
