# Constrained exchange-search design, 2026-11-05

> Historical same-sign model (`TxzYSign=+1`, including omitted legacy keys).
> On 2026-11-07 the collaborator confirmed opposite x/y signs. The energies
> and candidates here must not be interpreted as results for that corrected model.

This note records the parameter and initialization design. It contains no
optimized-energy results or phase-boundary claim.

## Constraints and unchanged hopping

The revised search requires both directional inequalities

```
Jx1 > Jxz1,  Jx2 > Jxz2,  Jxz2 / Jxz1 > 0.25.
```

All eight points below also satisfy the stronger inequality `Jx2 > Jxz1`.
The hopping parameters stay at their supplied values:
`Tx1=0.403`, `Tx2=0.398`, `Txz1=0.211`, `Txz2=0.190`.
Thus the actual cross-orbital hopping ratio in the Hamiltonian is
`Txz2/Txz1=0.900473934` throughout this search.

The earlier exploratory candidate had `Jxz1=0.20`, `Jxz2=0.0067`, so its
exchange ratio was `0.0335`. Its square root is `0.183030052`. Interpreting
that number as an effective hopping ratio requires the additional assumption
that both exchanges obey the same proportionality constant and denominator
in `J = C t^2/U`. It is not the hopping ratio actually supplied to the code.
The original table has `Jxz2/Jxz1=0.696103896`, whose square root is
`0.834328410`; this too differs from the actual hopping ratio above. The
exchange scan therefore must be reported explicitly, rather than described
as a unique DFT or experimental determination.

The earlier candidate violates `Jx1 > Jxz1` as well as the ratio bound. The
old interpolation midpoint also violates both: it has
`Jx1=0.040625`, `Jxz1=0.11925`, and `Jxz2/Jxz1=0.140461216`.
Neither is an allowed point in this revised search.

## Eight-point grid

Exact values are taken from the unique physics objects in
`run/cent_ipeps_followup_20261105/constrained_manifest.json`.
The six `near_` points interpolate the four exchanges between the original
table and the earlier exploratory candidate at fractions
`lambda = 0, 0.05, 0.10`, then multiply all four AF exchanges by
`s = 1, 0.5`. Hund `JH=1`, `V=3`, the onsite splitting `0.558`, all hopping,
and the zero interlayer hopping/exchange remain unchanged. In particular,
scaling the AF exchanges does not scale the ferromagnetic Hund coupling.

| Point | Jx1 | Jx2 | Jxz1 | Jxz2 | Jxz2/Jxz1 |
|---|---:|---:|---:|---:|---:|
| `near_lam000_s100` | 0.065 | 0.0633 | 0.0385 | 0.0268 | 0.696103896 |
| `near_lam000_s050` | 0.0325 | 0.03165 | 0.01925 | 0.0134 | 0.696103896 |
| `near_lam005_s100` | 0.0625625 | 0.06092625 | 0.046575 | 0.025795 | 0.553837896 |
| `near_lam005_s050` | 0.03128125 | 0.030463125 | 0.0232875 | 0.0128975 | 0.553837896 |
| `near_lam010_s100` | 0.060125 | 0.0585525 | 0.05465 | 0.02479 | 0.453613907 |
| `near_lam010_s050` | 0.0300625 | 0.02927625 | 0.027325 | 0.012395 | 0.453613907 |
| `boundary_r030_s100` | 0.065 | 0.0633 | 0.0585 | 0.01755 | 0.3 |
| `boundary_r030_s050` | 0.0325 | 0.03165 | 0.02925 | 0.008775 | 0.3 |

The two boundary points instead use `Jxz1/Jx1=0.9` and
`Jxz2/Jxz1=0.3`, at the same two AF scales. They probe stronger anisotropy
inside the stated bounds; the bounds alone do not establish DFT consistency.
Along the old interpolation line, `Jx1 > Jxz1` restricts
`lambda < 0.126040428`, so the selected near-table fractions remain inside
that bound. The uniform positive scale preserves every inequality and ratio.

## Initial-state coverage

Each point has eight main starts on the same 4x4 rung cell:

- `diagonal_minus`, `SpinOffset=0,1`, `ChargePattern=stripe_x`;
- `diagonal_plus`, `SpinOffset=0,1`, `ChargePattern=stripe_x`;
- `stripe_x`, `SpinOffset=0`, each of `ChargePattern=stripe_x,stripe_y`;
- `stripe_y`, `SpinOffset=0`, each of `ChargePattern=stripe_x,stripe_y`.

All use `ChargeOffset=0` and initial `LayerAlignment=-1`. The initial states
have N=48 electrons, 24 per layer, and total Sz=0. The tensors conserve only
N during evolution; initial Sz=0 is not an additional imposed tensor symmetry.
The planned common D=3 screen uses fixed Chi=8 and the schedule
25 sweeps at tau=0.1 followed by 15 at tau=0.03, total imaginary time 2.95.
This tests selected collinear branches and relative phases, not all states
at each wavevector. Final peaks must be measured on both layers, using
transverse as well as longitudinal spin components when present.

Any promising point also needs FM, Neel and quarter-axis guards before it is
described as the lowest sampled branch. The quarter-axis `++--` seeds require
perpendicular charge stripes or a balanced relative phase if an initially
unpolarized comparison is intended. Lowering AF exchanges can also favor FM;
a diagonal-versus-axial crossing alone does not exclude that possibility.

## Product-state diagnostic

For the literal `build_seed_labels` conventions, let x=col and y=row, use
`ChargePattern=stripe_x`, `ChargeOffset=0`, `LayerAlignment=-1`, and compare
`diagonal_minus, SpinOffset=0` with `stripe_y, SpinOffset=0`.
The x electron occupies layer `floor(col/2)%2`, the diagonal spin sign is
`++--` in `(row-col)%4`, and the axial sign is `(-1)^row`. East bonds are
type 1 at even row+col; south bonds are type 1 at odd row+col.

Direct counting of the periodic bonds gives the following exchange-energy
coefficients **per Ni**, including both layers:

| Initial product state | Jx1 | Jx2 | Jxz1 | Jxz2 |
|---|---:|---:|---:|---:|
| Negative diagonal, phase 0 | -3/32 | +3/32 | -1/4 | +1/4 |
| Axial `(0,pi)` | -1/32 | -1/32 | 0 | 0 |

For example, the occupied horizontal x-x bonds include four bonds of each
type, while the occupied vertical x-x bonds include eight of each type.
The Sz-product signs differ between the two spin textures. Each bond also
has two x-z contributions because each endpoint has one x electron per rung.
The cell contains 32 Ni, which supplies the per-Ni normalization above.

Consequently the exact initial product-state gap is

```
E_diagonal_minus/Ni - E_0_pi/Ni
    = (-Jx1 + 2 Jx2)/16 - (Jxz1 - Jxz2)/4.
```

Both initial states have the same onsite/Hund contribution, 1.279/Ni for
the unchanged onsite parameters. Their hopping expectation is zero because
they are number-basis product states. The following checks apply to the
unscaled near-table points:

| lambda | Product E_diagonal_minus/Ni | Product E_0_pi/Ni | Product gap/Ni |
|---|---:|---:|---:|
| 0 | 1.275915625 | 1.274990625 | +0.000925 |
| 0.05 | 1.2736516015625 | 1.2751409765625 | -0.001489375 |
| 0.10 | 1.271387578125 | 1.275291328125 | -0.00390375 |

At scale s=0.5 these initial exchange-energy gaps are halved; the common
onsite/Hund energy is unchanged. This diagnostic shows that the revised
bounds do not forbid a lower diagonal **initial product energy**. It does
not predict an optimized PEPS phase boundary. Subsequent state relaxation
changes onsite, kinetic and exchange correlations; attributing any additional
gain to one component requires an explicit energy decomposition. Longer-time,
D and environment checks remain separate from this design calculation.
