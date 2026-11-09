# Zero-interlayer single-layer iPEPS comparison

For TzPerp=JzPerp=0, the projected Hamiltonian separates into two copies
of the same single-layer model. We fix each copy to 1.5 electrons per Ni.
The new target uses one nine-state Ni per tensor, retaining both orbitals,
the original Hund convention and the opposite x/y cross-orbital hopping
signs. The ansatz conserves N, with N=24 on the 4×4 cell.

The numerical source is cdbda0aee7f4c6d637457870b5880d8ac3e323c3.
The independent Fock test agrees exactly for the Hamiltonian matrices.
The graded imaginary-time gates and decoupled bilayer factorization
agree to 1.34e-15, including nonempty spectator states in the other layer.
A native rotated axial pair at T=0.4 agrees within 3.55e-8/Ni and
retains the appropriate wavevectors. Frozen checkpoint replay agrees to
1e-12. The rebuilt bilayer driver reproduces the earlier smoke energy
and output keys on the unchanged checkpoint. The single-layer target
rejects inputs with nonzero layer coupling.

All comparisons use the 4×4 cell, T=2.95 from 25 steps at tau=0.1 and
15 at tau=0.03, and chi=32 with CTM tolerance 1e-5. The dimer starts are
uniform at each Ni, n_z=1 and n_x=0.5, while local occupations can relax.
The rotated runs transform the spin pattern, dimer covering and sweep
order together under (row,col) -> (col+1,row), with the global orbital
gauge that restores the cross-orbital signs. The q-minus starts include
both strong-chain spin phases. Each run starts fresh, with no old state
or gate cache shared across executables.

The first manifest contains the original table and the previous r=.30
exchange point at D=3 and D=4, with additional FM, Neel and quarter-axis
guards at the original table. The second manifest repeats the remaining
two historical exchange points at D=3. The campaign ran on macOS with
one numerical thread per case and up to four concurrent cases. Cent
had only 576 MiB of schedulable memory at the time of the check.

## Original-table energies and density

Each gap is relative to the lower measured period-two axial energy for
that D. The quarter-axis value takes the lower of its two orientations.

| Final wavevector | D3 gap/Ni | D4 gap/Ni |
|---|---:|---:|
| (pi/2,−pi/2) | +0.013086464 | +0.012821274 |
| (pi/2,pi/2) | +0.011056549 | +0.014230537 |
| (pi/2,0) or (0,pi/2) | +0.008682659 | +0.006726649 |
| (0,0) | +0.021444993 | +0.017699334 |
| (pi,pi) | +0.078354706 | +0.077178017 |

The lowest measured axial energies are 0.931506744/Ni and
0.925778690/Ni at D3 and D4. Their rotated partners differ by
9.94e-6/Ni and 2.50e-8/Ni, respectively. The lowest diagonal changes
from the plus wavevector at D3 to the minus wavevector at D4.
The minimum diagonal-to-axial gap changes from +0.011056549 to
+0.012821274/Ni. The sampled quarter-axis state is closer to the axial
minimum at both dimensions.

At D4 the lowest minus-diagonal state has n in
1.499999966–1.499999994 over all 16 Ni sites. Its orbital occupations
are n_x=0.527056045–0.527056076 and n_z=0.972943915–0.972943921.
The Sz Fourier amplitude is 0.500943 at the same diagonal wavevector.
The lowest axial state has n in 1.499999988–1.500000005, and the
lowest plus-diagonal state has n in 1.499553–1.500447. For comparison,
the earlier uniform-seed D3 bilayer minus-diagonal state had Ni
occupations 1.410875–1.589125. The single-layer ansatz and its rotated
initializers both differ from the earlier bilayer screening protocol.

## Exchange controls

| Exchange point | D | Axial energy/Ni | Minus-diagonal gap/Ni | Plus-diagonal gap/Ni |
|---|---:|---:|---:|---:|
| Original table | 3 | 0.931506744 | +0.013086464 | +0.011056549 |
| Near-table interpolation | 3 | 0.931582179 | +0.017111588 | +0.011046886 |
| Jxz2/Jxz1=0.30 | 3 | 0.931444837 | +0.019617181 | +0.011081678 |
| All planar J halved | 3 | 0.932619212 | +0.011542400 | +0.010874480 |
| Original table | 4 | 0.925778690 | +0.012821274 | +0.014230537 |
| Jxz2/Jxz1=0.30 | 4 | 0.925616673 | +0.019429526 | +0.014234454 |

The exchange-only comparisons keep all nonexchange parameters and the
initialization protocol fixed. Their D3 differences are small for the
best diagonal branch. The minus-diagonal minimum uses the spin phase
with parallel local moments on all 16 type-1 bonds and antiparallel
moments on all 16 type-2 bonds at original-table D4.

## Measurement coverage

All 56 cases completed the 40-step evolution. At chi=32, 55 final
measurements converged. The original-table D4 rotated minus-diagonal
phase-0 case exhausted 200 CTMRG sweeps. The optional 500-sweep retry
was stopped, and a separate frozen chi=24 measurement on the same
checkpoint exhausted its 80-sweep cap. The state and logs are retained.
Its energy is blank in the comparison. Both orientations of the
lower-energy minus-diagonal phase converged at chi=32.

The [case table](single_layer_20261109.csv) records seed controls,
measured wavevectors, density ranges and checkpoint hashes. Complete
native measurements, manifests and plots are in local ignored
run/ipeps_single_layer_20261109/. The energies compare the sampled
finite-D simple-update states at the same T and chi.
