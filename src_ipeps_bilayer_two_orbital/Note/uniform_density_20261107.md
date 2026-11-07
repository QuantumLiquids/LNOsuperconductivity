# Uniform initial density in the projected bilayer iPEPS

The preceding opposite-sign scan used a period-four distribution of the
initial $x$ electrons between the layers. Its product states had three
electrons on every rung and different occupations on the two Ni sites of
that rung. See [the prior scan](opposite_txz_20261107.md).

The new N-only initializer uses one $x$ electron in an equal-weight
superposition on each intralayer nearest-neighbor dimer. The two layers'
coverings are displaced by one bond. Each virtual bond carries at most one
dimer, so the exact initial maximum bond dimension is 2. The initial
occupations on every Ni are $n_z=1$ and $n_x=0.5$, and each layer initially
has exactly 24 electrons on the $4\times4$ cell. The spin of the occupied
$x$ orbital follows the local $z$ moment. A dimer across opposite
moments can superpose different $S^z$ sectors. The two dimer orientations
share the same site densities and have different initial bond correlations.
All local occupations can relax under the physical Hamiltonian.

The N-only and default targets build on macOS. Four native N-only checks
covered both directions, both covering offsets, and a smaller quarter-axis
cell. Initial site-density and spin expectations agreed with their exact
values to $9\times10^{-16}$. An independent evaluation of the seed
Hamiltonian energy agreed with CTMRG to $2\times10^{-15}$ per Ni.
Each check completed a nonzero imaginary-time step.

The cent comparison uses the original table with TxzYSign=-1, the
$4\times4$ rung cell, $D_{\max}=3$, $\chi=32$, and CTM tolerance
$10^{-5}$. The 40-step schedule is $25\times0.1+15\times0.03$, giving
$T=2.95$. Both dimer orientations are used for each of the two diagonal
and two axial seeds, plus the $(0,\pi/2)$ seed. The wrapper measures the
initial state and checks its site-resolved occupations before accepting a
result. Final phases are grouped by the measured spin wavevectors.

The numerical source is a44a24cd39ed05209f54cde0cf2e21210ca1969f.
The isolated cent deployment is
/home/gzcgu/haoxinwang/LNOsuperconductivity/ipeps_uniform_density_20261107/.
The manifest can be recreated with cluster/prepare_uniform_density.py.
Build 168724 passed the N-only model validation. The solver SHA256 is
b37ff7335ddd754d0682ce8b492ab3e5c633777f43e23b245d1ba5d0f068595c.
Array 168725 uses one physical core, no SMT, and 2 GiB per case with up
to five concurrent cases.

## Checkerboard symmetry control

The first two coverings select a dimer from the coordinate parity along
its direction. This breaks the checkerboard's even-sum translations in
each layer. The checkerboard control places the top-layer dimers on H1
and the bottom-layer dimers on H2 and uses a grouped H1, H2, V1, V2
sweep. Each group is a perfect matching. Even-sum translations and
bond-centered inversion, with a global spin flip where needed, relate
all sites in each of the phase-zero diagonal and axial seeds. The exact
gates within a group preserve those symmetries. Finite-D truncation can
still break them, including when it cuts a degenerate Schmidt subspace.
The calculation evolves the same Hamiltonian without site-dependent
chemical potentials or constraints on the measured local density.

All four short local tests with this protocol had maximum per-Ni
deviation below 0.01 at $T=0.5$. They covered both covering directions
and both offsets. The cent control uses the original table and the same
measurement and evolution settings as the first comparison, with one
checkerboard x covering for each of the four main magnetic seeds. Its
source is e231ad7e1fece20a257ccee573c8ec978632f956 and its build job is
168737. The separate deployment is
/home/gzcgu/haoxinwang/LNOsuperconductivity/ipeps_uniform_symmetry_20261107/.
The manifest uses the --checkerboard option of
cluster/prepare_uniform_density.py.

## Finite-time density and energy

All 14 cent initial states passed the site-resolved density check. The
largest deviation from the target occupations was below $9\times10^{-16}$.
Four of ten coordinate-parity cases produced final measurements at
$\chi=32$. Five other cases exhausted the 200- and 500-sweep CTMRG
attempts, and the quarter-axis y-dimer case was killed by the 2 GiB
memory limit. All ten completed the 40-step evolution and retained
their final checkpoints. A failed measurement has no assigned energy.

All four checkerboard cases in array 168738 completed the final
contraction and retained their input wavevectors in both layers.
The solver SHA256 is
ef368c8d9c6dd97c5ca348bc0a5e38def50927716f8d17940e559556503a756a.
For each case the table pools the local total occupations
$n_i=n_{x,i}+n_{z,i}$ of 32 Ni sites. The mean occupation of each layer
agrees with 1.5 to within $6\times10^{-9}$.

| Final wavevector | E/Ni | Local Ni density min–max | RMS of n−1.5 | Max of abs(n−1.5) |
|---|---:|---:|---:|---:|
| (pi/2,−pi/2) | 1.027620098 | 1.410875–1.589125 | 0.083860 | 0.089125 |
| (pi/2,pi/2) | 1.037554793 | 1.404605–1.596201 | 0.073061 | 0.096201 |
| (pi,0) | 1.049496155 | 1.387631–1.612369 | 0.096280 | 0.112369 |
| (0,pi) | 1.004311559 | 1.425125–1.574875 | 0.064778 | 0.074875 |

The largest local deviation is smaller than in the measured
coordinate-parity starts at the same q, where there is a counterpart.
The RMS deviation changes little for the two diagonals and increases
for (0,pi) from 0.050139 to 0.064778. Thus the grouped control reduces
the extremes while retaining charge modulation. For the minus
diagonal, the dominant component at (pi,pi) is +0.089125 and −0.078241
electron per Ni in the two layers. Their sum leaves the rung
occupations in 2.989116–3.010884.

The minus diagonal lies 0.023308540/Ni above the lowest measured axial
state in this control. The two axial wavevectors are symmetry-related
in the Hamiltonian, yet their energies differ by 0.045184597/Ni with
this one x-directed covering and finite-time grouped sweep. This
initialization and optimization dependence limits the phase comparison.
The site-density measurement concerns the finite-D evolved states.
It cannot determine either spontaneous charge order or a charge gap.

The memory-limited case was recovered in cent job 168742_0 by measuring the
same final checkpoint at Chi32, with 4 GiB and one physical core. The
checkpoint hash was unchanged and no additional evolution was performed.
The final wavevector is (0,pi/2), with E/Ni=1.062370419.
It is higher than the measured x-covering quarter-axis case.

## Local D2 comparison

The same checkerboard protocol was run on macOS at Dmax=2, T=2.95 and
Chi32 for the axial y seed with x dimers and both diagonals with each
dimer orientation. All five retained their spin wavevectors in both layers.
The density range is over all 32 Ni sites.

| Final wavevector | Covering | E/Ni | Local Ni density min–max | RMS of n−1.5 |
|---|---|---:|---:|---:|
| (0,pi) | x | 1.037781433 | 1.499358–1.500642 | 0.000454 |
| (pi/2,−pi/2) | x | 1.063225260 | 1.384493–1.621901 | 0.060117 |
| (pi/2,−pi/2) | y | 1.062983134 | 1.408363–1.559020 | 0.049903 |
| (pi/2,pi/2) | x | 1.066846081 | 1.415354–1.589601 | 0.048683 |
| (pi/2,pi/2) | y | 1.066289561 | 1.404865–1.560565 | 0.048578 |

The local D2 (0,pi) state is nearly uniform in total Ni occupation.
Both diagonals develop density modulation with either covering.
The flatter axial density at D2 is specific to this finite-D result.

The accompanying CSV contains the density metrics at the initial and
final times, with matched old D3 energies by measured q. Unavailable
measurements have empty fields. The frozen recovery is recorded above
and leaves the original OOM record intact. Full collections, manifests,
and the site-density maps are in local ignored
run/ipeps_uniform_density_20261107/.
