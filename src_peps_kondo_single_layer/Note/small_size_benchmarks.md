# Small-size benchmarks (OBC) for single-layer Kondo lattice PEPS

This note is meant for **regression tests** of `simple_update`, `mc_measure`, and `vmc_optimize`.
All numbers below are **ground-state energies** for tiny OBC clusters in fixed \((N_e, S^z_{\mathrm{tot}})\) sectors.

## Model and conventions (must match code)

Hamiltonian (same as `src_peps_kondo_single_layer`):
\[
H = -t\sum_{\langle ij\rangle,\sigma}\left(c^\dagger_{i\sigma}c_{j\sigma} + \mathrm{h.c.}\right)
  +U\sum_i n_{i\uparrow}n_{i\downarrow}
  +J_K\sum_i \mathbf{s}_i\cdot \mathbf{S}_i
  -\mu\sum_i (n_{i\uparrow}+n_{i\downarrow}).
\]

- **Local Hilbert space**: 8D = itinerant electron (4D: \(|0\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle\)) ⊗ local spin (2D: \(|\Uparrow\rangle,|\Downarrow\rangle\)).
- **Kondo sign**: in this repo we use \(H_K = J_K\,\mathbf{s}\cdot\mathbf{S}\). So **ferromagnetic Kondo** corresponds to **\(J_K<0\)**.
- **Quantum number used for small ED tables**: \(S^z_{\mathrm{tot}}\) includes **both** itinerant electron and local spin.
  We print \(S^z_{2,\mathrm{tot}} \equiv 2 S^z_{\mathrm{tot}}\) (integer).

## Important special-case (don’t “fix” it)

For an even number of sites \(N\), local-spin contribution \(S^z_{2,\mathrm{loc}} = 2N_{\Uparrow}-N\) is **even**.
Electron contribution \(S^z_{2,\mathrm{e}} = N_\uparrow-N_\downarrow\) has the **same parity** as \(N_e\).

Therefore, on even-\(N\) clusters (like 2×2, 4×4), the sector \((S^z_{2,\mathrm{tot}}=0)\) exists **only for even \(N_e\)**.
Odd \(N_e\) gives an **empty sector** (this is physics, not a bug).

### Project restriction (current workflow)

For this PEPS Kondo project we currently **force**:

- **itinerant electron number** \(N_e\) is **even**
- **total** \(S^z_{\mathrm{tot}} = 0\)

So runs with odd \(N_e\) or odd \((L_xL_y)\) are rejected early.

## 2×2 OBC (exact ED, fixed \(S^z_{2,\mathrm{tot}}=0\))

Conventions: \(t=1\), \(\mu=0\).
**Cross-verified** by pure-Python ED (`tools/ed_kondo_small.py`) and QuSpin (`tools/quspin_ed_kondo_2x2.py`).

### Case A: Free fermion (U=0, JK=0)

Ground-state energies:

| Ne | E0 |
|---:|---:|
| 0 | 0.000000000000 |
| 2 | -4.000000000000 |
| 4 | -4.000000000000 |
| 6 | -4.000000000000 |
| 8 | 0.000000000000 |

### Case B: Hubbard only (U=4, JK=0)

| Ne | E0 |
|---:|---:|
| 2 | -3.418550718874 |
| 4 | -2.102748483462 |
| 6 | 4.581449281126 |

### Case C: Kondo + hopping (U=0, JK=-1, FM)

| Ne | E0 |
|---:|---:|
| 2 | -4.134228326066 |
| 4 | -4.550574840474 |
| 6 | -4.134228326066 |

### Case D: Kondo + hopping (U=0, JK=+1, AF)

| Ne | E0 |
|---:|---:|
| 2 | -4.266880456223 |
| 4 | -5.306178212449 |
| 6 | -4.266880456223 |

### Case E: Full parameters (t=1, U=4, JK=-1, FM)

| Ne | E0 |
|---:|---:|
| 4 | -2.811582631385 |

## 4×4 OBC Benchmarks

For 4x4 (16 sites), the Hilbert space dimension (\(8^{16} \approx 2.8 \times 10^{14}\)) is too large for Exact Diagonalization.
We recommend the following benchmarks:

### 1. Free Fermion (Analytic, U=JK=0)

This tests the hopping term and lattice geometry implementation.
Noninteracting **spinful** electrons on an OBC 4×4 square with \(t=1\), \(\mu=0\).
Local spins are completely decoupled.

| Ne | E0 |
|---:|---:|
| 0 | 0.000000000000 |
| 8 | -17.888543819998 |
| 16 | -21.888543819998 |
| 24 | -17.888543819998 |
| 32 | -0.000000000000 |

### 2. Atomic limits (t=0)

Closed-form energies (any size).

**Only-U (JK=0, t=0)**:
\[ E = U\,N_d -\mu\,N_e \]
where \(N_d\) is number of doublons.

**Only-JK (U=0, t=0, μ=0)**:
Singly occupied sites contribute \(-\frac{3}{4}J_K\) (AF) or \(+\frac{1}{4}J_K\) (FM).
For 4x4, \(N=16\).
If \(N_e \le 16\), \(N_{\mathrm{singly}} = N_e\).
For \(N_e=16\), \(E_0(J_K=-1) = \frac{1}{4}(-1)(16) = -4.0\).

### 3. DMRG Reference

For interacting cases (e.g. \(U=4, J_K=-1\)), use DMRG on a 4x16 or 4x4 cylinder/strip.
4x4 OBC can be reliably solved by DMRG (bond dimension \(m \approx 2000\)).

## How to reproduce 2x2 ED

Run:

```bash
python3 src_peps_kondo_single_layer/tools/ed_kondo_small.py --print_default_tables
```

Or for QuSpin cross-check (requires conda env with `quspin`):

```bash
python src_peps_kondo_single_layer/tools/quspin_ed_kondo_2x2.py --case full
```
