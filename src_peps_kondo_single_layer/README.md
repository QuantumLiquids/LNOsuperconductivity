## Single-layer Kondo lattice with PEPS (OBC square lattice)

This directory is a **new PEPS-based implementation skeleton** for the **single-layer Kondo lattice** model
used in this project (the \(J_\perp = 0\) limit of the two-layer model in `Note/main_final.tex`).

The goal is to make the **conventions unambiguous** (local basis, fermion parity, signs), so later work
(simple update details, measurements, and \(t_2\) anisotropy) won't silently break physics.

### 1) Model definition (conventions)

We use the single-layer Hamiltonian

\[
H = - \sum_{\langle i,j\rangle,\sigma} t_{ij}\,(c^\dagger_{i\sigma} c_{j\sigma} + h.c.)
    + U\sum_i n_{i\uparrow} n_{i\downarrow}
    + J_K \sum_i \mathbf{s}_i \cdot \mathbf{S}_i
    - \mu \sum_i n_i.
\]

- **Itinerant electron**: spinful Hubbard electron on each site (4 states).
- **Localized spin**: spin-1/2 local moment on each site (2 states).
- Total on-site Hilbert space dimension: **\(4\times 2 = 8\)**.

#### Sign convention for the Kondo/Hund coupling

Your existing DMRG/VMPS code (`src_kondo_zigzag_ladder/*`) implements:

- **\(H_K = J_K\,\mathbf{s}\cdot\mathbf{S}\)** (positive \(J_K\) is **AFM**, negative \(J_K\) is **FM**).

In the paper draft (`Note/main_final.tex`) the coupling is written as \(-J_H\,\mathbf{s}\cdot\mathbf{S}\) with \(J_H>0\) for FM Hund's coupling.
Therefore the mapping is:

\[
J_K = -J_H.
\]

### 2) Local basis (8 states) — fixed encoding

We encode the combined local basis as **electron state × local spin**.

#### Electron basis (4 states, fixed order)

We use the Hubbard-like order:

- `E_D` = \(|\uparrow\downarrow\rangle\)
- `E_U` = \(|\uparrow\rangle\)
- `E_d` = \(|\downarrow\rangle\)
- `E_0` = \(|0\rangle\)

with the standard internal fermionic ordering **(up, then down)**.
This fixes the on-site signs like:
- \(c_{\downarrow}|\uparrow\downarrow\rangle = -|\uparrow\rangle\),
- \(c^\dagger_{\downarrow}|\uparrow\rangle = -|\uparrow\downarrow\rangle\).

#### Local spin basis (2 states, fixed order)

- `S_U` = \(|\Uparrow\rangle\)
- `S_d` = \(|\Downarrow\rangle\)

#### Combined 8-state encoding (single integer label 0..7)

We map:

```
index = 2 * electron + local_spin

electron: 0(E_D), 1(E_U), 2(E_d), 3(E_0)
spin    : 0(S_U), 1(S_d)
```

So:

- 0: \(|\uparrow\downarrow\rangle \otimes |\Uparrow\rangle\)
- 1: \(|\uparrow\downarrow\rangle \otimes |\Downarrow\rangle\)
- 2: \(|\uparrow\rangle \otimes |\Uparrow\rangle\)
- 3: \(|\uparrow\rangle \otimes |\Downarrow\rangle\)
- 4: \(|\downarrow\rangle \otimes |\Uparrow\rangle\)
- 5: \(|\downarrow\rangle \otimes |\Downarrow\rangle\)
- 6: \(|0\rangle \otimes |\Uparrow\rangle\)
- 7: \(|0\rangle \otimes |\Downarrow\rangle\)

### 3) Fermion structure

Only the **itinerant electron** contributes to fermion parity.

- **Fermion parity** \(p = N_e \bmod 2\).
- **Total \(S^z\)** includes both electron and local spin.

We expose this via `TENSOR_SYMMETRY_LEVEL` in `qldouble.h`:

- 0: fermion parity only (`fZ2QN`)
- 3: fermion parity + total \(S^z\) (`fZ2U1QN`) — recommended default
- 2: \(N\) + \(S^z\) (`fU1U1QN`) — optional

### 4) Lattice and anisotropy \(t_{ij}\)

We target a **square lattice with open boundary conditions (OBC)**.

- **Baseline**: uniform NN hopping \(t_{ij}=t\).
- **Future extension**: a NN bond-order pattern \(t_{ij}\in\{t, t_2\}\) corresponding to a \((\pi,\pi)\) bond order (see Fig.1(b) in the draft paper).

Implementation note: `qlpeps::SquareLatticeNNSimpleUpdateExecutor` assumes a **uniform NN bond Hamiltonian**.
To support two NN bond types \((t,t_2)\) we will add a local executor in this repo (do not patch the upstream `PEPS` repo).

### 5) Build / run

This repo must be able to find the `qlpeps` headers (from the `PEPS` project) on your include path.

- Build target: `peps_kondo_square_simple_update`
- Run (two-json style, similar to `finite-size_PEPS_tJ`):

```bash
./peps_kondo_square_simple_update params/physics_params.json params/simple_update_algo.json
```


