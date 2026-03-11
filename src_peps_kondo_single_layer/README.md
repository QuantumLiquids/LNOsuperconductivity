# PEPS Single-Layer Kondo Lattice on Square Lattice

This directory implements a single-layer Kondo lattice model solved with the
PEPS (Projected Entangled Pair States) tensor-network method. The three-stage
pipeline — Simple Update, VMC Optimization, Monte Carlo Measurement — is
designed to study the same physics as the DMRG zigzag-ladder code
(`src_kondo_zigzag_ladder/`) but on an untilted square lattice with standard
OBC.

**Status:** Work in progress. Verified on 2×2 (exact summation). Not yet
production-tested on large lattices.

## Hamiltonian

$$H = H_{\text{hop}} + H_{\text{onsite}} + H_{\text{Kondo}}$$

### Checkerboard NN hopping (zigzag chains on square lattice)

$$H_{\text{hop}} = -\sum_{\langle i,j \rangle, \sigma} t_{ij} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right)$$

The hopping amplitude $t_{ij}$ is **not** uniform — it follows a checkerboard
pattern that encodes zigzag chains on the square lattice. Define the sublattice
parity of site $(r, c)$ as:

$$p(r,c) = (r + c) \bmod 2$$

Then the bond hopping amplitudes are:

| Bond direction | $p = 0$ | $p = 1$ |
|----------------|---------|---------|
| Horizontal $(r,c) \to (r,c{+}1)$ | $t$ (intra-chain) | $t_2$ (inter-chain) |
| Vertical $(r,c) \to (r{+}1,c)$ | $t_2$ (inter-chain) | $t$ (intra-chain) |

If $t_2$ is not set (or set to 0), it defaults to $t$ (isotropic square
lattice, no zigzag structure).

**ASCII diagram** (4×4 lattice, `—` = horizontal, `|` = vertical):

```
  (0,0)——t——(0,1)——t2——(0,2)——t——(0,3)
    |          |          |          |
   t2         t          t2         t
    |          |          |          |
  (1,0)——t2——(1,1)——t——(1,2)——t2——(1,3)
    |          |          |          |
    t         t2          t         t2
    |          |          |          |
  (2,0)——t——(2,1)——t2——(2,2)——t——(2,3)
    |          |          |          |
   t2         t          t2         t
    |          |          |          |
  (3,0)——t2——(3,1)——t——(3,2)——t2——(3,3)
```

Tracing the intra-chain ($t$) bonds forms zigzag chains running diagonally:

```
Chain 0: (0,0) →h→ (0,1) →v→ (1,1) →h→ (1,2) →v→ (2,2) →h→ (2,3) →v→ (3,3)
Chain 1: (1,0) →v→ (2,0) →h→ (2,1) →v→ (3,1) →h→ (3,2)
Chain 2: (0,2) →h→ (0,3) →v→ (1,3)
  ...
```

### On-site terms

$$H_{\text{onsite}} = U \sum_i n_{i\uparrow} n_{i\downarrow} - \mu \sum_i n_i$$

- $U$: Hubbard repulsion on itinerant electrons.
- $\mu$: chemical potential (optional, default 0).

### Kondo coupling

$$H_{\text{Kondo}} = J_K \sum_i \mathbf{s}_i \cdot \mathbf{S}_i
= J_K \sum_i \left[ s^z_i S^z_i + \frac{1}{2}\left(s^+_i S^-_i + s^-_i S^+_i\right) \right]$$

- $\mathbf{s}_i$ = itinerant electron spin, $\mathbf{S}_i$ = localized spin-1/2.
- **Sign convention:** $J_K < 0$ = ferromagnetic Hund's coupling (FM).
- Mapping to paper notation: $J_K = -J_H$ where $J_H > 0$ is the Hund's coupling.

### Local Hilbert space

8-dimensional per site = (4 electron states) × (2 local spin states).
Combined index: $\text{idx} = 2 \times e + s$, where $e \in \{0,1,2,3\}$, $s \in \{0,1\}$.

| Index | Electron | Local spin | State |
|-------|----------|------------|-------|
| 0     | $\|\!\uparrow\downarrow\rangle$ (doublon) | $\|\!\Uparrow\rangle$ | $\|\!\uparrow\downarrow,\Uparrow\rangle$ |
| 1     | $\|\!\uparrow\downarrow\rangle$ (doublon) | $\|\!\Downarrow\rangle$ | $\|\!\uparrow\downarrow,\Downarrow\rangle$ |
| 2     | $\|\!\uparrow\rangle$ | $\|\!\Uparrow\rangle$ | $\|\!\uparrow,\Uparrow\rangle$ |
| 3     | $\|\!\uparrow\rangle$ | $\|\!\Downarrow\rangle$ | $\|\!\uparrow,\Downarrow\rangle$ |
| 4     | $\|\!\downarrow\rangle$ | $\|\!\Uparrow\rangle$ | $\|\!\downarrow,\Uparrow\rangle$ |
| 5     | $\|\!\downarrow\rangle$ | $\|\!\Downarrow\rangle$ | $\|\!\downarrow,\Downarrow\rangle$ |
| 6     | $\|0\rangle$ (empty) | $\|\!\Uparrow\rangle$ | $\|0,\Uparrow\rangle$ |
| 7     | $\|0\rangle$ (empty) | $\|\!\Downarrow\rangle$ | $\|0,\Downarrow\rangle$ |

Defined in `qldouble.h`. Electron basis ordering: `E_D(0)` = doublon, `E_U(1)` = spin-up,
`E_d(2)` = spin-dn, `E_0(3)` = empty. Local spin: `S_U(0)` = Up, `S_d(1)` = Dn.

### Fermion structure

Only the **itinerant electron** carries fermion parity. The localized spin is bosonic.

- Fermion parity: $p = N_e \bmod 2$.
- Total $S^z$ includes both electron and local spin contributions.

## PEPS ↔ DMRG Coordinate Mapping

> **Important for plotting:** The PEPS and DMRG codes use different coordinate
> conventions. Results must be transposed when comparing figures.

### The mapping

The DMRG code (`src_kondo_zigzag_ladder/tilted_zigzag_lattice.h`) uses a
45-degree tilted lattice with coordinates $(y, x)$ where $y$ is the chain index
and $x$ is the position along the chain. The `ElectronCoord` function maps
these to real-space plotting coordinates $(c_x, c_y)$:

$$c_x = \lfloor x/2 \rfloor + y, \qquad
c_y = \begin{cases}
\lfloor x/2 \rfloor - y & \text{if } x \text{ even} \\
\lfloor x/2 \rfloor + 1 - y & \text{if } x \text{ odd}
\end{cases}$$

Tracing DMRG chain $y=0$ through this mapping gives:

| DMRG $(y,x)$ | Real-space $(c_x, c_y)$ | PEPS $(r, c)$ |
|---------------|------------------------|----------------|
| $(0, 0)$      | $(0, 0)$               | $(0, 0)$       |
| $(0, 1)$      | $(0, 1)$               | $(0, 1)$       |
| $(0, 2)$      | $(1, 1)$               | $(1, 1)$       |
| $(0, 3)$      | $(1, 2)$               | $(1, 2)$       |
| $(0, 4)$      | $(2, 2)$               | $(2, 2)$       |
| $(0, 5)$      | $(2, 3)$               | $(2, 3)$       |

So the site-level mapping is:

$$\boxed{\text{PEPS row} = c_x, \qquad \text{PEPS col} = c_y}$$

### The reflection (transposition)

The DMRG plotting convention uses $c_x$ as the **horizontal** axis and $c_y$ as
the **vertical** axis. The natural PEPS plotting convention uses col as
horizontal and row as vertical. Since $c_x = \text{row}$ and $c_y = \text{col}$:

$$\text{DMRG horizontal axis} = \text{PEPS vertical axis (row)}$$
$$\text{DMRG vertical axis} = \text{PEPS horizontal axis (col)}$$

**This is a reflection (transposition) about the main diagonal.** The zigzag
chains run in the same diagonal direction in both codes, but the plotting axes
are swapped.

### Practical plotting guide

To make PEPS figures consistent with existing DMRG figures
(e.g., `plot/kondo_coupled_zigzag_chain/`):

```python
# PEPS data is indexed as data[row][col]
# To match DMRG plotting convention, swap axes:

# Option 1: Transpose the data
plt.imshow(data.T, origin='lower')
plt.xlabel('row (= DMRG $c_x$)')
plt.ylabel('col (= DMRG $c_y$)')

# Option 2: Plot with swapped coordinates
for row in range(Ly):
    for col in range(Lx):
        # DMRG-style plot: x = row, y = col
        plt.scatter(row, col, ...)
```

Alternatively, the DMRG `index_to_coord` function
(`plot/kondo_coupled_zigzag_chain/plot_spin_corr_pbc_two_panel.py`) returns
`(x_phys, y_phys) = (c_x, c_y)` which equals `(PEPS row, PEPS col)`.

## Pipeline

### Step 1: Simple Update (imaginary time evolution)

Initializes the PEPS wavefunction by applying imaginary-time evolution gates.

```bash
./peps_kondo_square_simple_update physics_params.json simple_update_algo.json
```

**Output:** `peps/` (raw PEPS tensors) and `tpsfinal/` (SplitIndexTPS for VMC).

When $t \neq t_2$, the Simple Update uses exact per-bond Hamiltonians (not averaging).
This was implemented by extending the upstream `SquareLatticeNNSimpleUpdateExecutor`
with a constructor accepting `TenMatrix<Tensor>` for non-uniform bond Hamiltonians.

### Step 2: VMC Optimization

Variationally optimizes the PEPS wavefunction using stochastic methods.

```bash
mpirun -np <N> ./peps_kondo_square_vmc_optimize physics_params.json vmc_algorithm_params.json
```

**Input:** reads from `tpsfinal/`. **Output:** writes optimized TPS back to `tpsfinal/`.

### Step 3: Monte Carlo Measurement

Measures energy and observables by VMC sampling.

```bash
mpirun -np <N> ./peps_kondo_square_mc_measure physics_params.json mc_measure_algorithm_params.json
```

**Input:** reads from `tpsfinal/`. **Output:** measurement results in working directory.

### Z2-only variants

For faster runs with reduced symmetry (fermion parity only, no $S^z$ conservation):

```bash
./peps_kondo_square_simple_update_z2 ...
mpirun -np <N> ./peps_kondo_square_vmc_optimize_z2 ...
mpirun -np <N> ./peps_kondo_square_mc_measure_z2 ...
```

### Exact summation (small-lattice debugging)

For 2×2 lattices, exact contraction replaces Monte Carlo:

```bash
./peps_kondo_2x2_exact_sum_optimize physics_params.json vmc_algorithm_params.json
./peps_kondo_2x2_exact_sum_optimize_z2 ...
```

## JSON Parameter Reference

### `physics_params.json`

```jsonc
{
    "Lx": 4,                // lattice width (number of columns)
    "Ly": 4,                // lattice height (number of rows)
    "t": 1.0,               // intra-chain hopping (zigzag)
    "t2": 0.3,              // inter-chain hopping (zigzag); omit or 0 for isotropic
    "U": 14.0,              // Hubbard U
    "Jk": -4.0,             // Kondo coupling J_K (FM = negative)
    "Mu": 0.0,              // chemical potential (optional)
    "ElectronNum": 4,       // total itinerant electrons (must be even)
    "ElectronSz2": 0        // 2 * Sz of itinerant electrons (integer)
}
```

**Notes:**
- `ElectronNum` must be even (restricted sector).
- $L_x \times L_y$ must be even (for $S^z_{\text{total}} = 0$ compatibility).
- If `t2` is omitted or 0, it defaults to `t`.
- JSON key for Kondo coupling is `Jk` (capital J, lowercase k).

### `simple_update_algo.json`

```jsonc
{
    "Dmin": 2,              // minimum bond dimension
    "Dmax": 8,              // maximum bond dimension
    "Tau": 0.01,            // imaginary time step
    "Step": 1000,           // number of SU steps
    "TruncErr": 1e-10,      // SVD truncation threshold
    "ThreadNum": 4          // OpenMP threads (optional)
}
```

### `vmc_algorithm_params.json`

```jsonc
{
    "OptimizerType": "SR",       // "SR" (stochastic reconfiguration), "SGD", "Adam", "AdaGrad"
    "MaxIterations": 200,
    "LearningRate": 0.01,

    // BMPS contraction
    "Db_min": 4,
    "Db_max": 16,
    "TruncErr": 1e-10,
    "MPSCompressScheme": 0,
    "ThreadNum": 4,

    // Monte Carlo sampling
    "MC_samples": 1000,
    "WarmUp": 200,
    "MCLocalUpdateSweepsBetweenSample": 10,

    // Sector control
    "ElectronNum": 4,
    "ElectronSz2": 0,

    // IO
    "WavefunctionBase": "tps",

    // SR-specific (if OptimizerType = "SR")
    "CGMaxIter": 100,
    "CGTol": 1e-8,
    "CGDiagShift": 0.01
}
```

### `mc_measure_algorithm_params.json`

```jsonc
{
    // BMPS contraction
    "Db_min": 4,
    "Db_max": 16,
    "TruncErr": 1e-10,
    "MPSCompressScheme": 0,
    "ThreadNum": 4,

    // Monte Carlo sampling
    "MC_samples": 5000,
    "WarmUp": 500,
    "MCLocalUpdateSweepsBetweenSample": 10,

    // Sector control
    "ElectronNum": 4,
    "ElectronSz2": 0,

    // Updater (recommended: KondoNNConserved)
    "Updater": "KondoNNConserved",

    // IO
    "WavefunctionBase": "tps"
}
```

## Compile-Time Symmetry

Controlled by `TENSOR_SYMMETRY_LEVEL` (set per target in `CMakeLists.txt`):

| Level | QN type   | Description                          |
|-------|-----------|--------------------------------------|
| 0     | fZ2QN     | Fermion parity only (fastest)        |
| 3     | fZ2U1QN   | Fermion parity + total $S^z$ (default) |

Default targets use level 3. Targets with `_z2` suffix use level 0.

## Source Files

| File | Purpose |
|------|---------|
| `qldouble.h` | Type definitions, Hilbert space basis, symmetry dispatch |
| `common_params.h` | Physics + SU algorithm parameter structs |
| `enhanced_params_parser.h` | VMC optimization parameters (optimizer, LR, clipping) |
| `mc_measure_params.h` | MC measurement parameters, configuration builder |
| `square_kondo_model.h` | Model solver: energy evaluation, observables (VMC/measure) |
| `square_kondo_nn_updater.h` | Custom NN local updater for MC sampling |
| `peps_kondo_square_simple_update.cpp` | Simple Update entry point |
| `vmc_optimize.cpp` | VMC optimization entry point |
| `mc_measure.cpp` | MC measurement entry point |
| `peps_kondo_2x2_exact_sum_optimize.cpp` | Exact summation optimizer (2×2 debugging) |

## Measured Observables

The MC measurement reports:

- **energy**: total energy per sample ($H_{\text{hop}} + H_{\text{onsite}} + H_{\text{Kondo}}$)
- **charge**: itinerant electron density $\langle n_i \rangle$ per site
- **spin_z**: total $S^z$ (electron + local) per site
- **spin_z_e**: itinerant electron $s^z_i$ per site
- **spin_z_loc**: localized spin $S^z_i$ per site
- **kondo_szSz**: on-site $\langle s^z_i S^z_i \rangle$ per site

## External Dependencies

- **qlten** (TensorToolkit): tensor primitives
- **qlmps** (UltraDMRG): `CaseParamsParser` for JSON config, `IsPathExist`
- **qlpeps** (PEPS): simple update executor, VMC optimizer, MC measurer, TPS types
