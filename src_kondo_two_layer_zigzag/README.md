# Two-Layer Zigzag Kondo Ladder

**Purpose:** Study the effect of a finite interlayer coupling $J_\perp$ on the $(\\pi/2, \\pi/2)$ diagonal spin-stripe order at ambient pressure (Referee A, point 1).

## Hamiltonian

$$
H = H_{\text{hop}} + H_U + H_K + H_\perp
$$

### Intra-layer terms (identical for each layer $\ell = 0, 1$)

**Kinetic energy** (zigzag connectivity):

$$
H_{\text{hop}} = -t \sum_{\ell,\sigma} \sum_{x,y} c^\dagger_{(x,y),\ell,\sigma}\, c_{(x+1,y),\ell,\sigma}
\;-\; t_2 \sum_{\ell,\sigma} \sum_{x,y} c^\dagger_{(x,y),\ell,\sigma}\, c_{(x+1,y\pm1),\ell,\sigma}
\;+\; \text{h.c.}
$$

- $t$: intra-chain nearest-neighbor hopping along $x$
- $t_2$: inter-chain diagonal hopping (zigzag pattern, see below)

**Hubbard repulsion** on itinerant sites:

$$
H_U = U \sum_{\ell,x,y} n_{(x,y),\ell,\uparrow}\, n_{(x,y),\ell,\downarrow}
$$

**Kondo coupling** between itinerant electron spin $\mathbf{s}$ and localized spin $\mathbf{S}$:

$$
H_K = J_K \sum_{\ell,x,y} \mathbf{s}_{(x,y),\ell} \cdot \mathbf{S}_{(x,y),\ell}
$$

> Sign convention: $J_K = -J_H$. Ferromagnetic Hund's coupling corresponds to $J_K < 0$.

### Interlayer coupling

**AFM Heisenberg exchange** between localized spins on the two layers:

$$
H_\perp = J_\perp \sum_{x,y} \mathbf{S}_{(x,y),0} \cdot \mathbf{S}_{(x,y),1}
$$

$J_\perp > 0$ is antiferromagnetic. At ambient pressure $J_\perp$ is small; high pressure increases $t_\perp$, giving $J_\perp \sim t_\perp^2/U$.

## Zigzag Geometry

Each layer has $L_y$ chains running along $x$. The inter-chain hopping $t_2$ follows a zigzag (tilted) pattern:

- Even $x$: site $(x, y)$ hops to $(x+1, y+1)$
- Odd $x$: site $(x, y)$ hops to $(x+1, y-1)$

For $L_y = 2$:

```
  y=1:  o       o-------o       o-------o
        |      /         \     /         \
        |    t2            t2 /           ...
        |   /                /
  y=0:  o--t--o       o--t--o       o--t--o
            x=0  x=1  x=2  x=3  x=4  x=5
```

Boundary conditions along $y$ are controlled by the `Geometry` parameter:
- `"OBC"`: open boundaries (no wrapping)
- `"PBC"`: periodic wrapping of $t_2$ links

## MPS Site Ordering

For geometric site $(x, y)$, four consecutive MPS sites are assigned:

| MPS index                | Degree of freedom           |
|--------------------------|-----------------------------|
| $4(y + L_y x) + 0$      | Layer-0 itinerant electron  |
| $4(y + L_y x) + 1$      | Layer-0 localized spin      |
| $4(y + L_y x) + 2$      | Layer-1 itinerant electron  |
| $4(y + L_y x) + 3$      | Layer-1 localized spin      |

Total MPS length: $N = 4 \times L_y \times L_x$.

Even-indexed MPS sites are fermionic (electrons); odd-indexed are spin-1/2 (localized). Jordan-Wigner strings are inserted between all fermionic sites in hopping terms.

## Hilbert Space

Reuses the 8-dimensional Kondo site from `src_kondo_1d_chain/kondo_hilbert_space.h`:
- Itinerant electron: $\{|0\rangle, |\!\uparrow\rangle, |\!\downarrow\rangle, |\!\uparrow\downarrow\rangle\}$ (4 states, `pb_outE`)
- Localized spin: $\{|\!\uparrow\rangle, |\!\downarrow\rangle\}$ (2 states, `pb_outL`)

## Electron Filling

Quarter filling of the itinerant band: $L_x \times L_y$ electrons distributed across $2 L_x L_y$ itinerant sites (0.5 electrons/site/layer), matching the nominal $d_{x^2-y^2}$ filling in La$_3$Ni$_2$O$_7$.

## Parameters (`params.json`)

| Key          | Type       | Description                                    |
|--------------|------------|------------------------------------------------|
| `Geometry`   | string     | `"OBC"` or `"PBC"` (y-direction)               |
| `Lx`         | int        | Length along x (chain direction)                |
| `Ly`         | int        | Number of zigzag chains per layer               |
| `t`          | double     | Intra-chain NN hopping                          |
| `t2`         | double     | Inter-chain zigzag hopping                      |
| `Jk`         | double     | Kondo coupling ($< 0$ for FM Hund's)            |
| `Jperp`      | double     | Interlayer localized-spin exchange ($> 0$ AFM)   |
| `U`          | double     | Hubbard U on itinerant sites                    |
| `Sweeps`     | int        | Sweeps per bond-dimension stage                 |
| `Dmin`       | int        | Minimum bond dimension                          |
| `Dmax`       | int[]      | Bond dimensions (run sequentially)              |
| `CutOff`     | double     | SVD truncation cutoff                           |
| `LanczErr`   | double     | Lanczos convergence threshold                   |
| `MaxLanczIter` | int      | Max Lanczos iterations                          |
| `noise`      | double[]   | Noise schedule (per sweep)                      |
| `Threads`    | int        | OpenMP threads for tensor operations            |

Example (ambient pressure, small $J_\perp$):

```json
{
  "CaseParams": {
    "Geometry": "OBC",
    "Lx": 20, "Ly": 2,
    "t": 1.0, "t2": 0.3,
    "Jk": -4.0, "Jperp": 0.1, "U": 14.0,
    "noise": [1e-5, 1e-5, 0, 0],
    "Sweeps": 10, "Dmin": 100,
    "Dmax": [500, 1000, 2000, 4000],
    "CutOff": 1e-8, "LanczErr": 1e-9,
    "MaxLanczIter": 30, "Threads": 28
  }
}
```

## Measurements

The executable performs VMPS sweep stages over `Dmax`, and **after each bond dimension**
it runs measurements (MPI round-robin scheduled).

| Observable           | Sites measured                  | Output prefix  |
|----------------------|---------------------------------|----------------|
| $\langle S^z_i S^z_j \rangle$ (itinerant) | Layer-0 electron ref vs all itinerant sites to the right | `szsz`  |
| $\langle S^+_i S^-_j \rangle$ (itinerant) | same                        | `spsm`  |
| $\langle S^-_i S^+_j \rangle$ (itinerant) | same                        | `smsp`  |
| $\langle n_i n_j \rangle$ (itinerant)      | same                        | `nfnf`  |
| $\langle S^z_i S^z_j \rangle$ (localized, layer 0) | Layer-0 localized ref vs **layer-0 localized** targets | `l0szsz` |
| $\langle S^+_i S^-_j \rangle$ (localized, layer 0) | same                    | `l0spsm` |
| $\langle S^-_i S^+_j \rangle$ (localized, layer 0) | same                    | `l0smsp` |
| $\langle S^z_i S^z_j \rangle$ (localized, layer 1) | Layer-1 localized ref vs **layer-1 localized** targets | `l1szsz` |
| $\langle S^+_i S^-_j \rangle$ (localized, layer 1) | same                    | `l1spsm` |
| $\langle S^-_i S^+_j \rangle$ (localized, layer 1) | same                    | `l1smsp` |
| $\langle S^z_i \rangle$, $\langle n_i \rangle$ | All itinerant sites    | `sz_elec`, `n_elec` |
| $\langle S^z_i \rangle$ (localized)        | All localized sites per layer  | `sz_loc0`, `sz_loc1` |

Reference site is placed at $x = L_x/4$, $y = 0$ to maximize the correlation distance.
All outputs include the active bond dimension token `D<...>` in filenames.

## Building

Target: `kondo_two_layer_zigzag_vmps` (defined in the top-level `CMakeLists.txt`).

```bash
cd build && make kondo_two_layer_zigzag_vmps
```

## Running

```bash
mpirun -np <N> ./kondo_two_layer_zigzag_vmps params.json
```

## Post-Processing and Plot Pipeline

Standard workflow (cluster run -> rsync -> local plot) is documented in:

`docs/workflows/kondo_two_layer_zigzag_data_pipeline.md`

Local command (from repo root):

```bash
python3 plot/kondo_two_layer_zigzag/postprocess.py \
  --data-dir ./data \
  --out-dir ./plot/kondo_two_layer_zigzag/figures \
  --jperp 0.5 --jk -4 --t2 -0.6 --u 18 --lx 20 --ly 4 \
  --fit-order auto
```

The script enforces complete multi-`D` datasets and layer-purity checks before plotting.
