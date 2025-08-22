# LNO Superconductivity Project - Model Documentation

## Overview
This project contains DMRG (Density Matrix Renormalization Group) calculations for various quantum many-body models related to LNO (LaNiO3) superconductivity. All measurement data is stored in JSON format with correlation measurements between sites.

## Models by Source Directory

### 1. src_kondo_1d_chain/ - 1D Kondo Model
**Model Type**: One-dimensional Kondo lattice model
**Key Parameters**:
- `L`: System length (1D chain)
- `t`: Hopping parameter
- `JK`: Kondo interaction(negative of Hund's coupling)
- `U`: On-site Coulomb interaction
- `Dmax`: Maximum bond dimension

**Data File Naming Pattern**: `{correlation_type}t{t}Jk{JK}U{U}.json`
- Example: `szszt20.3Jk-1U0.json`
- Correlation types: `szsz` (spin Sz-Sz correlation)

### 2. src_kondo_two_leg_ladder/ - Single-layer Two-Leg Kondo Ladder
**Model Type**: Two-leg Kondo ladder model (quasi-1D)
**Key Parameters**:
- `Lx`: System length along x-direction
- `t`: Intra-chain hopping
- `t2`: Inter-chain hopping
- `JK`: Kondo interaction(negative of Hund's coupling)
- `U`: On-site Coulomb interaction
- `Geometry`: PBC (periodic) or OBC (open boundary conditions)

**Data File Naming Patterns**:
1. **Conventional square lattice**: `{correlation_type}conventional_squareJk{JK}U{U}Lx{Lx}D{D}.json`
   - Example: `szszconventional_squareJk-4U18Lx50D6100.json`
   - Standard two-leg ladder geometry
2. **45-degree rotated lattice extraction**: `{correlation_type}t{t}Jk{JK}U{U}Lx{Lx}D{D}.json`
   - Example: `szszt20.3Jk-4U10Lx100D6000.json`
   - Quasi-1D ladder extracted along 45-degree direction of the square lattice
- Correlation types: `szsz`, `nf_local`, `scs_diag_*`, `sct_diag_*`, `nfnf`, `smsp`, `spsm`

### 3. src_kondo_two_layer_2d/ - Two-Layer Kondo Model (2d lattice)
**Model Type**: Two-layer Kondo model on 2d lattice (fixed 2d)
**Key Parameters**:
- `Lx`: System length
- `t`: Intra-layer hopping
- `JK`: Kondo interaction(negative of Hund's coupling)
- `Jperp`: Inter-layer coupling
- `U`: On-site Coulomb interaction

**Data File Naming Pattern**: 
- Example: `szszconventional_squareJk-100Jperp10U18Lx20D5000.json`

**Hamiltonian**:
$$
\begin{aligned}
    H=&-\sum_{\langle ij\rangle,\ell,\sigma} t_{ij} c_{i,\ell,\sigma}^\dagger c_{j,\ell,\sigma}+h.c.+U\sum_{i\ell}n_{i\ell \uparrow}n_{i\ell\downarrow}\\
    &-J_H \sum_{i\ell}\bm{s}_{i,\ell}\cdot\bm{S}_{i,\ell}+J_\bot\sum_i \bm{S}_{i,1}\cdot\bm{S}_{i,2}.
\end{aligned}
$$

### 4. src_tj_double_layer_single_orbital_2d/ - Single Orbital t-J Model (Double Layer)
**Model Type**: Two-layer t-J model with single orbital (tJ-like, 3-dim local Hilbert space)
**Key Parameters**:
- `Lx`, `Ly`: System dimensions
- `t`: Intra-layer nearest-neighbor hopping
- `t_perp`: Inter-layer hopping
- `J`: Intra-layer super-exchange
- `J_perp`: Inter-layer super-exchange
- `delta`: Hopping anisotropy
- `Numhole`: Hole doping (0 = quarter filling)
- `pinning_field`: Whether pinning field is applied

**Data File Naming Pattern**: `{correlation_type}SingleOrbitaldelta={delta}_{geometry}_{Lx}x{Ly}{Pinning}.{format}`
- Example: `SpinSingleOrbitaldelta=0.3_2x20NoPing.svg`
- Correlation types: `Spin`, `SpinCorr`

### 5. src_tj_single_layer_single_orbital_2d_anisotropic/ - Single Orbital t-J Model (Single Layer)
**Model Type**: Single-layer anisotropic t-J model (tJ-like, 3-dim local Hilbert space)
**Key Parameters**:
- `Lx`, `Ly`: System dimensions
- `t`: Nearest-neighbor hopping
- `delta`: Hopping anisotropy
- `J`: Super-exchange coupling
- `Numhole`: Hole doping

**Data File Naming Pattern**: Similar to double layer but for single layer

### 6. src_two_layer_two_orbital_all_dof/ - Two-Band Two-Layer (Two-Orbital) Model
**Model Type**: Two-band two-layer (two-orbital) model with all degrees of freedom (Hubbard-like local Hilbert space; band ≃ orbital)
**Hamiltonian**:
```
H = -t_∥ ∑_{l=1,2} ∑_{⟨i,j⟩,σ} d_{l,i,σ}^† d_{l,j,σ} + h.c. 
    + ε_d ∑_{l=1,2} ∑_i d_{l,i,σ}^† d_{l,i,σ}
    - t_⊥ ∑_i f_{1,i,σ}^† f_{2,i,σ} + h.c.
    + ε_f ∑_{l=1,2} ∑_i f_{l,i,σ}^† f_{l,i,σ}
    + ∑_{l,i} U(n_{f,l,i,↑} n_{f,l,i,↓} + n_{d,l,i,↑} n_{d,l,i,↓})
    - 2J_H S_{d,l,i} · S_{f,l,i}
```

**Key Parameters**:
- `t_∥`: Intra-layer hopping
- `t_⊥`: Inter-layer hopping
- `ε_d`, `ε_f`: On-site energies
- `U`: Coulomb interaction
- `J_H`: Hund's coupling

## Data File Structure

All correlation data files contain JSON arrays with the format:
```json
[
  [[site1, site2], correlation_value],
  [[site1, site3], correlation_value],
  ...
]
```

Where:
- `site1, site2`: Site indices (1D) or coordinates (2D)
- `correlation_value`: Real number representing the correlation strength

## Correlation Types

1. **szsz**: Spin-spin correlation ⟨S^z_i S^z_j⟩
2. **nf_local**: Local fermion density ⟨n_i⟩
3. **scs_diag_***: Superconducting correlation (singlet, diagonal)
4. **sct_diag_***: Superconducting correlation (triplet, diagonal)
5. **nfnf**: Density-density correlation ⟨n_i n_j⟩
6. **smsp**: Spin correlation ⟨S^-_i S^+_j⟩
7. **spsm**: Spin correlation ⟨S^+_i S^-_j⟩

## File Naming Convention Summary

### Standard Format (Lx=100 style):
`{correlation_type}conventional_squareJk{JK}Jperp{Jperp}U{U}Lx{Lx}D{D}.json`

### Legacy Format (files_without_lx.txt):
`{correlation_type}t{t}Jk{JK}U{U}.json`

### Parameters in Filenames:
- `t`: Hopping parameter
- `Jk` or `JK`: Kondo coupling (Hund's coupling)
- `Jperp`: Inter-layer/perpendicular coupling
- `U`: Coulomb interaction
- `Lx`: System length
- `D`: Bond dimension (DMRG parameter)
- `delta`: Hopping anisotropy (for t-J models) 