# Kondo Lattice Model PEPS VMC Validation Report

**Date**: 2026-01-02  

## Overview

This document records the validation of the PEPS VMC implementation for the single-layer Kondo Lattice Model against Exact Diagonalization (ED) results.

## Model Definition

The Hamiltonian:

```
H = H_hopping + H_Hubbard + H_Kondo

H_hopping = -t Σ_{<i,j>,σ} (c†_{iσ} c_{jσ} + h.c.)
H_Hubbard = U Σ_i n_{i↑} n_{i↓} - μ Σ_i n_i
H_Kondo   = JK Σ_i (s_i · S_i)
          = JK Σ_i [s_z^i S_z^i + (1/2)(s_+^i S_-^i + s_-^i S_+^i)]
```

Where:
- `c†_{iσ}`, `c_{iσ}`: Itinerant electron creation/annihilation operators
- `s_i`: Itinerant electron spin-1/2 operator at site i
- `S_i`: Localized spin-1/2 operator at site i
- `JK < 0`: Ferromagnetic (FM) Kondo coupling
- `JK > 0`: Antiferromagnetic (AFM) Kondo coupling

### Local Hilbert Space

8-dimensional local space = (4 electron states) × (2 local spin states):

| idx | Electron | Local Spin | Notation |
|-----|----------|------------|----------|
| 0   | ↑↓ (doublon) | ↑ | \|D,Up⟩ |
| 1   | ↑↓ (doublon) | ↓ | \|D,Dn⟩ |
| 2   | ↑ | ↑ | \|U,Up⟩ |
| 3   | ↑ | ↓ | \|U,Dn⟩ |
| 4   | ↓ | ↑ | \|d,Up⟩ |
| 5   | ↓ | ↓ | \|d,Dn⟩ |
| 6   | 0 (empty) | ↑ | \|0,Up⟩ |
| 7   | 0 (empty) | ↓ | \|0,Dn⟩ |

Encoding: `idx = 2 * electron_state + local_spin_state`

## Test Methodology

### Exact Diagonalization (ED)

- Script: `src_peps_kondo_single_layer/tools/ed_kondo_small.py`
- Method: Full Hilbert space construction with Lanczos diagonalization
- Symmetry sector: Fixed electron number (Ne) and total Sz (Sz_total = 0)

### PEPS VMC (Exact Summation)

- Initialization: Simple Update with imaginary time evolution
- Optimization: AdaGrad with Exact Summation (no Monte Carlo noise)
- Executable: `peps_kondo_2x2_exact_sum_optimize_z2`
- Symmetry: Z2 quantum number (fermion parity)
- Bond dimension: D = 6

### Validation Protocol

1. Run ED to obtain exact ground state energy
2. Initialize PEPS with Simple Update (~1000 steps)
3. Optimize PEPS with Exact Summation (~100-300 iterations)
4. Compare final PEPS energy with ED energy

## Test Results

### Test 1: Ferromagnetic Kondo (JK < 0)

**Parameters:**
- Lattice: 2×2 (OBC)
- t = 1.0, U = 4.0, **JK = -1.0**, μ = 0.0
- Ne = 4 (half-filling), Sz_total = 0

**Results:**

| Method | Energy | 
|--------|--------|
| ED (exact) | **-2.811582631** |
| PEPS VMC | **-2.811476** |
| Error | **0.004%** |

### Test 2: Antiferromagnetic Kondo (JK > 0)

**Parameters:**
- Lattice: 2×2 (OBC)
- t = 1.5, U = 2.0, **JK = +0.8**, μ = 0.0
- Ne = 4 (half-filling), Sz_total = 0

**Results:**

| Method | Energy |
|--------|--------|
| ED (exact) | **-5.704985872** |
| PEPS VMC | **-5.702702** |
| Error | **0.04%** |

## Bug Fix Summary

### Issue Identified

The original `CalEnergyAndHolesImpl` (inherited from `SquareNNModelEnergySolver`) only computed:
- NN hopping energy (off-diagonal bond term)
- Diagonal on-site energy (U, μ, JK·sz·Sz)

**Missing term**: The off-diagonal Kondo flip-flop contribution `(JK/2)(s_+S_- + s_-S_+)`

This term connects states:
- |↑,↓⟩ (idx=3) ↔ |↓,↑⟩ (idx=4)

### Solution

1. **Changed inheritance**: `SquareKondoModel` now directly inherits from 
   `ModelEnergySolver<SquareKondoModel>` instead of `SquareNNModelEnergySolver<SquareKondoModel>`.
   This ensures CRTP correctly dispatches to our custom `CalEnergyAndHolesImpl`.

2. **Added Kondo flip-flop calculation**: New method `EvaluateKondoFlipFlopEnergy()` computes:
   ```
   E_flip = (JK/2) Σ_i ⟨ψ'|ψ⟩/⟨ψ|ψ⟩
   ```
   where |ψ'⟩ is the state with idx=3↔idx=4 flipped at site i.

3. **Correct wavefunction ratio calculation**: For single-site operations, use 
   `ReplaceOneSiteTrace(tn, site, tensor)` for both numerator and denominator,
   avoiding boundary issues with `Trace(site, orient)`.

### Key Code Changes

File: `src_peps_kondo_single_layer/square_kondo_model.h`

```cpp
// New inheritance (enables correct CRTP dispatch)
class SquareKondoModel : public qlpeps::ModelEnergySolver<SquareKondoModel>,
                         public qlpeps::SquareNNModelMeasurementSolver<SquareKondoModel> {

  // Full energy calculation
  template<typename TenElemT, typename QNT, bool calchols = true>
  TenElemT CalEnergyAndHolesImpl(...) {
    // 1. Hopping energy (horizontal + vertical bonds)
    TenElemT bond_energy = CalHorizontalBondEnergyAndHoles(...) + CalVerticalBondEnergy(...);
    
    // 2. Diagonal on-site energy
    TenElemT onsite_energy = EvaluateTotalOnsiteEnergy(config);
    
    // 3. Kondo flip-flop energy (NEW)
    TenElemT flip_energy = EvaluateKondoFlipFlopEnergy(split_index_tps, tps_sample);
    
    return bond_energy + onsite_energy + flip_energy;
  }
};
```

## Conclusion

The PEPS VMC implementation for the Kondo Lattice Model has been validated against ED 
with excellent agreement (< 0.05% error). Both FM and AFM Kondo coupling cases are 
correctly handled.

## Files Modified

- `src_peps_kondo_single_layer/square_kondo_model.h`: Core model implementation
- `src_peps_kondo_single_layer/peps_kondo_2x2_exact_sum_optimize.cpp`: Resume capability added

## How to Reproduce

```bash
# 1. Run ED
cd src_peps_kondo_single_layer/tools
python3 ed_kondo_small.py --one --Lx 2 --Ly 2 --Ne 4 --Sz2_total 0 \
    --t 1.0 --U 4.0 --JK -1.0 --mu 0.0

# 2. Run PEPS Simple Update
cd cmake-build-peps-llvm
./peps_kondo_square_simple_update_z2 \
    ../src_peps_kondo_single_layer/params/tests_2x2/physics_full_2x2.json \
    ../src_peps_kondo_single_layer/params/tests_2x2/su_algo_z2_D6.json

# 3. Run PEPS Exact Summation Optimization
mpirun -np 1 ./peps_kondo_2x2_exact_sum_optimize_z2 \
    ../src_peps_kondo_single_layer/params/tests_2x2/physics_full_2x2.json \
    ../src_peps_kondo_single_layer/params/tests_2x2/exact_sum_z2_D6.json
```

## References

- Kondo Lattice Model: Doniach, S. (1977). Physica B+C, 91, 231-234.
- PEPS algorithm: Verstraete, F., & Cirac, J. I. (2004). arXiv:cond-mat/0407066
- VMC for PEPS: Sandvik, A. W., & Vidal, G. (2007). PRL 99, 220602

