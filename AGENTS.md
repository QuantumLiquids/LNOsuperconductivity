# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

This is a research codebase for studying superconductivity in bilayer nickelate La3Ni2O7 using large-scale tensor-network methods (DMRG and PEPS). The central physics: a two-orbital ferromagnetic Kondo-Hubbard model explains the diagonal (pi/2, pi/2) spin stripe order at ambient pressure and interlayer singlet pairing at high pressure. The paper is submitted to PRL (LM19284, first-round review).

Active manuscript files:
- `draft/main_final.tex` (main text)
- `draft/SupplementaryMaterial.tex`
- `draft/nickelate.bib`
- `draft/reply_file/reply.tex` (reviewer reply draft)

## Build Instructions

Build from `build/` and try `make <target>` first.

### macOS (Apple Silicon, Homebrew LLVM)

Only rerun CMake if make fails (for example, OpenMP issues):

```bash
cd build
SDK="$(xcrun --sdk macosx --show-sdk-path)"
cmake .. -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++ \
         -DCMAKE_PREFIX_PATH=path/to/TensorToolkit:path/to/UltraDMRG \
         -DQLMPS_USE_GPU=OFF \
         -DCMAKE_OSX_SYSROOT="$SDK" \
         -DCMAKE_CXX_FLAGS="-nostdinc++ -isystem $SDK/usr/include/c++/v1"
make -j4
```

If CMake reports `Could NOT find OpenMP_CXX`, rerun the exact same CMake command once (known Homebrew LLVM OpenMP detection flakiness).

### Cluster

See `skills/cluster-lno.md` for SSH, build, job submission, and file transfer details.

### Key CMake Options

- `-DQLMPS_USE_GPU=ON`: enable GPU (CUDA + cuBLAS + cuTensor; CUDA arch 70/80).
- `-DRealCode=ON`: use `double` tensors instead of `complex`.
- `-DCOMPILE_FIX_CODE=ON`: compile environment/MPS fixing utilities.
- `-DQLTEN_TIMING_MODE=ON`: enable tensor-level timing output.

## Executables and Running Jobs

All DMRG/VMPS executables take `params.json` and run under MPI:

```bash
mpirun -np <N> ./kondo_two_layer_vmps params.json
```

PEPS executables take two JSON files (physics + algorithm):

```bash
./peps_kondo_square_simple_update params/physics_params.json params/simple_update_algo.json
```

Single-process measurement executables (no `mpirun`): `kondo_ladder_measure`, `kondo_chain_measure`, `kondo_two_layer_measure_sp`, `measure4band1`, `measure1_ani_tJ`.

## Code Architecture

Each `src_*/` directory is a self-contained model variant. They share `src_tj_double_layer_single_orbital_2d/myutil.cpp` as a common utility.

- `src_kondo_1d_chain/`: 1D Kondo chain (8D per site).
- `src_kondo_zigzag_ladder/`: single-layer zigzag/two-leg Kondo ladder (8D per site).
- `src_kondo_two_layer_2d/`: main model, two-layer Kondo on 2D square lattice (8D x 2 layers).
- `src_two_layer_two_orbital_all_dof/`: full Hubbard-Kanamori two-orbital two-layer (16D per site).
- `src_tj_double_layer_single_orbital_2d/`: double-layer single-orbital t-J model (3D per site).
- `src_tj_single_layer_single_orbital_2d_anisotropic/`: single-layer anisotropic t-J (3D per site).
- `src_peps_kondo_single_layer/`: new (untested) single-layer Kondo PEPS (8D per site).

### External Libraries (header-only, outside this repo)

- `qlten` (TensorToolkit): tensor primitives, SVD, contraction. Headers at `~/TensorToolkit/include`.
- `qlmps` (UltraDMRG): DMRG/VMPS algorithms, MPS types. Headers at `~/UltraDMRG/include`.
- `qlpeps` (PEPS): simple update, VMC, measurement. Headers at `~/PEPS/include`.

## Key Physical Conventions

Kondo/Hund coupling sign:
- Code uses `JK` with `H_K = J_K s·S`.
- Paper writes `-J_H s·S` with `J_H > 0` for ferromagnetic Hund coupling.
- Mapping: `JK = -J_H`, so FM coupling corresponds to `JK < 0`.

2D->1D DMRG mapping:
- Snake pattern, x-first then y, then on-site DoF.
- Reference: `src_kondo_two_layer_2d/DMRG_Mapping.md`.

PEPS symmetry levels (`TENSOR_SYMMETRY_LEVEL`):
- 0: fZ2QN (fermion parity only)
- 3: fZ2U1QN (fermion parity + total Sz), default/recommended.

Electron filling:
- La3Ni2O7 has average filling 1.5 electrons/site: 1 from d_z2 and 0.5 from d_x2-y2.
- This motivates replacing d_z2 Hubbard U with effective spin `J_perp ~ t_perp^2 / U`.

## Data Files

Measurement output format:
- JSON arrays like `[[[site1, site2], value], ...]`.

Naming convention:
- Lowercase with underscores between words and hyphens for signed/decimal parameter values.
- Example: `singlet_sc_corr_jk-4_jperp0.1_u18_lx50.json`.

Parameter order in filenames:
- `jk` -> `jperp` -> `u` -> `lx` -> `ly` -> `jh` -> `t2` -> `delta`.

## Code Style and Validation

- C++17 (PEPS targets use C++20), Google C++ Style, 2-space indentation.
- `lower_snake_case` for functions/variables, `PascalCase` for types.
- No formal test suite: validate with small lattices (for example, 2x4) and manually inspect energies/observables.

## PEPS Architecture (`src_peps_kondo_single_layer/`)

Work in progress, verified on 2x2 only. Pipeline: Simple Update -> VMC Optimize -> MC Measure.

Key files:
- `qldouble.h`: Type definitions, Hilbert space basis (compile-time symmetry dispatch via `TENSOR_SYMMETRY_LEVEL`).
- `square_kondo_model.h`: Main model solver (CRTP plugin for `ModelEnergySolver` + `SquareNNModelMeasurementSolver`).
- `square_kondo_nn_updater.h`: Custom local updater for simple update NN bond gates.
- `common_params.h`, `enhanced_params_parser.h`, `mc_measure_params.h`: Parameter parsing.

Design: plugin-style models (template argument to `VMCPEPSOptimizer`/`MCPEPSMeasurer`), mixin composition for measurements, OBC/PBC auto-dispatch via `SplitIndexTPS`.

Data flow: Simple Update writes `peps/` + `tpsfinal/`. VMC reads/writes `tpsfinal/`. MC measurement reads `tpsfinal/`.

Checkerboard zigzag hopping: `t` (intra-chain), `t2` (inter-chain) on square lattice. Parity = (row+col)%2 determines which bonds use t vs t2. All three stages (SU, VMC, measure) use exact per-bond Hamiltonians. See `src_peps_kondo_single_layer/README.md` for details.
