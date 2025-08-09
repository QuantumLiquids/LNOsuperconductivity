# LNO Superconductivity Project

## Overview
This project contains DMRG (Density Matrix Renormalization Group) calculations for various quantum many-body models related to LNO (LaNiO3) superconductivity. The project implements multiple models including Kondo lattice models, t-J models, and multi-orbital models for studying superconductivity in layered nickelates.

For detailed model descriptions and data file structures, see [Model_Notes.md](Model_Notes.md).

## Dependencies

### Required Dependencies
- **C++ Compiler**: C++17 or above
- **Build System**: CMake (version 3.14 or higher)
- **Math Libraries**: Intel MKL or OpenBLAS
- **Parallelization**: MPI
- **Tensor Library**: [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit)
- **DMRG Library**: [QuantumLiquids/UltraDMRG](https://github.com/QuantumLiquids/UltraDMRG)

### Optional Dependencies
- **GPU Acceleration**: CUDA compiler, cuBLAS, cuSolver, cuTensor2
- **High-Performance Tensor Transpose**: HPTT (for CPU-only builds)

## Build Instructions

### Prerequisites
Install the required dependencies:
   - **MPI**: OpenMPI, MPICH, or IntelMPI
   - **BLAS/LAPACK**: Intel MKL (recommended for x86) or OpenBLAS (for ARM)
   - **TensorToolkit and UltraDMRG**: Follow installation instructions from their respective repositories

For Intel x86 user, we recommand use Intel oneAPI toolkit to include both MPI and BLAS/LAPACK dependencise.

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/LNOsuperconductivity.git
   cd LNOsuperconductivity
   ```

2. Create build directory and configure:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_CXX_COMPILER=your_cxx_compiler \
            -DCMAKE_PREFIX_PATH=path/to/tensortoolkit/and/ultradmrg \
            -DQLMPS_USE_GPU=ON/OFF \
            -DCUTENSOR_ROOT=path/to/cutensor/if/using/cuda
   ```

3. Build the project:
   ```bash
   make -j4
   ```

### Common CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `-DQLMPS_USE_GPU=ON/OFF` | Enable GPU acceleration | OFF |
| `-DRealCode=ON/OFF` | Use real or complex tensors | OFF (complex) |
| `-DQLTEN_TIMING_MODE=ON/OFF` | Enable timing mode | OFF |
| `-DQLMPS_TIMING_MODE=ON/OFF` | Enable MPS timing mode | OFF |
| `-DCOMPILE_FIX_CODE=ON/OFF` | Compile fixing code | OFF |

### Platform-Specific Notes

#### x86 Platforms
- Automatically uses Intel MKL as BLAS implementation
- Set `BLA_VENDOR=Intel10_64lp` for Intel MKL
- Set `BLAS_INCLUDE_DIR=$ENV{MKLROOT}/include`

#### ARM Platforms (Apple Silicon)
- Defaults to OpenBLAS
- Set `BLA_VENDOR=OpenBLAS`
- Set `OpenBLAS_ROOT=/opt/homebrew/opt/openblas/` (for Homebrew installations)

## Executables and Usage

After building, you'll have the following executables:

### Single Orbital Models
| Executable | Description | Usage |
|------------|-------------|-------|
| `dmrg_single_orbital` | DMRG for double-layer t-J model | `mpirun -np <num_proc> ./dmrg_single_orbital params.json` |
| `dmrg_ani_tJ` | DMRG for single-layer anisotropic t-J model | `mpirun -np <num_proc> ./dmrg_ani_tJ params.json` |
| `measure1_ani_tJ` | Measure correlations for single-layer t-J | `./measure1_ani_tJ params.json` |

### Multi-Orbital Models
| Executable | Description | Usage |
|------------|-------------|-------|
| `dmrg4band` | DMRG for two-band two-orbital model | `mpirun -np <num_proc> ./dmrg4band params.json` |
| `measure4band1` | Measure correlations for 4-band model | `./measure4band1 params.json` |
| `measure4bandSC` | Measure SC correlations for 4-band model | `mpirun -np <num_proc> ./measure4bandSC params.json` |

### Kondo Models
| Executable | Description | Usage |
|------------|-------------|-------|
| `kondo_chain_vmps` | VMPS for 1D Kondo chain | `mpirun -np <num_proc> ./kondo_chain_vmps params.json` |
| `kondo_ladder_vmps` | VMPS for two-leg Kondo ladder | `mpirun -np <num_proc> ./kondo_ladder_vmps params.json` |
| `kondo_ladder_measure` | Measure correlations for Kondo ladder | `./kondo_ladder_measure params.json` |
| `kondo_ladder_conventional_square_vmps` | VMPS for conventional square Kondo lattice | `mpirun -np <num_proc> ./kondo_ladder_conventional_square_vmps params.json` |
| `kondo_two_layer_vmps` | VMPS for two-layer Kondo model | `mpirun -np <num_proc> ./kondo_two_layer_vmps params.json` |

## Parameter Files

Each executable requires a JSON parameter file. Example parameter files can be found in the respective `src_*/` directories:

- `src_single_orbital/params.json` - Single orbital model parameters
- `src_2layer_2orbital_all_dof/params.json` - Multi-orbital model parameters
- `src_kondo_1D/params.json` - Kondo chain parameters
- `src_kondo_two_leg/params.json` - Kondo ladder parameters

## Data Output

All correlation measurements are stored in JSON format with the structure:
```json
[
  [[site1, site2], correlation_value],
  [[site1, site3], correlation_value],
  ...
]
```

Data files follow naming conventions documented in [Model_Notes.md](Model_Notes.md).

## Troubleshooting

### Common Issues

1. **MPI not found**: Ensure MPI is properly installed and `mpicc`/`mpicxx` are in your PATH
2. **BLAS/LAPACK not found**: Check that MKL or OpenBLAS is properly installed and configured
3. **TensorToolkit/UltraDMRG not found**: Ensure these libraries are installed and `CMAKE_PREFIX_PATH` is set correctly
4. **GPU compilation errors**: Verify CUDA installation and set `CUTENSOR_ROOT` if using cuTensor

### Debug Mode
To build in debug mode:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j4
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the LGPL-3.0 license.

## Contact

For questions, issues, or collaboration opportunities, please contact:
- **Hao-Xin Wang**: [wanghaoxin1996@gmail.com](mailto:wanghaoxin1996@gmail.com)

## Acknowledgments

This project builds upon the [QuantumLiquids/TensorToolkit](https://github.com/QuantumLiquids/TensorToolkit) and [QuantumLiquids/UltraDMRG](https://github.com/QuantumLiquids/UltraDMRG) libraries.
