// SPDX-License-Identifier: MIT
/*
 * VMC optimization driver for the single-layer Kondo lattice PEPS (square lattice, OBC).
 *
 * Usage:
 *   mpirun -np <N> ./peps_kondo_square_vmc_optimize <physics_params.json> <vmc_algorithm_params.json>
 *
 * Input:
 *   - SplitIndexTPS is loaded from <WavefunctionBase>final/ (default: tpsfinal/)
 *
 * Output:
 *   - Optimizer dumps are controlled by VMCPEPSOptimizerParams (TPSDumpPath, ConfigurationDumpDir, etc.)
 */

#include "./qldouble.h"
#include "./square_kondo_model.h"
#include "./square_kondo_nn_updater.h"
#include "./enhanced_params_parser.h"

#include "qlpeps/qlpeps.h"
#include "qlpeps/api/vmc_api.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if __has_include(<mpi.h>)
#include <mpi.h>
#elif __has_include("mpi.h")
#include "mpi.h"
#else
// Editor fallback: real build must provide MPI.
using MPI_Comm = int;
static constexpr MPI_Comm MPI_COMM_WORLD = 0;
inline int MPI_Init(void *, void *) { return 0; }
inline int MPI_Finalize(void) { return 0; }
inline int MPI_Comm_rank(MPI_Comm, int *rank) { if (rank) *rank = 0; return 0; }
inline int MPI_Comm_size(MPI_Comm, int *sz) { if (sz) *sz = 1; return 0; }
#endif

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <physics_params.json> <vmc_algorithm_params.json>\n";
    return 1;
  }

  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0, mpi_size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);

  peps_kondo_params::EnhancedVMCOptimizeParams params(argv[1], argv[2]);

#ifdef _OPENMP
  omp_set_num_threads(static_cast<int>(params.bmps_params.ThreadNum));
#endif

  // Require SplitIndexTPS in <wavefunction_base>final/
  const std::string base = params.wavefunction_base; // default "tps"
  const std::string tps_final = base + "final";

  qlpeps::SplitIndexTPS<TenElemT, QNT> sitps;
  if (qlmps::IsPathExist(tps_final)) {
    if (rank == 0) std::cout << "Loading SplitIndexTPS from: " << tps_final << std::endl;
    sitps = qlpeps::SplitIndexTPS<TenElemT, QNT>(params.physical_params.Ly, params.physical_params.Lx);
    sitps.Load(tps_final);
  } else {
    if (rank == 0) {
      std::cerr << "ERROR: Missing wavefunction directory '" << tps_final
                << "'. Run Simple Update first to produce SplitIndexTPS under tpsfinal/.\n";
    }
    MPI_Finalize();
    return 2;
  }

  // Build optimizer params (rank-aware config load)
  auto vmc_params = params.CreateVMCOptimizerParams(rank);

  // Model energy solver (NN only for now; t2 can be added later)
  peps_kondo::SquareKondoModel solver(params.physical_params.t,
                                      params.physical_params.U,
                                      params.physical_params.JK,
                                      params.physical_params.mu);

  using Updater = peps_kondo::MCUpdateSquareKondoNNConservedOBC<>;
  Updater updater(params.bmps_params.Seed, params.bmps_params.ThreadNum,
                  static_cast<int>(params.electron_num),
                  params.sz2_electron);
  (void) qlpeps::VmcOptimize(vmc_params, sitps, comm, solver, updater);

  MPI_Finalize();
  return 0;
}


