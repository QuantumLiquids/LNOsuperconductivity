// SPDX-License-Identifier: MIT
/*
 * Monte Carlo measurement for PEPS single-layer Kondo lattice on square lattice.
 *
 * Workflow:
 *  1) Run simple update to generate SplitIndexTPS under "tpsfinal/".
 *  2) Run this program to measure energy and observables by VMC sampling.
 *
 * Usage:
 *   mpirun -np <N> ./peps_kondo_square_mc_measure <physics_params.json> <mc_measure_algorithm_params.json>
 */

#include <iostream>
#include <string>

#if __has_include(<mpi.h>)
#include <mpi.h>
#elif __has_include("mpi.h")
#include "mpi.h"
#else
// Fallback for editors/clangd when MPI headers are not on the includePath.
// The real build must provide MPI; otherwise this executable is meaningless.
using MPI_Comm = int;
static constexpr MPI_Comm MPI_COMM_WORLD = 0;
inline int MPI_Init(void *, void *) { return 0; }
inline int MPI_Finalize(void) { return 0; }
inline int MPI_Comm_rank(MPI_Comm, int *rank) { if (rank) *rank = 0; return 0; }
#endif

#include "qlpeps/qlpeps.h"
#include "qlpeps/api/vmc_api.h"

#include "./qldouble.h"
#include "./mc_measure_params.h"
#include "./square_kondo_model.h"
#include "./square_kondo_nn_updater.h"

using namespace qlpeps;

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <physics_params.json> <mc_measure_algorithm_params.json>\n";
    return -1;
  }

  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  peps_kondo_params::EnhancedMCMeasureParams params(argv[1], argv[2]);

  // Load SplitIndexTPS from <WavefunctionBase>final/
  SplitIndexTPS<TenElemT, QNT> sitps(params.physical_params.Ly, params.physical_params.Lx);
  const std::string tps_dir = params.wavefunction_base + std::string("final");
  if (!qlmps::IsPathExist(tps_dir)) {
    if (rank == 0) {
      std::cerr << "ERROR: missing wavefunction directory '" << tps_dir
                << "'. Run Simple Update first (should dump to tpsfinal/).\n";
    }
    MPI_Finalize();
    return -2;
  }
  if (rank == 0) {
    std::cout << "Loading SplitIndexTPS from: " << tps_dir << "\n";
  }
  sitps.Load(tps_dir);

  // Build measurement params (also builds or loads initial config)
  auto measure_params = params.CreateMCMeasurementParams(rank);

  // Model solver
  peps_kondo::SquareKondoModel solver(params.physical_params.t,
                                      params.physical_params.U,
                                      params.physical_params.JK,
                                      params.physical_params.mu);

  // Updater:
  // We intentionally only use the Kondo-aware NN updater here, because it preserves the
  // correct conserved quantities (Ne and Sz_total) without over-constraining the Markov chain.
  if (params.updater != "KondoNNConserved") {
    if (rank == 0) {
      std::cerr << "[WARN] Updater=" << params.updater
                << " is deprecated in this model. Forcing Updater=KondoNNConserved.\n";
    }
  }
  using Updater = peps_kondo::MCUpdateSquareKondoNNConservedOBC<>;
  Updater updater(params.bmps_params.Seed, params.bmps_params.ThreadNum,
                  static_cast<int>(params.electron_num),
                  params.sz2_electron);
  (void) qlpeps::MonteCarloMeasure<TenElemT, QNT, Updater, peps_kondo::SquareKondoModel>(
      sitps, measure_params, comm, solver, updater);

  MPI_Finalize();
  return 0;
}


