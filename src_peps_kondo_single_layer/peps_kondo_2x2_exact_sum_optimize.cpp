// SPDX-License-Identifier: MIT
/*
 * Exact-summation optimizer for 2x2 OBC single-layer Kondo lattice PEPS.
 *
 * Goal:
 * - Remove Monte-Carlo noise completely.
 * - Provide a deterministic, tiny-cluster regression test for:
 *   - Hamiltonian sign conventions (especially fermion signs in hopping)
 *   - Energy estimator consistency (bond + onsite)
 *   - Optimizer plumbing (gradient flow)
 *
 * This intentionally mirrors:
 *   PEPS/tests/test_optimizer/test_optimizer_adagrad_exact_sum.cpp
 *
 * Usage (run inside a directory that contains `tpsfinal/` from simple_update):
 *   mpirun -np 1 ./peps_kondo_2x2_exact_sum_optimize <physics.json> <algo.json>
 *
 * Notes:
 * - This program is only meant for 2x2 OBC.
 * - Sector restriction is enforced: Ne even and Sz_total = 0.
 */

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/exact_summation_energy_evaluator.h"
#include "qlpeps/optimizer/optimizer.h"
#include "qlpeps/optimizer/optimizer_params.h"

#if __has_include(<mpi.h>)
#include <mpi.h>
#elif __has_include("mpi.h")
#include "mpi.h"
#else
using MPI_Comm = int;
static constexpr MPI_Comm MPI_COMM_WORLD = 0;
inline int MPI_Init(void *, void *) { return 0; }
inline int MPI_Finalize(void) { return 0; }
inline int MPI_Comm_rank(MPI_Comm, int *rank) { if (rank) *rank = 0; return 0; }
inline int MPI_Comm_size(MPI_Comm, int *sz) { if (sz) *sz = 1; return 0; }
#endif

#include "./common_params.h"
#include "./qldouble.h"
#include "./square_kondo_model.h"

namespace {

inline int ElectronNumFromCombined(size_t c) {
  const size_t e = c / 2;  // 0=D,1=U,2=d,3=0
  if (e == 0) return 2;
  if (e == 1 || e == 2) return 1;
  return 0;
}

inline int ElectronSz2FromCombined(size_t c) {
  const size_t e = c / 2;
  if (e == 1) return +1;
  if (e == 2) return -1;
  return 0;
}

inline int LocalSz2FromCombined(size_t c) {
  const size_t s = c % 2;  // 0=Up,1=Dn
  return (s == 0) ? +1 : -1;
}

inline int SzTot2FromCombined(size_t c) {
  return ElectronSz2FromCombined(c) + LocalSz2FromCombined(c);
}

struct ExactSumAlgoParams : public qlmps::CaseParamsParserBasic {
  explicit ExactSumAlgoParams(const char *algo_file) : qlmps::CaseParamsParserBasic(algo_file) {
    optimizer_type = ParseStrOr("OptimizerType", "AdaGrad");
    max_iterations = static_cast<size_t>(ParseIntOr("MaxIterations", 100));
    learning_rate = ParseDoubleOr("LearningRate", 0.1);
    energy_tolerance = ParseDoubleOr("EnergyTolerance", 0.0);
    gradient_tolerance = ParseDoubleOr("GradientTolerance", 0.0);

    // AdaGrad knobs
    epsilon = ParseDoubleOr("Epsilon", 1e-8);
    initial_accumulator = ParseDoubleOr("InitialAccumulator", 0.0);

    // SR/CG knobs (if ever used)
    cg_max_iter = static_cast<size_t>(ParseIntOr("CGMaxIter", 100));
    cg_tol = ParseDoubleOr("CGTol", 1e-8);
    cg_residue_restart = static_cast<size_t>(ParseIntOr("CGResidueRestart", 20));
    cg_diag_shift = ParseDoubleOr("CGDiagShift", 0.01);
    normalize_update = ParseBoolOr("NormalizeUpdate", false);

    // BMPS truncation for exact summation contraction
    Db_min = static_cast<size_t>(ParseIntOr("Db_min", 4));
    Db_max = static_cast<size_t>(ParseIntOr("Db_max", 16));
    trunc_err = ParseDoubleOr("TruncErr", 1e-12);
    mps_compress_scheme = static_cast<qlpeps::CompressMPSScheme>(ParseIntOr("MPSCompressScheme", 0));
  }

  std::string optimizer_type;
  size_t max_iterations = 0;
  double learning_rate = 0.0;
  double energy_tolerance = 0.0;
  double gradient_tolerance = 0.0;

  // AdaGrad
  double epsilon = 1e-8;
  double initial_accumulator = 0.0;

  // SR
  size_t cg_max_iter = 100;
  double cg_tol = 1e-8;
  size_t cg_residue_restart = 20;
  double cg_diag_shift = 0.01;
  bool normalize_update = false;

  // BMPS
  size_t Db_min = 4;
  size_t Db_max = 16;
  double trunc_err = 1e-12;
  qlpeps::CompressMPSScheme mps_compress_scheme{};
};

qlpeps::OptimizerParams BuildOptimizerParams(const ExactSumAlgoParams &p) {
  qlpeps::OptimizerParams::BaseParams base(
      p.max_iterations,
      p.energy_tolerance,
      p.gradient_tolerance,
      /*plateau_patience=*/p.max_iterations,
      p.learning_rate,
      nullptr);

  std::string t = p.optimizer_type;
  if (t == "sr" || t == "SR") t = "StochasticReconfiguration";

  if (t == "AdaGrad" || t == "adagrad") {
    return qlpeps::OptimizerParams(base, qlpeps::AdaGradParams(p.epsilon, p.initial_accumulator));
  }
  if (t == "SGD" || t == "sgd") {
    return qlpeps::OptimizerParams(base, qlpeps::SGDParams(/*momentum=*/0.0, /*nesterov=*/false, /*weight_decay=*/0.0));
  }

  // Default: SR
  qlpeps::ConjugateGradientParams cg(p.cg_max_iter, p.cg_tol, p.cg_residue_restart, p.cg_diag_shift);
  return qlpeps::OptimizerParams(base, qlpeps::StochasticReconfigurationParams(cg, p.normalize_update));
}

std::vector<qlpeps::Configuration> GenerateAll2x2ConfigsSector(size_t Ne_target) {
  const size_t Lx = 2, Ly = 2;
  std::vector<qlpeps::Configuration> out;
  out.reserve(4096);

  // Enumerate all 8^4 configurations.
  for (size_t c0 = 0; c0 < 8; ++c0) {
    for (size_t c1 = 0; c1 < 8; ++c1) {
      for (size_t c2 = 0; c2 < 8; ++c2) {
        for (size_t c3 = 0; c3 < 8; ++c3) {
          const int ne = ElectronNumFromCombined(c0) + ElectronNumFromCombined(c1) +
                         ElectronNumFromCombined(c2) + ElectronNumFromCombined(c3);
          if (static_cast<size_t>(ne) != Ne_target) continue;
          const int sz2 = SzTot2FromCombined(c0) + SzTot2FromCombined(c1) +
                          SzTot2FromCombined(c2) + SzTot2FromCombined(c3);
          if (sz2 != 0) continue;  // Sz_total = 0
          std::vector<std::vector<size_t>> grid(Ly, std::vector<size_t>(Lx, 0));
          grid[0][0] = c0;
          grid[0][1] = c1;
          grid[1][0] = c2;
          grid[1][1] = c3;
          out.emplace_back(grid);
        }
      }
    }
  }

  if (out.empty()) {
    throw std::runtime_error("ExactSum: empty sector on 2x2 (check Ne parity / sector constraints).");
  }
  return out;
}

template <typename TenElemT, typename QNT>
std::vector<qlpeps::Configuration> FilterConfigsByNonEmptySplitTensors(
    const qlpeps::SplitIndexTPS<TenElemT, QNT> &sitps,
    const std::vector<qlpeps::Configuration> &configs,
    size_t max_report = 10) {
  // Some physical components can be structurally forbidden by QN flow and stored as "empty tensors".
  // Those configurations have psi(config)=0 exactly. They must be skipped; exact summation should not crash on them.
  std::vector<qlpeps::Configuration> kept;
  kept.reserve(configs.size());

  size_t dropped = 0;
  size_t reported = 0;
  for (const auto &cfg : configs) {
    bool ok = true;
    for (size_t row = 0; row < sitps.rows() && ok; ++row) {
      for (size_t col = 0; col < sitps.cols(); ++col) {
        const size_t c = cfg({row, col});
        const auto &comp = sitps({row, col})[c];
        if (comp.GetActualDataSize() == 0) {
          ok = false;
          ++dropped;
          if (reported < max_report) {
            std::cerr << "[ExactSum][drop] empty split tensor at site=(" << row << "," << col
                      << ") local_state=" << c << "\n";
            ++reported;
          }
        }
      }
    }
    if (ok) kept.push_back(cfg);
  }

  if (kept.empty()) {
    throw std::runtime_error("ExactSum: all configurations got filtered out by empty split tensors (psi=0 everywhere).");
  }
  if (dropped > 0) {
    std::cout << "[ExactSum] filtered out " << dropped << " / " << configs.size()
              << " configurations due to empty split tensors (psi=0).\n";
  }
  return kept;
}

template <typename TenElemT, typename QNT, typename ModelT>
std::vector<qlpeps::Configuration> FilterConfigsBySuccessfulBMPS(
    const qlpeps::SplitIndexTPS<TenElemT, QNT> &sitps,
    const std::vector<qlpeps::Configuration> &configs,
    const typename qlpeps::BMPSContractor<TenElemT, QNT>::TruncateParams &trun_para,
    ModelT &model,
    size_t Ly,
    size_t Lx,
    size_t max_report = 10) {
  // Even if all chosen split tensors are non-empty, BMPS contraction may still produce empty tensors
  // for configurations with exactly zero amplitude due to global QN incompatibility/cancellation.
  // Probe by actually constructing TPSWaveFunctionComponent (and optionally a local-energy call).
  std::vector<qlpeps::Configuration> kept;
  kept.reserve(configs.size());

  size_t dropped = 0;
  size_t reported = 0;
  for (size_t i = 0; i < configs.size(); ++i) {
    const auto &cfg = configs[i];
    try {
      qlpeps::TPSWaveFunctionComponent<TenElemT, QNT, qlpeps::NoDress, qlpeps::BMPSContractor>
          tps_sample(sitps, cfg, trun_para);

      const double w = std::norm(tps_sample.amplitude);
      if (!(std::isfinite(w)) || w <= 0.0) {
        ++dropped;
        continue;
      }

      // Also probe the local-energy path (this is exactly what ExactSumEnergyEvaluatorMPI will do).
      qlpeps::TensorNetwork2D<TenElemT, QNT> holes_dag(Ly, Lx);
      (void) model.template CalEnergyAndHoles<TenElemT, QNT, true>(
          &sitps, &tps_sample, holes_dag);

      kept.push_back(cfg);
    } catch (const std::exception &e) {
      ++dropped;
      if (reported < max_report) {
        std::cerr << "[ExactSum][drop] BMPS/energy failed for config #" << i
                  << " reason: " << e.what() << "\n";
        ++reported;
      }
    }
  }

  if (kept.empty()) {
    throw std::runtime_error("ExactSum: all configurations got filtered out by BMPS/energy probing.");
  }
  if (dropped > 0) {
    std::cout << "[ExactSum] filtered out " << dropped << " / " << configs.size()
              << " configurations due to BMPS/energy probe failures (treated as psi=0).\n";
  }
  return kept;
}

template <typename TenElemT, typename QNT>
void MaterializeDefaultSplitTensorsInPlace(qlpeps::SplitIndexTPS<TenElemT, QNT> &sitps) {
  // Some split tensors are stored as "default" to represent exact zeros (e.g. forbidden by symmetry).
  // Unfortunately, parts of qlpeps optimizer path assume tensors are non-default. So we materialize
  // defaults into explicit zero tensors with the same index structure as any non-default component
  // on the same site.
  for (size_t row = 0; row < sitps.rows(); ++row) {
    for (size_t col = 0; col < sitps.cols(); ++col) {
      auto &vec = sitps({row, col});
      size_t tmpl_idx = static_cast<size_t>(-1);
      for (size_t i = 0; i < vec.size(); ++i) {
        if (!vec[i].IsDefault()) {
          tmpl_idx = i;
          break;
        }
      }
      if (tmpl_idx == static_cast<size_t>(-1)) {
        throw std::runtime_error("ExactSum: site has all-default split tensors (unexpected).");
      }
      const auto tmpl = vec[tmpl_idx];
      for (size_t i = 0; i < vec.size(); ++i) {
        if (vec[i].IsDefault()) {
          auto z = tmpl;
          z *= TenElemT(0.0);
          vec[i] = z;
        }
      }
    }
  }
}

template <typename TenElemT, typename QNT>
void MaterializeDefaultGradientInPlace(
    qlpeps::SplitIndexTPS<TenElemT, QNT> &grad,
    const qlpeps::SplitIndexTPS<TenElemT, QNT> &state) {
  for (size_t row = 0; row < grad.rows(); ++row) {
    for (size_t col = 0; col < grad.cols(); ++col) {
      for (size_t i = 0; i < grad({row, col}).size(); ++i) {
        if (!grad({row, col})[i].IsDefault()) continue;
        // Use state's component as template (should already be materialized).
        auto z = state({row, col})[i];
        z *= TenElemT(0.0);
        grad({row, col})[i] = z;
      }
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <physics.json> <algo.json>\n";
    return 1;
  }

  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0, mpi_size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);

  try {
    peps_kondo_params::PhysicalParams phys(argv[1]);
    ExactSumAlgoParams algo(argv[2]);

    if (phys.Lx != 2 || phys.Ly != 2) {
      throw std::runtime_error("ExactSum: this tool only supports Lx=2, Ly=2.");
    }
    peps_kondo_params::EnforceRestrictedSectorOrDie(phys.Lx, phys.Ly, phys.ElectronNum, "exact_sum_optimize");

    // Load initial wavefunction from local tpsfinal/ (must be produced by simple_update).
    const std::string tps_final = "tpsfinal";
    if (!qlmps::IsPathExist(tps_final)) {
      throw std::runtime_error("ExactSum: missing ./tpsfinal (run simple_update in this directory first).");
    }

    qlpeps::SplitIndexTPS<TenElemT, QNT> sitps(phys.Ly, phys.Lx);
    sitps.Load(tps_final);
    MaterializeDefaultSplitTensorsInPlace(sitps);

    // Model solver
    peps_kondo::SquareKondoModel model(phys.t, phys.U, phys.JK, phys.mu);

    // BMPS truncation parameters for exact contraction
    using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
    qlpeps::BMPSTruncateParams<RealT> trun_para(
        algo.Db_min, algo.Db_max, algo.trunc_err, algo.mps_compress_scheme,
        std::make_optional<double>(algo.trunc_err), std::make_optional<size_t>(10));

    // Pure optimizer
    qlpeps::OptimizerParams opt_params = BuildOptimizerParams(algo);
    qlpeps::Optimizer<TenElemT, QNT> optimizer(opt_params, comm, rank, mpi_size);

    // Enumerate configs in the desired sector (Ne fixed, Sz_total=0), then filter out configs
    // that lead to empty intermediate tensors in BMPS contraction (psi=0 cases).
    auto all_configs_raw = GenerateAll2x2ConfigsSector(phys.ElectronNum);
    auto all_configs_nonempty = FilterConfigsByNonEmptySplitTensors(sitps, all_configs_raw);
    auto all_configs = FilterConfigsBySuccessfulBMPS(sitps, all_configs_nonempty, trun_para, model, phys.Ly, phys.Lx);
    if (rank == 0) {
      std::cout << "[ExactSum] total configs in sector (after filtering): " << all_configs.size() << "\n";
    }

    auto evaluator = [&](const qlpeps::SplitIndexTPS<TenElemT, QNT> &state)
        -> std::tuple<TenElemT, qlpeps::SplitIndexTPS<TenElemT, QNT>, double> {
      auto [energy, grad, err] = qlpeps::ExactSumEnergyEvaluatorMPI<peps_kondo::SquareKondoModel, TenElemT, QNT>(
          state, all_configs, trun_para, model, phys.Ly, phys.Lx, comm, rank, mpi_size);
      MaterializeDefaultGradientInPlace(grad, state);
      return {energy, grad, err};  // err==0 for exact sum
    };

    typename qlpeps::Optimizer<TenElemT, QNT>::OptimizationCallback cb;
    cb.on_iteration = [rank](size_t it, double e, double eerr, double gnorm) {
      if (rank == 0) {
        std::cout << "Iter " << std::setw(3) << it
                  << "  E0=" << std::setw(16) << std::fixed << std::setprecision(12) << e
                  << "  ||grad||=" << std::scientific << std::setprecision(3) << gnorm
                  << "  err=" << std::fixed << std::setprecision(2) << eerr
                  << "\n";
      }
    };

    auto result = optimizer.IterativeOptimize(sitps, evaluator, cb);
    if (rank == 0) {
      std::cout << "[ExactSum] Final energy: " << std::fixed << std::setprecision(12)
                << std::real(result.final_energy) << "\n";
    }

    // Save optimized state back to tpsfinal/ for resume capability
    MPI_Barrier(comm);
    if (rank == 0) {
      std::cout << "[ExactSum] Saving optimized state to " << tps_final << "/ ..." << std::endl;
      result.optimized_state.Dump(tps_final);
      std::cout << "[ExactSum] State saved. You can re-run this command to continue optimization." << std::endl;
    }
    MPI_Barrier(comm);
  } catch (const std::exception &e) {
    if (rank == 0) std::cerr << "ERROR: " << e.what() << "\n";
    MPI_Finalize();
    return 2;
  }

  MPI_Finalize();
  return 0;
}


