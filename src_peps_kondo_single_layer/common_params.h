// SPDX-License-Identifier: MIT
/*
 * Single-layer Kondo lattice PEPS params.
 *
 * Keep JSON schema intentionally close to finite-size_PEPS_tJ for consistency:
 * - physics_params.json holds model parameters and lattice sizes
 * - simple_update_algo.json holds SU algorithm parameters
 */
#ifndef LNO_PEPS_KONDO_SINGLE_LAYER_COMMON_PARAMS_H
#define LNO_PEPS_KONDO_SINGLE_LAYER_COMMON_PARAMS_H

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "qlmps/case_params_parser.h"

namespace peps_kondo_params {

// Project-wide symmetry/sector restriction (requested):
// - Total itinerant electron number Ne must be even.
// - Total Sz_total is restricted to 0 in our current workflow.
//
// Since Sz_total includes local spins, this implies on even-site OBC clusters:
//   - Lx*Ly must be even (otherwise with Ne even, Sz_total=0 is impossible by parity).
//
// Keep this check centralized so SU/VMC/measure all fail fast in the same way.
inline void EnforceRestrictedSectorOrDie(size_t Lx, size_t Ly, size_t electron_num, const std::string &who) {
  const size_t N = Lx * Ly;
  if ((electron_num % 2) != 0) {
    throw std::runtime_error(who + ": restricted workflow requires even ElectronNum (Ne).");
  }
  if ((N % 2) != 0) {
    throw std::runtime_error(who + ": restricted workflow requires even #sites (Lx*Ly even) for Sz_total=0 with Ne even.");
  }
}

struct PhysicalParams : public qlmps::CaseParamsParserBasic {
  size_t Lx{0};
  size_t Ly{0};

  // NN hopping amplitudes (baseline: uniform t; t2 reserved for future NN bond-order pattern)
  double t{0.0};
  double t2{0.0};

  // On-site Hubbard U for itinerant electrons.
  double U{0.0};

  // Kondo/Hund coupling: H_K = JK * (s · S). FM corresponds to JK < 0.
  double JK{0.0};

  // Chemical potential for itinerant electron number control (optional).
  double mu{0.0};

  // Optional: initial electron filling controls for Simple Update initial product state.
  // NOTE: the Hamiltonian implemented in SU conserves electron number, so ElectronNum must be set
  // if you want a non-trivial finite-density wavefunction.
  size_t ElectronNum{0};   // total number of itinerant electrons (0..2*Lx*Ly)
  int ElectronSz2{0};      // 2*Sz of itinerant electrons (integer), default 0

  explicit PhysicalParams(const char *physics_file) : qlmps::CaseParamsParserBasic(physics_file) {
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
    t = ParseDouble("t");
    t2 = ParseDoubleOr("t2", 0.0);
    U = ParseDoubleOr("U", 0.0);
    JK = ParseDoubleOr("Jk", 0.0);
    mu = ParseDoubleOr("Mu", 0.0);

    const size_t N = Lx * Ly;
    // Default to a safe sector: make the default electron number even.
    const size_t default_ne = (N / 2) & ~static_cast<size_t>(1);
    ElectronNum = static_cast<size_t>(ParseIntOr("ElectronNum", static_cast<int>(default_ne)));
    ElectronSz2 = ParseIntOr("ElectronSz2", 0);
  }
};

struct SimpleUpdateAlgorithmParams : public qlmps::CaseParamsParserBasic {
  size_t Dmin{0};
  size_t Dmax{0};
  double TruncErr{1e-10};
  double Tau{0.01};
  size_t Step{0};
  size_t ThreadNum{1};

  explicit SimpleUpdateAlgorithmParams(const char *algo_file) : qlmps::CaseParamsParserBasic(algo_file) {
    Dmin = ParseInt("Dmin");
    Dmax = ParseInt("Dmax");
    TruncErr = ParseDoubleOr("TruncErr", 1e-10);
    Tau = ParseDouble("Tau");
    Step = ParseInt("Step");
    ThreadNum = ParseIntOr("ThreadNum", 1);
  }
};

struct Params {
  PhysicalParams physical;
  SimpleUpdateAlgorithmParams algo;

  Params(const char *physics_file, const char *algo_file) : physical(physics_file), algo(algo_file) {}
};

}  // namespace peps_kondo_params

#endif  // LNO_PEPS_KONDO_SINGLE_LAYER_COMMON_PARAMS_H


