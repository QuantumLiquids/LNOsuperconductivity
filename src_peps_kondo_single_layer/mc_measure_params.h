// SPDX-License-Identifier: MIT
/*
 * Monte-Carlo measurement params for PEPS Kondo lattice.
 *
 * Mirrors finite-size_PEPS_tJ's two-file parameter style:
 * - physics_params.json: lattice + couplings
 * - mc_measure_algorithm_params.json: BMPS + MC sampling params + IO
 */
#ifndef LNO_PEPS_KONDO_MC_MEASURE_PARAMS_H
#define LNO_PEPS_KONDO_MC_MEASURE_PARAMS_H

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include "qlmps/case_params_parser.h"
#include "qlpeps/qlpeps.h"

#include "./common_params.h"

namespace peps_kondo_params {

inline int ElectronNumFromCombinedState(size_t c) {
  const size_t e = c / 2; // 0=D, 1=U, 2=d, 3=0
  if (e == 0) return 2;
  if (e == 1 || e == 2) return 1;
  return 0;
}

inline int ElectronSz2FromCombinedState(size_t c) {
  const size_t e = c / 2;
  if (e == 1) return +1;
  if (e == 2) return -1;
  return 0;
}

inline int LocalSz2FromCombinedState(size_t c) {
  const size_t s = c % 2; // 0=Up => +1, 1=Dn => -1
  return (s == 0) ? +1 : -1;
}

inline int TotalSz2FromCombinedState(size_t c) {
  return ElectronSz2FromCombinedState(c) + LocalSz2FromCombinedState(c);
}

inline void EnforceRestrictedSectorOnConfigurationOrDie(const qlpeps::Configuration &cfg, const std::string &who) {
  long ne = 0;
  long sz2 = 0;
  for (const size_t &c : cfg) {
    ne += ElectronNumFromCombinedState(c);
    sz2 += TotalSz2FromCombinedState(c);
  }
  if ((ne % 2) != 0) {
    throw std::runtime_error(who + ": restricted workflow requires even total Ne, but configuration has Ne=" + std::to_string(ne));
  }
  if (sz2 != 0) {
    throw std::runtime_error(who + ": restricted workflow requires total Sz2_total=0, but configuration has Sz2_total=" + std::to_string(sz2));
  }
}

struct BMPSParams : public qlmps::CaseParamsParserBasic {
  size_t Db_min{4};
  size_t Db_max{8};
  double TruncErr{1e-10};
  qlpeps::CompressMPSScheme MPSCompressScheme{};
  size_t ThreadNum{1};

  explicit BMPSParams(const char *algo_file) : qlmps::CaseParamsParserBasic(algo_file) {
    Db_min = ParseIntOr("Dbmps_min", ParseIntOr("Db_min", 4));
    Db_max = ParseIntOr("Dbmps_max", ParseIntOr("Db_max", 8));
    TruncErr = ParseDoubleOr("TruncErr", 1e-10);
    MPSCompressScheme = static_cast<qlpeps::CompressMPSScheme>(ParseIntOr("MPSCompressScheme", 0));
    ThreadNum = ParseIntOr("ThreadNum", 1);
  }
};

struct MonteCarloNumericalParams : public qlmps::CaseParamsParserBasic {
  size_t MC_samples{1000};
  size_t WarmUp{200};
  size_t MCLocalUpdateSweepsBetweenSample{10};

  explicit MonteCarloNumericalParams(const char *algo_file) : qlmps::CaseParamsParserBasic(algo_file) {
    MC_samples = ParseIntOr("MC_samples", 1000);
    WarmUp = ParseIntOr("WarmUp", 200);
    MCLocalUpdateSweepsBetweenSample = ParseIntOr("MCLocalUpdateSweepsBetweenSample", 10);
  }
};

struct LoadedConfigurationResult {
  qlpeps::Configuration config;
  bool warmed_up;
};

// Build a simple initial configuration for the 8D Kondo site:
// - itinerant electrons: fixed total Ne with Nup/Ndn (no doublons by default)
// - local spins: Neel by default
inline LoadedConfigurationResult BuildAndMaybeLoadConfigurationKondo(
    size_t Lx,
    size_t Ly,
    const std::string &load_dir,
    int rank,
    size_t electron_num,
    int sz2_electron = 0,        // 2*Sz_e (integer)
    bool allow_doublon = false,
    bool local_spin_neel = true
) {
  qlpeps::Configuration config(Ly, Lx);

  bool warmed_up = false;
  if (!load_dir.empty()) {
    const bool ok = config.Load(load_dir, rank);
    warmed_up = ok;
    if (warmed_up) {
      EnforceRestrictedSectorOnConfigurationOrDie(config, "mc_measure/vmc (loaded config)");
      std::cout << "[rank " << rank << "] Loaded configuration from: " << load_dir << std::endl;
      return {std::move(config), true};
    }
  }

  // Build from scratch
  const size_t N = Lx * Ly;
  EnforceRestrictedSectorOrDie(Lx, Ly, electron_num, "mc_measure/vmc (config generator)");
  if (electron_num > 2 * N) electron_num = 2 * N;

  // Default: no doublons unless explicitly requested (keep it simple)
  size_t Nd = 0;
  if (allow_doublon) {
    // naive: try to pack a small fraction of doublons; user should tune via config file later
    Nd = 0;
  }

  // Compute Nup/Ndn from Ne and Sz2
  // Ne = Nup + Ndn + 2*Nd  (if doublons allowed; here Nd=0 typically)
  const long Ne_single = static_cast<long>(electron_num) - 2L * static_cast<long>(Nd);
  long Nup = (Ne_single + static_cast<long>(sz2_electron)) / 2;
  long Ndn = Ne_single - Nup;
  if (Nup < 0) Nup = 0;
  if (Ndn < 0) Ndn = 0;

  // Place electrons into site list: label is combined state index 0..7 (see qldouble.h).
  // electron basis: D(0), U(1), d(2), 0(3)
  // spin basis    : Up(0), Dn(1)
  std::vector<size_t> site_states;
  site_states.reserve(N);

  // Start with all empty electron, local spin to be set later
  for (size_t i = 0; i < N; ++i) site_states.push_back(0); // placeholder

  // Build a list of electron labels per site (without local spin)
  std::vector<int> e_labels;
  e_labels.reserve(N);
  for (size_t i = 0; i < Nd; ++i) e_labels.push_back(0); // doublon
  for (size_t i = 0; i < static_cast<size_t>(Nup); ++i) e_labels.push_back(1); // up
  for (size_t i = 0; i < static_cast<size_t>(Ndn); ++i) e_labels.push_back(2); // down
  while (e_labels.size() < N) e_labels.push_back(3); // empty
  e_labels.resize(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(e_labels.begin(), e_labels.end(), gen);

  // Fill 2D
  std::vector<std::vector<size_t>> grid(Ly, std::vector<size_t>(Lx, 0));
  size_t idx = 0;

  // Enforce total Sz_total = 0 by choosing local spins such that:
  //   Sz2_total = Sz2_e + Sz2_local = 0
  // local Sz2 = 2*Nup_local - N  =>  Nup_local = (N - Sz2_e)/2
  if ((static_cast<long>(N) - static_cast<long>(sz2_electron)) % 2 != 0) {
    throw std::runtime_error("mc_measure/vmc (config generator): cannot satisfy Sz_total=0 due to parity mismatch.");
  }
  const long Nup_local = (static_cast<long>(N) - static_cast<long>(sz2_electron)) / 2;
  if (Nup_local < 0 || Nup_local > static_cast<long>(N)) {
    throw std::runtime_error("mc_measure/vmc (config generator): cannot satisfy Sz_total=0 (invalid local spin magnetization).");
  }
  std::vector<int> s_labels;
  s_labels.reserve(N);
  if (local_spin_neel && sz2_electron == 0) {
    // Strict Neel only when it is compatible with Sz_total=0.
    for (size_t y = 0; y < Ly; ++y) {
      for (size_t x = 0; x < Lx; ++x) {
        s_labels.push_back(((x + y) % 2 == 0) ? 0 : 1);
      }
    }
  } else {
    for (size_t i = 0; i < static_cast<size_t>(Nup_local); ++i) s_labels.push_back(0);
    while (s_labels.size() < N) s_labels.push_back(1);
    std::shuffle(s_labels.begin(), s_labels.end(), gen);
  }

  for (size_t y = 0; y < Ly; ++y) {
    for (size_t x = 0; x < Lx; ++x) {
      const int e = e_labels[idx++];
      const int s = s_labels[(y * Lx) + x];
      grid[y][x] = static_cast<size_t>(2 * e + s);
    }
  }

  config = qlpeps::Configuration(grid);
  EnforceRestrictedSectorOnConfigurationOrDie(config, "mc_measure/vmc (generated config)");
  std::cout << "[rank " << rank << "] Initial configuration generated (random electrons, "
            << (local_spin_neel ? "Neel local spins" : "random local spins") << ")." << std::endl;
  return {std::move(config), false};
}

struct EnhancedMCMeasureParams : public qlmps::CaseParamsParserBasic {
  EnhancedMCMeasureParams(const char *physics_file, const char *measure_file)
      : qlmps::CaseParamsParserBasic(measure_file),
        physical_params(physics_file),
        mc_params(measure_file),
        bmps_params(measure_file) {
    wavefunction_base = ParseStrOr("WavefunctionBase", "tps");
    configuration_load_dir = ParseStrOr("ConfigurationLoadDir", wavefunction_base + std::string("final"));
    configuration_dump_dir = ParseStrOr("ConfigurationDumpDir", wavefunction_base + std::string("final"));
    measurement_dump_dir = ParseStrOr("MeasurementDumpDir", "./");

    // Updater options:
    // - KondoNNConserved: custom NN updater (recommended) that conserves Ne and Sz_total
    // - NNExchange: plain NN swap (usually too restrictive; kept for debugging)
    // - TNN3SiteExchange: 3-site permutation update (still preserves state counts)
    updater = ParseStrOr("Updater", "KondoNNConserved");

    // For NNExchange updater, Ne is conserved. Default to the same sector used by SU initializer.
    electron_num = static_cast<size_t>(ParseIntOr("ElectronNum", static_cast<int>(physical_params.ElectronNum)));
    sz2_electron = ParseIntOr("ElectronSz2", physical_params.ElectronSz2);
    allow_doublon = ParseBoolOr("AllowDoublon", false);
    local_spin_neel = ParseBoolOr("LocalSpinNeel", true);

    // Fail-fast: project restriction (Ne even, Sz_total=0 feasible).
    EnforceRestrictedSectorOrDie(physical_params.Lx, physical_params.Ly, electron_num, "mc_measure params");
  }

  PhysicalParams physical_params;
  MonteCarloNumericalParams mc_params;
  BMPSParams bmps_params;

  std::string wavefunction_base = "tps";
  std::string configuration_load_dir = "tpsfinal";
  std::string configuration_dump_dir = "tpsfinal";
  std::string measurement_dump_dir = "./";
  std::string updater = "NNExchange";

  size_t electron_num = 0;
  int sz2_electron = 0;
  bool allow_doublon = false;
  bool local_spin_neel = true;

  qlpeps::MCMeasurementParams CreateMCMeasurementParams(int rank = 0) {
    auto cfg = BuildAndMaybeLoadConfigurationKondo(
        physical_params.Lx,
        physical_params.Ly,
        configuration_load_dir,
        rank,
        electron_num,
        sz2_electron,
        allow_doublon,
        local_spin_neel);

    qlpeps::MonteCarloParams mc_params_obj(
        mc_params.MC_samples,
        mc_params.WarmUp,
        mc_params.MCLocalUpdateSweepsBetweenSample,
        cfg.config,
        cfg.warmed_up,
        configuration_dump_dir);

    qlpeps::PEPSParams peps_params_obj(
        qlpeps::BMPSTruncatePara(bmps_params.Db_min, bmps_params.Db_max,
                                 bmps_params.TruncErr,
                                 bmps_params.MPSCompressScheme,
                                 std::make_optional<double>(bmps_params.TruncErr),
                                 std::make_optional<size_t>(10)));

    qlpeps::MCMeasurementParams out(mc_params_obj, peps_params_obj, measurement_dump_dir);
    return out;
  }
};

}  // namespace peps_kondo_params

#endif  // LNO_PEPS_KONDO_MC_MEASURE_PARAMS_H


