// SPDX-License-Identifier: MIT
/*
 * Parameter system for PEPS single-layer Kondo lattice.
 *
 * This keeps the Kondo PEPS VMC parser aligned with the Heisenberg-style SR/MinSR
 * schema while preserving the two-file physics/algorithm split used here.
 *
 * The goal is simple: keep parameters boring, explicit, and hard to misuse.
 */
#ifndef LNO_PEPS_KONDO_ENHANCED_PARAMS_PARSER_H
#define LNO_PEPS_KONDO_ENHANCED_PARAMS_PARSER_H

#include <optional>
#include <string>

#include "qlmps/case_params_parser.h"
#include "qlpeps/qlpeps.h"
#include "qlpeps/algorithm/vmc_update/vmc_peps_optimizer_params.h"
#include "qlpeps/algorithm/vmc_update/monte_carlo_peps_params.h"

#include "./common_params.h"
#include "./mc_measure_params.h" // BuildAndMaybeLoadConfigurationKondo + MC/BMPS numerical params
#include "./optimizer_params_compat.h"

namespace peps_kondo_params {

/**
 * Unified VMC optimization params (physics + algorithm).
 *
 * JSON keys (CaseParams):
 * - OptimizerType, MaxIterations, LearningRate, (optimizer-specific fields)
 * - WavefunctionBase, ConfigurationLoadDir, ConfigurationDumpDir, TPSDumpPath
 * - MC_samples, WarmUp, MCLocalUpdateSweepsBetweenSample
 * - Db_min/Db_max/TruncErr/MPSCompressScheme/ThreadNum
 * - ElectronNum/ElectronSz2/AllowDoublon/LocalSpinNeel
 */
struct EnhancedVMCOptimizeParams : public qlmps::CaseParamsParserBasic {
  EnhancedVMCOptimizeParams(const char *physics_file, const char *algorithm_file)
      : qlmps::CaseParamsParserBasic(algorithm_file),
        physical_params(physics_file),
        mc_params(algorithm_file),
        bmps_params(algorithm_file) {
    optimizer_params = OptimizerCompatParams(
        *this,
        OptimizerParseOptions{
            .default_optimizer_type = "",
            .require_optimizer_type = true,
            .default_max_iterations = 0,
            .require_max_iterations = true,
            .default_learning_rate = 0.0,
            .require_learning_rate = true,
            .require_adam_params = true,
            .require_adagrad_params = true,
        });

    // IO
    wavefunction_base = ParseStrOr("WavefunctionBase", "tps");
    configuration_load_dir = ParseStrOr("ConfigurationLoadDir", wavefunction_base + std::string("final"));
    configuration_dump_dir = ParseStrOr("ConfigurationDumpDir", wavefunction_base + std::string("final"));
    tps_dump_path = ParseStrOr("TPSDumpPath", "./");

    // Sector control (must match the wavefunction sector, otherwise psi=0 footgun)
    electron_num = static_cast<size_t>(ParseIntOr("ElectronNum", static_cast<int>(physical_params.ElectronNum)));
    sz2_electron = ParseIntOr("ElectronSz2", physical_params.ElectronSz2);
    allow_doublon = ParseBoolOr("AllowDoublon", false);
    local_spin_neel = ParseBoolOr("LocalSpinNeel", true);
    EnforceRestrictedSectorOrDie(physical_params.Lx, physical_params.Ly, electron_num, "vmc_optimize params");
    if (!allow_doublon) {
      EnforceNoDoublonInitializerSectorOrDie(
          physical_params.Lx, physical_params.Ly, electron_num, sz2_electron, "vmc_optimize params");
    }

    // Optional gradient clipping (kept as optional, not forced)
    double tmp = 0.0;
    if (TryParseDouble("ClipNorm", tmp)) clip_norm = tmp; else clip_norm.reset();
    if (TryParseDouble("ClipValue", tmp)) clip_value = tmp; else clip_value.reset();
  }

  PhysicalParams physical_params;
  MonteCarloNumericalParams mc_params;
  BMPSParams bmps_params;

  // IO configuration
  std::string wavefunction_base = "tps";
  std::string configuration_load_dir = "tpsfinal";
  std::string configuration_dump_dir = "tpsfinal";
  std::string tps_dump_path = "./";

  // Sector control for configuration
  size_t electron_num = 0;
  int sz2_electron = 0;
  bool allow_doublon = false;
  bool local_spin_neel = true;

  // Optimizer configuration
  OptimizerCompatParams optimizer_params;

  // Optional gradient clipping
  std::optional<double> clip_norm;
  std::optional<double> clip_value;

  qlpeps::VMCPEPSOptimizerParams CreateVMCOptimizerParams(int rank) {
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
        qlpeps::BMPSTruncateParams<double>(bmps_params.Db_min, bmps_params.Db_max,
                                 bmps_params.TruncErr,
                                 bmps_params.MPSCompressScheme,
                                 std::make_optional<double>(bmps_params.TruncErr),
                                 std::make_optional<size_t>(10)));

    return qlpeps::VMCPEPSOptimizerParams(
        optimizer_params.CreateOptimizerParams(),
        mc_params_obj,
        peps_params_obj,
        tps_dump_path);
  }
};

} // namespace peps_kondo_params

#endif // LNO_PEPS_KONDO_ENHANCED_PARAMS_PARSER_H
