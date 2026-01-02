// SPDX-License-Identifier: MIT
/*
 * Parameter system for PEPS single-layer Kondo lattice.
 *
 * This intentionally mirrors finite-size_PEPS_tJ/src/enhanced_params_parser.h:
 * - one "enhanced" parser that owns physics + algorithm params
 * - helper builders that return qlpeps::{VMCPEPSOptimizerParams, MCMeasurementParams}
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
#include "qlpeps/optimizer/optimizer_params.h"

#include "./common_params.h"
#include "./mc_measure_params.h" // BuildAndMaybeLoadConfigurationKondo + MC/BMPS numerical params

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
    optimizer_type = ParseStr("OptimizerType");
    if (optimizer_type == "SR" || optimizer_type == "sr") optimizer_type = "StochasticReconfiguration";

    max_iterations = static_cast<size_t>(ParseInt("MaxIterations"));
    learning_rate = ParseDouble("LearningRate");
    energy_tolerance = ParseDoubleOr("EnergyTolerance", 0.0);
    gradient_tolerance = ParseDoubleOr("GradientTolerance", 0.0);
    plateau_patience = static_cast<size_t>(ParseIntOr("PlateauPatience", static_cast<int>(max_iterations)));

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

    // Optimizer-specific knobs
    if (optimizer_type == "SGD") {
      momentum = ParseDoubleOr("Momentum", 0.0);
      nesterov = ParseBoolOr("Nesterov", false);
      weight_decay = ParseDoubleOr("WeightDecay", 0.0);
    } else if (optimizer_type == "Adam") {
      beta1 = ParseDouble("Beta1");
      beta2 = ParseDouble("Beta2");
      epsilon = ParseDouble("Epsilon");
      weight_decay = ParseDouble("WeightDecay");
    } else if (optimizer_type == "AdaGrad") {
      epsilon = ParseDouble("Epsilon");
      initial_accumulator = ParseDouble("InitialAccumulator");
    } else {
      // Default and "StochasticReconfiguration"
      cg_max_iter = static_cast<size_t>(ParseIntOr("CGMaxIter", 100));
      cg_tol = ParseDoubleOr("CGTol", 1e-8);
      cg_residue_restart = static_cast<size_t>(ParseIntOr("CGResidueRestart", 20));
      cg_diag_shift = ParseDoubleOr("CGDiagShift", 0.01);
      normalize_update = ParseBoolOr("NormalizeUpdate", false);
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
  std::string optimizer_type;
  size_t max_iterations = 0;
  double learning_rate = 0.0;
  double energy_tolerance = 0.0;
  double gradient_tolerance = 0.0;
  size_t plateau_patience = 0;

  // SGD
  double momentum = 0.0;
  bool nesterov = false;
  double weight_decay = 0.0;

  // Adam
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;

  // AdaGrad
  double initial_accumulator = 0.0;

  // SR
  size_t cg_max_iter = 100;
  double cg_tol = 1e-8;
  size_t cg_residue_restart = 20;
  double cg_diag_shift = 0.01;
  bool normalize_update = false;

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
        qlpeps::BMPSTruncatePara(bmps_params.Db_min, bmps_params.Db_max,
                                 bmps_params.TruncErr,
                                 bmps_params.MPSCompressScheme,
                                 std::make_optional<double>(bmps_params.TruncErr),
                                 std::make_optional<size_t>(10)));

    qlpeps::OptimizerParams::BaseParams base_params(
        max_iterations, energy_tolerance, gradient_tolerance, plateau_patience,
        learning_rate, nullptr);

    qlpeps::OptimizerParams opt;
    if (optimizer_type == "SGD") {
      opt = qlpeps::OptimizerParams(base_params, qlpeps::SGDParams(momentum, nesterov, weight_decay));
    } else if (optimizer_type == "Adam") {
      opt = qlpeps::OptimizerParams(base_params, qlpeps::AdamParams(beta1, beta2, epsilon, weight_decay));
    } else if (optimizer_type == "AdaGrad") {
      opt = qlpeps::OptimizerParams(base_params, qlpeps::AdaGradParams(epsilon, initial_accumulator));
    } else {
      qlpeps::ConjugateGradientParams cg(cg_max_iter, cg_tol, cg_residue_restart, cg_diag_shift);
      opt = qlpeps::OptimizerParams(base_params, qlpeps::StochasticReconfigurationParams(cg, normalize_update));
    }

    return qlpeps::VMCPEPSOptimizerParams(opt, mc_params_obj, peps_params_obj, tps_dump_path);
  }
};

} // namespace peps_kondo_params

#endif // LNO_PEPS_KONDO_ENHANCED_PARAMS_PARSER_H


