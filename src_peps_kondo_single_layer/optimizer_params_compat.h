// SPDX-License-Identifier: MIT
/*
 * Shared optimizer parameter compatibility layer for Kondo PEPS.
 *
 * Canonical schema follows ../HeisenbergVMCPEPS for SR/MinSR naming while
 * accepting deprecated aliases from older Kondo PEPS and finite-size_PEPS_tJ.
 */
#ifndef LNO_PEPS_KONDO_OPTIMIZER_PARAMS_COMPAT_H
#define LNO_PEPS_KONDO_OPTIMIZER_PARAMS_COMPAT_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "qlmps/case_params_parser.h"
#include "qlpeps/optimizer/optimizer_params.h"

namespace peps_kondo_params {

struct OptimizerParseOptions {
  std::string default_optimizer_type;
  bool require_optimizer_type = true;
  size_t default_max_iterations = 0;
  bool require_max_iterations = true;
  double default_learning_rate = 0.0;
  bool require_learning_rate = true;
  bool require_adam_params = true;
  bool require_adagrad_params = true;
};

inline std::string TrimAsciiWhitespace(std::string text) {
  auto is_space = [](unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
  };
  while (!text.empty() && is_space(static_cast<unsigned char>(text.front()))) {
    text.erase(text.begin());
  }
  while (!text.empty() && is_space(static_cast<unsigned char>(text.back()))) {
    text.pop_back();
  }
  return text;
}

inline std::string ToLowerAscii(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

inline std::string NormalizeOptimizerType(const std::string &value) {
  const std::string key = ToLowerAscii(TrimAsciiWhitespace(value));
  if (key == "sr" || key == "stochasticreconfiguration") return "StochasticReconfiguration";
  if (key == "sgd") return "SGD";
  if (key == "adam") return "Adam";
  if (key == "adagrad") return "AdaGrad";
  if (key == "minsr" || key == "min-sr" || key == "min_sr") return "MinSR";
  return TrimAsciiWhitespace(value);
}

inline std::string ResolveAliasKey(
    qlmps::CaseParamsParserBasic &parser,
    const std::string &canonical_key,
    const std::string &legacy_key,
    const std::string &context = "") {
  const bool has_canonical = parser.Has(canonical_key);
  const bool has_legacy = parser.Has(legacy_key);
  if (has_canonical && has_legacy) {
    throw std::invalid_argument(
        "Ambiguous: both '" + legacy_key + "' (deprecated) and '" + canonical_key +
        "' are present. Remove '" + legacy_key + "'.");
  }
  if (has_legacy) {
    std::cerr << "[warn] JSON key '" << legacy_key << "' is deprecated; use '"
              << canonical_key << "' instead.";
    if (!context.empty()) std::cerr << " " << context;
    std::cerr << std::endl;
    return legacy_key;
  }
  return canonical_key;
}

inline double ParseCGRelativeTolerance(
    qlmps::CaseParamsParserBasic &parser,
    double default_value) {
  const bool has_canonical = parser.Has("CGRelativeTolerance");
  const bool has_legacy = parser.Has("CGTol");
  if (has_canonical && has_legacy) {
    throw std::invalid_argument(
        "Ambiguous: both 'CGTol' (deprecated) and 'CGRelativeTolerance' "
        "are present. Remove 'CGTol'.");
  }
  if (has_legacy) {
    const double legacy_value = parser.ParseDouble("CGTol");
    const double converted = std::sqrt(legacy_value);
    std::cerr << "[warn] JSON key 'CGTol' is deprecated; use "
                 "'CGRelativeTolerance' instead. Auto-converting: sqrt("
              << legacy_value << ") = " << converted << std::endl;
    return converted;
  }
  return parser.ParseDoubleOr("CGRelativeTolerance", default_value);
}

inline qlpeps::MinSRSolverMode ParseMinSRSolverMode(const std::string &value) {
  const std::string key = ToLowerAscii(TrimAsciiWhitespace(value));
  if (key == "auto" || key == "kauto") return qlpeps::MinSRSolverMode::kAuto;
  if (key == "replicated" || key == "kreplicated") return qlpeps::MinSRSolverMode::kReplicated;
  if (key == "distributed" || key == "kdistributed") return qlpeps::MinSRSolverMode::kDistributed;
  throw std::invalid_argument(
      "MinSRSolverMode must be one of: Auto, Replicated, Distributed");
}

inline double ParseMinSRRPinv(qlmps::CaseParamsParserBasic &parser) {
  const std::string key = ResolveAliasKey(
      parser, "MinSRRPinv", "MinSRRelativePInv");
  return parser.ParseDoubleOr(key, 1e-12);
}

inline double ParseMinSRAPinv(qlmps::CaseParamsParserBasic &parser) {
  const std::string key = ResolveAliasKey(
      parser, "MinSRAPinv", "MinSRAbsolutePInv");
  return parser.ParseDoubleOr(key, 0.0);
}

inline std::string ParseOptimizerType(
    qlmps::CaseParamsParserBasic &parser,
    const OptimizerParseOptions &options) {
  if (options.require_optimizer_type) {
    return NormalizeOptimizerType(parser.ParseStr("OptimizerType"));
  }
  return NormalizeOptimizerType(
      parser.ParseStrOr("OptimizerType", options.default_optimizer_type));
}

inline size_t ParseMaxIterations(
    qlmps::CaseParamsParserBasic &parser,
    const OptimizerParseOptions &options) {
  if (options.require_max_iterations) {
    return static_cast<size_t>(parser.ParseInt("MaxIterations"));
  }
  return static_cast<size_t>(
      parser.ParseIntOr("MaxIterations", static_cast<int>(options.default_max_iterations)));
}

inline double ParseLearningRate(
    qlmps::CaseParamsParserBasic &parser,
    const OptimizerParseOptions &options) {
  if (options.require_learning_rate) {
    return parser.ParseDouble("LearningRate");
  }
  return parser.ParseDoubleOr("LearningRate", options.default_learning_rate);
}

struct OptimizerCompatParams {
  std::string optimizer_type;
  size_t max_iterations = 0;
  double learning_rate = 0.0;
  double energy_tolerance = 0.0;
  double gradient_tolerance = 0.0;
  size_t plateau_patience = 0;

  double momentum = 0.0;
  bool nesterov = false;
  double weight_decay = 0.0;

  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;

  double initial_accumulator = 0.0;

  size_t cg_max_iter = 100;
  double cg_relative_tolerance = 1e-8;
  int cg_residual_recompute_interval = 20;
  double sr_diag_shift = 0.01;
  bool normalize_update = false;

  double minsr_r_pinv = 1e-12;
  double minsr_a_pinv = 0.0;
  bool minsr_soft_cutoff = true;
  qlpeps::MinSRSolverMode minsr_solver_mode = qlpeps::MinSRSolverMode::kAuto;

  OptimizerCompatParams() = default;

  OptimizerCompatParams(
      qlmps::CaseParamsParserBasic &parser,
      const OptimizerParseOptions &options) {
    optimizer_type = ParseOptimizerType(parser, options);
    max_iterations = ParseMaxIterations(parser, options);
    learning_rate = ParseLearningRate(parser, options);
    energy_tolerance = parser.ParseDoubleOr("EnergyTolerance", 0.0);
    gradient_tolerance = parser.ParseDoubleOr("GradientTolerance", 0.0);
    plateau_patience = static_cast<size_t>(
        parser.ParseIntOr("PlateauPatience", static_cast<int>(max_iterations)));

    if (optimizer_type == "SGD") {
      momentum = parser.ParseDoubleOr("Momentum", 0.0);
      nesterov = parser.ParseBoolOr("Nesterov", false);
      weight_decay = parser.ParseDoubleOr("WeightDecay", 0.0);
    } else if (optimizer_type == "Adam") {
      if (options.require_adam_params) {
        beta1 = parser.ParseDouble("Beta1");
        beta2 = parser.ParseDouble("Beta2");
        epsilon = parser.ParseDouble("Epsilon");
        weight_decay = parser.ParseDouble("WeightDecay");
      } else {
        beta1 = parser.ParseDoubleOr("Beta1", 0.9);
        beta2 = parser.ParseDoubleOr("Beta2", 0.999);
        epsilon = parser.ParseDoubleOr("Epsilon", 1e-8);
        weight_decay = parser.ParseDoubleOr("WeightDecay", 0.0);
      }
    } else if (optimizer_type == "AdaGrad") {
      if (options.require_adagrad_params) {
        epsilon = parser.ParseDouble("Epsilon");
        initial_accumulator = parser.ParseDouble("InitialAccumulator");
      } else {
        epsilon = parser.ParseDoubleOr("Epsilon", 1e-8);
        initial_accumulator = parser.ParseDoubleOr("InitialAccumulator", 0.0);
      }
    } else if (optimizer_type == "StochasticReconfiguration") {
      cg_max_iter = static_cast<size_t>(parser.ParseIntOr("CGMaxIter", 100));
      cg_relative_tolerance = ParseCGRelativeTolerance(parser, 1e-8);
      cg_residual_recompute_interval = parser.ParseIntOr(
          ResolveAliasKey(parser, "CGResidualRecomputeInterval", "CGResidueRestart"), 20);
      sr_diag_shift = parser.ParseDoubleOr(
          ResolveAliasKey(
              parser, "SRDiagShift", "CGDiagShift",
              "('diag_shift' moved from CG to SR in upstream PEPS.)"),
          0.01);
      normalize_update = parser.ParseBoolOr("NormalizeUpdate", false);
    } else if (optimizer_type == "MinSR") {
      minsr_r_pinv = ParseMinSRRPinv(parser);
      minsr_a_pinv = ParseMinSRAPinv(parser);
      minsr_soft_cutoff = parser.ParseBoolOr("MinSRSoftCutoff", true);
      minsr_solver_mode = ParseMinSRSolverMode(
          parser.ParseStrOr("MinSRSolverMode", "Auto"));
    }

    Validate();
  }

  void Validate() const {
    if (optimizer_type != "SGD" && optimizer_type != "Adam" &&
        optimizer_type != "AdaGrad" && optimizer_type != "StochasticReconfiguration" &&
        optimizer_type != "MinSR") {
      throw std::invalid_argument("Unknown optimizer type: " + optimizer_type);
    }
    if (optimizer_type == "MinSR") {
      if (minsr_r_pinv < 0.0) {
        throw std::invalid_argument("MinSRRPinv must be >= 0");
      }
      if (minsr_a_pinv < 0.0) {
        throw std::invalid_argument("MinSRAPinv must be >= 0");
      }
    }
  }

  qlpeps::OptimizerParams CreateOptimizerParams() const {
    qlpeps::OptimizerParams::BaseParams base_params(
        max_iterations,
        energy_tolerance,
        gradient_tolerance,
        plateau_patience,
        learning_rate,
        nullptr);

    if (optimizer_type == "SGD") {
      return qlpeps::OptimizerParams(
          base_params, qlpeps::SGDParams(momentum, nesterov, weight_decay));
    }
    if (optimizer_type == "Adam") {
      return qlpeps::OptimizerParams(
          base_params, qlpeps::AdamParams(beta1, beta2, epsilon, weight_decay));
    }
    if (optimizer_type == "AdaGrad") {
      return qlpeps::OptimizerParams(
          base_params, qlpeps::AdaGradParams(epsilon, initial_accumulator));
    }
    if (optimizer_type == "MinSR") {
      return qlpeps::OptimizerParams(
          base_params,
          qlpeps::MinSRParams(
              minsr_r_pinv, minsr_a_pinv, minsr_soft_cutoff, minsr_solver_mode));
    }

    qlpeps::ConjugateGradientParams cg_params{
        .max_iter = cg_max_iter,
        .relative_tolerance = cg_relative_tolerance,
        .residual_recompute_interval = cg_residual_recompute_interval,
    };
    qlpeps::StochasticReconfigurationParams sr_params{
        .cg_params = cg_params,
        .diag_shift = sr_diag_shift,
        .normalize_update = normalize_update,
    };
    return qlpeps::OptimizerParams(base_params, sr_params);
  }
};

}  // namespace peps_kondo_params

#endif  // LNO_PEPS_KONDO_OPTIMIZER_PARAMS_COMPAT_H
