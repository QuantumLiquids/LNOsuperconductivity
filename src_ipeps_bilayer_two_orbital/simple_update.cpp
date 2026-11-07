// SPDX-License-Identifier: MIT
// Infinite, periodically repeated bilayer-rung PEPS: cached NN simple update
// followed by physical CTMRG measurements. SU normalization is never an energy.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qlpeps/algorithm/ctmrg/ctmrg.h"
#include "qlpeps/algorithm/expectation_value/ctm_observables.h"
#include "qlpeps/algorithm/local_update/imaginary_time_gate.h"
#include "qlpeps/algorithm/simple_update/simple_update_projection.h"
#include "qlpeps/two_dim_tn/ipeps/infinite_simple_update_state.h"
#include "model.h"
#include "gate_builder.h"

#ifndef LNO_IPEPS_ENERGY_COMPONENTS
#define LNO_IPEPS_ENERGY_COMPONENTS 0
#endif
#if LNO_IPEPS_ENERGY_COMPONENTS
#include "energy_components.h"
#endif

namespace {
namespace fs = std::filesystem;
using namespace bilayer_ipeps;
using State = qlpeps::InfiniteSimpleUpdateState<double, QNT>;
using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

Json read_json(const fs::path &path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open JSON file: " + path.string());
  Json result;
  input >> result;
  if (result.contains("CaseParams")) result = result.at("CaseParams");
  if (!result.is_object()) throw std::invalid_argument("Expected a JSON object");
  return result;
}

void write_json(const fs::path &path, const Json &value) {
  std::ofstream output;
  output.exceptions(std::ios::failbit | std::ios::badbit);
  output.open(path);
  output << std::setw(2) << value << '\n';
}

size_t positive_size(const Json &j, const char *key, size_t fallback) {
  const auto value = qlpeps::config::ReadSize(j, key, fallback);
  if (value == 0) throw std::invalid_argument(std::string(key) + " must be positive");
  return value;
}

double finite_number(const Json &j, const char *key, double fallback) {
  return qlpeps::config::ReadDouble(j, key, fallback);
}

int read_layer_alignment(const Json &j) {
  if (!j.contains("LayerAlignment")) return -1;
  const auto &value = j.at("LayerAlignment");
  if (!value.is_number_integer() || (value != -1 && value != 1)) {
    throw std::invalid_argument("LayerAlignment must be integer -1 or +1");
  }
  return value.get<int>();
}

struct Stage { size_t steps; double tau; };

struct Algorithm {
  size_t rows, cols, measure_every, threads, charge_offset, spin_offset;
  std::string seed, charge_pattern;
  fs::path gate_cache_directory;
  bool measure_initial, measure_stage_ends;
  size_t checkpoint_every;
  int alignment;
  qlpeps::SimpleUpdateTruncateParams trunc;
  qlpeps::CTMRGParams ctm;
  std::vector<Stage> schedule;

  explicit Algorithm(const Json &j, bool measurement_only)
      : rows(positive_size(j, "Rows", 4)), cols(positive_size(j, "Cols", 4)),
        measure_every(positive_size(j, "MeasureEvery", 5)),
        threads(positive_size(j, "ThreadNum", 1)),
        charge_offset(qlpeps::config::ReadSize(j, "ChargeOffset", 0)),
        spin_offset(qlpeps::config::ReadSize(j, "SpinOffset", 0)),
        seed(qlpeps::config::ReadString(j, "InitState", std::string("diagonal_plus"))),
        charge_pattern(qlpeps::config::ReadString(j, "ChargePattern", std::string("auto"))),
        gate_cache_directory(qlpeps::config::ReadString(j, "GateCacheDir", std::string())),
        measure_initial(qlpeps::config::ReadBool(j, "MeasureInitial", true)),
        measure_stage_ends(qlpeps::config::ReadBool(j, "MeasureStageEnds", true)),
        checkpoint_every(qlpeps::config::ReadSize(j, "CheckpointEvery", 0)),
        alignment(read_layer_alignment(j)),
        trunc(positive_size(j, "Dmin", 1), positive_size(j, "Dmax", 2),
              finite_number(j, "TruncErr", 1e-12),
              finite_number(j, "InverseCutoff", 1e-12)) {
    static const std::vector<std::string> allowed{
        "Rows", "Cols", "MeasureEvery", "ThreadNum", "InitState", "LayerAlignment",
        "ChargePattern", "ChargeOffset", "SpinOffset", "GateCacheDir",
        "MeasureInitial", "MeasureStageEnds", "CheckpointEvery",
        "Dmin", "Dmax", "TruncErr", "InverseCutoff", "Chi", "MaxCTMSweeps",
        "MinCTMSweeps", "CTMTolerance", "CTMTruncErr", "CTMSVDCutoff", "Schedule"};
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end()) {
        throw std::invalid_argument("Unknown algorithm key: " + it.key());
      }
    }
    (void)qlpeps::UnitCellGeometry(rows, cols);
    if (rows * cols > static_cast<size_t>(std::numeric_limits<int>::max() / 4)) {
      throw std::invalid_argument("Unit cell exceeds the integer quantum-number range");
    }
    // Both checkerboard anisotropy and the quarter-wave diagonal stripe must
    // close exactly at the cell seams. The model seed helper checks periods.
    if ((rows % 2) || (cols % 2)) {
      throw std::invalid_argument("Checkerboard zigzag unit-cell Rows and Cols must be even");
    }
    if (alignment != -1 && alignment != 1) {
      throw std::invalid_argument("LayerAlignment must be -1 or +1");
    }
    if (charge_offset > 3 || (charge_pattern != "auto" &&
        charge_pattern != "stripe_x" && charge_pattern != "stripe_y")) {
      throw std::invalid_argument("ChargePattern must be auto, stripe_x or stripe_y; ChargeOffset must be 0..3");
    }
    if (spin_offset > 3) throw std::invalid_argument("SpinOffset must be 0..3");
    if (threads > static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("ThreadNum exceeds the numeric backend range");
    }
    qlpeps::ValidateSimpleUpdateTruncateParams(trunc);
    ctm.max_bond_dim = positive_size(j, "Chi", 8);
    ctm.max_sweeps = positive_size(j, "MaxCTMSweeps", 100);
    ctm.min_sweeps = positive_size(j, "MinCTMSweeps", 2);
    ctm.spectral_tolerance = ctm.observable_tolerance = finite_number(j, "CTMTolerance", 1e-8);
    ctm.truncation_error = finite_number(j, "CTMTruncErr", 1e-12);
    ctm.svd_relative_cutoff = finite_number(j, "CTMSVDCutoff", 1e-12);
    qlpeps::ValidateCTMRGParams(ctm);
    if (j.contains("Schedule")) {
      const auto &stages = j.at("Schedule");
      if (!stages.is_array()) throw std::invalid_argument("Schedule must be an array");
      size_t total_steps = 0;
      for (const auto &stage : stages) {
        if (!stage.is_object() || stage.size() != 2 || !stage.contains("Steps") || !stage.contains("Tau")) {
          throw std::invalid_argument("Each Schedule entry must contain exactly Steps and Tau");
        }
        const auto steps = positive_size(stage, "Steps", 0);
        const auto tau = finite_number(stage, "Tau", -1);
        if (tau < 0 || steps > std::numeric_limits<size_t>::max() - total_steps) {
          throw std::invalid_argument("Invalid schedule time step or overflowing step count");
        }
        total_steps += steps;
        schedule.push_back({steps, tau});
      }
    }
    if (!measurement_only && schedule.empty()) {
      throw std::invalid_argument("Evolution requires a nonempty Schedule");
    }
  }
};

// A scan reuses the same physical gates for many magnetic/charge starts and D.
// The exact physics JSON and executable build identify a cache; a changed
// Hamiltonian or rebuilt executable must use a new directory. Runs sharing one
// directory are sequential. Each gate is committed atomically after writing.
std::array<Tensor, kGateSlots> prepare_gates(
    const PhysicsParams &physics, const Json &physics_json, double tau,
    const fs::path &directory, std::optional<EvolutionGateBuilder> &builder) {
  std::array<fs::path, kGateSlots> paths;
  if (!directory.empty()) {
    const Json identity{{"format", 2}, {"physics", physics_json},
        {"gate_slots", {"H1", "H2", "V1", "V2"}},
        {"txz_orientation_signs", {{"horizontal", 1}, {"vertical", physics.txz_y_sign}}},
        {"build", std::string(__DATE__) + " " + __TIME__},
        {"basis", std::string("projected_rung81_IN_v1_") + kSymmetryName}};
    const auto metadata = directory / "metadata.json";
    if (fs::exists(directory)) {
      if (!fs::exists(metadata) || read_json(metadata) != identity) {
        throw std::invalid_argument("Gate cache does not match Hamiltonian/build: " + directory.string());
      }
    } else {
      fs::create_directories(directory);
      write_json(metadata, identity);
    }
    std::ostringstream name;
    name << "tau_" << std::setprecision(std::numeric_limits<double>::max_digits10) << tau;
    for (size_t i = 0; i < kGateSlots; ++i) {
      const std::string direction = gate_orientation(i) == qlpeps::HORIZONTAL ? "H" : "V";
      paths[i] = directory / (name.str() + "_" + direction + std::to_string(gate_type(i)) + ".qlten");
    }
    if (std::all_of(paths.begin(), paths.end(), [](const auto &path) { return fs::exists(path); })) {
      std::array<Tensor, kGateSlots> result;
      const std::vector<IndexT> indexes{qlten::InverseIndex(physical_index()),
          qlten::InverseIndex(physical_index()), physical_index(), physical_index()};
      for (size_t i = 0; i < kGateSlots; ++i) {
        std::ifstream input(paths[i], std::ios::binary);
        input.exceptions(std::ios::failbit | std::ios::badbit);
        input >> result[i];
        if (result[i].GetIndexes() != indexes || !(result[i].GetQuasi2Norm() > 0) ||
            !std::isfinite(result[i].GetQuasi2Norm())) {
          throw std::runtime_error("Invalid cached gate: " + paths[i].string());
        }
      }
      std::cout << "loaded gates from " << directory << '\n';
      return result;
    }
  }
  if (!builder) builder.emplace(physics);
  auto result = builder->build(tau);
  if (!directory.empty()) for (size_t i = 0; i < kGateSlots; ++i) {
    const fs::path temporary = paths[i].string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << result[i];
    output.close();
    fs::rename(temporary, paths[i]);
  }
  return result;
}

// Checkerboard type and spatial orientation independently select a Hamiltonian.
// In particular, a vertical cross-orbital hopping may have the opposite sign.
int bond_type(const qlpeps::UnitCellBond &bond) {
  return bilayer_ipeps::bond_type(bond.first.row(), bond.first.col(), bond.orientation);
}

const std::array<std::string, 18> kObservables{
    "n_total", "n_z", "n_x", "n_top", "n_bottom", "n_top_z", "n_top_x",
    "n_bottom_z", "n_bottom_x", "sz_total", "sz_z", "sz_x", "sz_top",
    "sz_bottom", "sz_top_z", "sz_top_x", "sz_bottom_z", "sz_bottom_x"};
const std::array<std::string, 9> kTransverseObservables{
    "sx_total", "sx_z", "sx_x", "sx_top", "sx_bottom", "sx_top_z", "sx_top_x", "sx_bottom_z", "sx_bottom_x"};

Json symmetry_sector(const State &state) {
  long long particles = 0, sz2 = 0;
  for (size_t row = 0; row < state.rows(); ++row) for (size_t col = 0; col < state.cols(); ++col) {
    const auto divergence = state.GetPEPS().Gamma({row, col}).Div();
    // Physical legs point IN; virtual charges cancel under cell translations.
    particles -= particle_charge(divergence);
#if LNO_IPEPS_SPIN_U1
    sz2 -= divergence.GetQNVal(1).GetVal();
#endif
  }
  return {{"particles_per_cell", particles}, {"twice_sz_per_cell", kConserveSpin ? Json(sz2) : Json(nullptr)},
          {"symmetry", kSymmetryName}};
}

Json compute_fourier(const std::vector<double> &values, size_t rows, size_t cols) {
  constexpr double pi = 3.1415926535897932384626433832795;
  // All ten inequivalent q/-q pairs of a real 4x4 texture. Enlarged cells
  // may require additional momenta beyond this common screening registry.
  const std::array<std::array<double, 2>, 10> momenta{{
      {{pi / 2, pi / 2}}, {{pi / 2, -pi / 2}}, {{pi, 0}}, {{0, pi}}, {{pi, pi}}, {{0, 0}},
      {{pi / 2, 0}}, {{0, pi / 2}}, {{pi / 2, pi}}, {{pi, pi / 2}}}};
  const std::array<const char *, 10> names{
      "pi2_pi2", "pi2_minus_pi2", "pi_0", "0_pi", "pi_pi", "0_0",
      "pi2_0", "0_pi2", "pi2_pi", "pi_pi2"};
  Json output = Json::object();
  for (size_t q = 0; q < momenta.size(); ++q) {
    // q_x refers to col, q_y to row in the square-lattice coordinate system.
    const bool commensurate = std::abs(std::remainder(momenta[q][0] * cols, 2 * pi)) < 1e-10 &&
        std::abs(std::remainder(momenta[q][1] * rows, 2 * pi)) < 1e-10;
    if (!commensurate) continue;
    std::complex<double> amplitude(0, 0);
    for (size_t row = 0; row < rows; ++row) for (size_t col = 0; col < cols; ++col) {
      const double phase = momenta[q][0] * col + momenta[q][1] * row;
      amplitude += values[row * cols + col] * std::exp(std::complex<double>(0, phase));
    }
    amplitude /= static_cast<double>(rows * cols);
    output[names[q]] = {{"real", amplitude.real()}, {"imag", amplitude.imag()},
                        {"abs", std::abs(amplitude)}, {"abs_squared", std::norm(amplitude)}};
  }
  return output;
}

double checked_ratio(double numerator, double normalization, const std::string &context) {
  if (!std::isfinite(numerator) || !std::isfinite(normalization) || normalization == 0) {
    throw std::runtime_error(context + ": nonfinite insertion or zero normalization");
  }
  const double result = numerator / normalization;
  if (!std::isfinite(result)) throw std::runtime_error(context + ": nonfinite expectation");
  return result;
}

Json measure_state(const State &state, const PhysicsParams &physics,
             const Algorithm &algorithm) {
  const auto start = Clock::now();
  const auto ordinary = state.ToInfinitePEPS();
  const auto ctm = qlpeps::ConvergeCTM(ordinary, algorithm.ctm);
  if (!ctm.diagnostics.converged) {
    throw std::runtime_error("CTMRG did not converge after " +
        std::to_string(ctm.diagnostics.iterations) + " sweeps; checkpoint retained for --measure");
  }
  const qlpeps::InfiniteNormNetwork<double, QNT> network(ordinary);
  qlpeps::CTMObservableEnvironment<double, QNT> environment(ctm.environment);
  std::array<std::vector<double>, kObservables.size()> values;
  std::array<std::vector<double>, kObservables.size()> diagonals;
  std::array<std::vector<double>, kTransverseObservables.size()> transverse_values;
  struct LocalEntry { size_t input, output; double value; };
  std::array<std::vector<LocalEntry>, kTransverseObservables.size()> transverse_entries;
  for (size_t o = 0; o < kObservables.size(); ++o) diagonals[o] = diagonal_observable(kObservables[o]);
  for (size_t o = 0; o < kTransverseObservables.size(); ++o) {
    const auto matrix = transverse_spin_matrix(kTransverseObservables[o]);
    for (size_t input = 0; input < 81; ++input) for (size_t output = 0; output < 81; ++output) {
      if (matrix[output][input] != 0) transverse_entries[o].push_back({input, output, matrix[output][input]});
    }
  }
  Json sites = Json::array();
  for (size_t row = 0; row < state.rows(); ++row) for (size_t col = 0; col < state.cols(); ++col) {
    const auto rho = environment.OneSiteDensityMatrix(network, {row, col});
    double normalization = 0;
    for (size_t i = 0; i < 81; ++i) normalization += rho.GetElem({i, i});
    Json site{{"row", row}, {"col", col}};
    for (size_t o = 0; o < kObservables.size(); ++o) {
      double numerator = 0;
      for (size_t i = 0; i < 81; ++i) numerator += rho.GetElem({i, i}) * diagonals[o][i];
      const double value = checked_ratio(numerator, normalization, kObservables[o]);
      site[kObservables[o]] = value;
      values[o].push_back(value);
    }
    for (size_t o = 0; o < kTransverseObservables.size(); ++o) {
      double numerator = 0;
      for (const auto &entry : transverse_entries[o]) numerator += rho.GetElem({entry.input, entry.output}) * entry.value;
      const double value = checked_ratio(numerator, normalization, kTransverseObservables[o]);
      site[kTransverseObservables[o]] = value;
      transverse_values[o].push_back(value);
    }
    sites.push_back(std::move(site));
  }
  struct MatrixEntry { size_t a, b, out_a, out_b; double value; };
  std::array<std::vector<MatrixEntry>, kGateSlots> sparse_hamiltonians;
#if LNO_IPEPS_ENERGY_COMPONENTS
  const auto component_physics = energy_component_physics(physics);
  std::array<std::array<std::vector<MatrixEntry>, kEnergyComponentNames.size()>, kGateSlots> sparse_components;
  std::array<double, kEnergyComponentNames.size()> component_energies{};
#endif
  for (size_t slot = 0; slot < kGateSlots; ++slot) {
    const int type = gate_type(slot);
    const auto orientation = gate_orientation(slot);
    auto &entries = sparse_hamiltonians[slot];
    for (size_t a = 0; a < 81; ++a) for (size_t b = 0; b < 81; ++b) {
      for (const auto &transition : bond_transitions(a, b, physics, type, orientation)) {
        entries.push_back({a, b, transition.output_first, transition.output_second, transition.value});
      }
#if LNO_IPEPS_ENERGY_COMPONENTS
      for (size_t c = 0; c < component_physics.size(); ++c) {
        for (const auto &transition : bond_transitions(a, b, component_physics[c], type, orientation)) {
          sparse_components[slot][c].push_back(
              {a, b, transition.output_first, transition.output_second, transition.value});
        }
      }
#endif
    }
  }
  double energy = 0;
  Json bonds = Json::array();
  for (const auto &bond : state.GetGeometry().Bonds()) {
    const auto rho = environment.BondDensityMatrix(network, bond);
    double numerator = 0, normalization = 0;
    const int type = bond_type(bond);
    const size_t slot = gate_slot(type, bond.orientation);
    for (size_t a = 0; a < 81; ++a) for (size_t b = 0; b < 81; ++b) {
      normalization += rho.GetElem({a, b, a, b});
    }
    for (const auto &entry : sparse_hamiltonians[slot]) {
      numerator += rho.GetElem({entry.a, entry.b, entry.out_a, entry.out_b}) * entry.value;
    }
    const double value = checked_ratio(numerator, normalization, "bond energy");
    energy += value;
#if LNO_IPEPS_ENERGY_COMPONENTS
    for (size_t c = 0; c < component_energies.size(); ++c) {
      double component_numerator = 0;
      for (const auto &entry : sparse_components[slot][c]) {
        component_numerator += rho.GetElem({entry.a, entry.b, entry.out_a, entry.out_b}) * entry.value;
      }
      component_energies[c] += checked_ratio(
          component_numerator, normalization, kEnergyComponentNames[c]);
    }
#endif
    bonds.push_back({{"row", bond.first.row()}, {"col", bond.first.col()},
        {"direction", bond.orientation == qlpeps::HORIZONTAL ? "east" : "south"},
        {"bond_type", type}, {"txz_orientation_sign", bond.orientation == qlpeps::HORIZONTAL ? 1 : physics.txz_y_sign},
        {"grand_energy", value}});
  }
  const auto count = static_cast<double>(state.rows() * state.cols());
  Json means = Json::object(), fourier = Json::object();
  for (size_t o = 0; o < kObservables.size(); ++o) {
    double sum = 0;
    for (const double value : values[o]) sum += value;
    means[kObservables[o]] = sum / count;
    fourier[kObservables[o]] = compute_fourier(values[o], state.rows(), state.cols());
  }
  for (size_t o = 0; o < kTransverseObservables.size(); ++o) {
    double sum = 0;
    for (double value : transverse_values[o]) sum += value;
    means[kTransverseObservables[o]] = sum / count;
    fourier[kTransverseObservables[o]] = compute_fourier(transverse_values[o], state.rows(), state.cols());
  }
  const double density = means.at("n_total").get<double>();
  const double physical_energy = energy / count + physics.mu * density;
  const auto dimensions = state.GetPEPS().GetMinMaxBondDimension();
  Json result{{"grand_energy_per_rung", energy / count},
      {"physical_energy_per_rung", physical_energy},
      {"physical_energy_per_ni", physical_energy / 2},
      {"means", means}, {"exact_symmetry_sector", symmetry_sector(state)},
      {"measured_spin_components", {"x", "z"}}, {"real_ansatz_sy_expectation", 0.0},
      {"density_deviation_from_three_per_rung", density - 3},
      {"sites", sites}, {"bonds", bonds}, {"fourier", fourier},
      {"fourier_definition", "abs(sum_r <O_r> exp(i q.r)/N); qx=col, qy=row; not a structure factor"},
      {"bond_dimension_min", dimensions.first}, {"bond_dimension_max", dimensions.second},
      {"ctm_converged", true}, {"ctm_sweeps", ctm.diagnostics.iterations},
      {"chi", algorithm.ctm.max_bond_dim}, {"measurement_seconds", seconds_since(start)}};
  result["txz_orientation_signs"] = {{"horizontal", 1}, {"vertical", physics.txz_y_sign}};
#if LNO_IPEPS_ENERGY_COMPONENTS
  Json components = Json::object();
  double component_sum = 0;
  for (size_t c = 0; c < component_energies.size(); ++c) {
    components[kEnergyComponentNames[c]] = component_energies[c] / (2 * count);
    component_sum += component_energies[c];
  }
  result["energy_components_per_ni"] = std::move(components);
  result["energy_components_definition"] =
      "Coupling-weighted pieces of the same bond-density-matrix grand-energy estimator; "
      "Hund includes its density term, onsite endpoint weight is 1/4. Physical sum adds Mu*n/Ni.";
  result["energy_component_closure"] = {
      {"grand_sum_per_ni", component_sum / (2 * count)},
      {"grand_residual_per_ni", (component_sum - energy) / (2 * count)},
      {"chemical_potential_correction_per_ni", physics.mu * density / 2},
      {"physical_sum_per_ni", (component_sum / count + physics.mu * density) / 2},
      {"physical_residual_per_ni", (component_sum / count + physics.mu * density - physical_energy) / 2}};
#endif
  return result;
}

std::string step_name(size_t step) {
  std::ostringstream output;
  output << "checkpoint_step" << std::setw(8) << std::setfill('0') << step;
  return output.str();
}

void require_geometry(const State &state, const Algorithm &algorithm) {
  if (state.rows() != algorithm.rows || state.cols() != algorithm.cols) {
    throw std::invalid_argument("Checkpoint geometry does not match Rows/Cols");
  }
  for (size_t row = 0; row < state.rows(); ++row) for (size_t col = 0; col < state.cols(); ++col) {
    if (state.GetPEPS().Gamma({row, col}).GetIndex(4) != physical_index()) {
      throw std::invalid_argument("Checkpoint physical index does not match the projected rung basis");
    }
  }
  const auto sector = symmetry_sector(state);
  const auto expected_particles = 3LL * static_cast<long long>(state.rows()) * state.cols();
  if (sector.at("particles_per_cell").get<long long>() != expected_particles) {
    throw std::invalid_argument("Checkpoint particle sector differs from three electrons per rung");
  }
}
}  // namespace

int main(int argc, char **argv) {
  fs::path run_directory;
  bool created_run_directory = false;
  try {
    const bool measurement_only = argc > 1 && std::string(argv[1]) == "--measure";
#if LNO_IPEPS_ENERGY_COMPONENTS
    if (!measurement_only) throw std::invalid_argument("Energy-component executable requires --measure.");
#endif
    if ((!measurement_only && argc != 4 && argc != 5) || (measurement_only && argc != 6)) {
      std::cerr << "Usage: " << argv[0] << " physics.json algorithm.json new_run_directory [restart_checkpoint]\n"
                << "       " << argv[0] << " --measure physics.json algorithm.json checkpoint new_result.json\n";
      return 1;
    }
    const int offset = measurement_only ? 1 : 0;
    const auto physics_json = read_json(argv[1 + offset]);
    const auto algorithm_json = read_json(argv[2 + offset]);
    const auto physics = PhysicsParams::from_json(physics_json);
    validate_physics(physics);
    const Algorithm algorithm(algorithm_json, measurement_only);
    qlten::hp_numeric::SetCpuNumericsThreads(static_cast<int>(algorithm.threads));
    std::cout << std::setprecision(12) << std::unitbuf;
    if (measurement_only) {
      const fs::path output = argv[5];
      if (fs::exists(output)) throw std::runtime_error("Refusing to overwrite result: " + output.string());
      const auto state = State::LoadCheckpoint(argv[4]);
      require_geometry(state, algorithm);
      const auto result = measure_state(state, physics, algorithm);
      write_json(output, {{"physics", physics_json}, {"algorithm", algorithm_json},
          {"checkpoint", fs::absolute(argv[4]).string()}, {"measurement", result}});
      std::cout << "CTMRG physical energy/rung=" << result.at("physical_energy_per_rung")
                << " n/rung=" << result.at("means").at("n_total") << '\n';
      return 0;
    }
    run_directory = argv[3];
    if (fs::exists(run_directory)) throw std::runtime_error("Refusing to overwrite run directory: " + run_directory.string());
    fs::create_directories(run_directory);
    created_run_directory = true;
    write_json(run_directory / "physics.json", physics_json);
    write_json(run_directory / "algorithm.json", algorithm_json);
    auto state = argc == 5 ? State::LoadCheckpoint(argv[4]) : State::ProductState(
        physical_index(), algorithm.rows, algorithm.cols,
        build_seed_labels(algorithm.rows, algorithm.cols, algorithm.seed, algorithm.alignment,
                          algorithm.charge_pattern, algorithm.charge_offset, algorithm.spin_offset));
    require_geometry(state, algorithm);
    Json summary{{"status", "running"}, {"physics", physics_json}, {"algorithm", algorithm_json},
        {"measurements", Json::array()}, {"sweeps", Json::array()}};
    if (argc == 5) summary["restart_checkpoint"] = fs::absolute(argv[4]).string();
    const auto run_start = Clock::now();
    size_t step = 0;
    auto measure = [&](size_t stage, double tau) {
      const auto checkpoint = run_directory / step_name(step);
      state.SaveCheckpoint(checkpoint);
      write_json(run_directory / "latest_checkpoint.json", {{"step", step}, {"checkpoint", fs::absolute(checkpoint).string()}});
      auto record = measure_state(state, physics, algorithm);
      record["step"] = step;
      record["stage"] = stage;
      record["tau"] = tau;
      record["checkpoint"] = fs::absolute(checkpoint).string();
      write_json(run_directory / (step_name(step) + ".json"), record);
      summary["measurements"].push_back(record);
      summary["elapsed_seconds"] = seconds_since(run_start);
      write_json(run_directory / "summary.json", summary);
      std::cout << "measurement step=" << step << " physical_energy/rung="
                << record.at("physical_energy_per_rung") << " n/rung="
                << record.at("means").at("n_total") << " D="
                << record.at("bond_dimension_max") << " CTM_sweeps="
                << record.at("ctm_sweeps") << " seconds=" << record.at("measurement_seconds") << '\n';
    };
    std::cout << algorithm.rows << 'x' << algorithm.cols << " infinite bilayer-rung cell, d=81, Dmax="
              << algorithm.trunc.D_max << ", symmetry=" << kSymmetryName << ", seed=" << algorithm.seed << '\n';
    if (algorithm.measure_initial) measure(0, 0);
    else {
      state.SaveCheckpoint(run_directory / step_name(0));
      write_json(run_directory / "latest_checkpoint.json", {{"step", 0},
          {"checkpoint", fs::absolute(run_directory / step_name(0)).string()}});
      write_json(run_directory / "summary.json", summary);
    }
    std::optional<EvolutionGateBuilder> builder;
    for (size_t stage = 0; stage < algorithm.schedule.size(); ++stage) {
      const auto &schedule = algorithm.schedule[stage];
      const auto gate_start = Clock::now();
      std::cout << "preparing four orientation/type gates: stage=" << stage << " tau=" << schedule.tau << '\n';
      const auto gates = prepare_gates(physics, physics_json, schedule.tau,
                                       algorithm.gate_cache_directory, builder);
      std::cout << "gate preparation seconds=" << seconds_since(gate_start) << '\n';
      for (size_t within_stage = 0; within_stage < schedule.steps; ++within_stage) {
        const auto sweep_start = Clock::now();
        auto candidate = state.GetPEPS();
        double max_discarded_weight = 0;
        for (const auto &bond : state.GetGeometry().Bonds()) {
          const auto &gate = gates[gate_slot(bond_type(bond), bond.orientation)];
          const auto projection = qlpeps::NearestNeighborSiteProject(
              candidate, gate, bond.first, bond.orientation, algorithm.trunc);
          if (!(projection.norm > 0) || !std::isfinite(projection.norm) ||
              !std::isfinite(projection.trunc_err) || projection.trunc_err < 0 || !projection.bond_dim) {
            throw std::runtime_error("Invalid simple-update projection diagnostics");
          }
          max_discarded_weight = std::max(max_discarded_weight, projection.trunc_err);
        }
        state.ReplacePEPS(std::move(candidate));
        ++step;
        const double sweep_seconds = seconds_since(sweep_start);
        summary["sweeps"].push_back({{"step", step}, {"stage", stage}, {"tau", schedule.tau},
            {"max_discarded_weight", max_discarded_weight}, {"seconds", sweep_seconds}});
        std::cout << "SU step=" << step << " tau=" << schedule.tau << " max_discarded_weight="
                  << max_discarded_weight << " seconds=" << sweep_seconds << '\n';
        const bool last_stage_step = within_stage + 1 == schedule.steps;
        const bool final_step = last_stage_step && stage + 1 == algorithm.schedule.size();
        if (step % algorithm.measure_every == 0 || final_step ||
            (algorithm.measure_stage_ends && last_stage_step)) {
          measure(stage, schedule.tau);
        } else if (algorithm.checkpoint_every != 0 && step % algorithm.checkpoint_every == 0) {
          const auto checkpoint = run_directory / step_name(step);
          state.SaveCheckpoint(checkpoint);
          write_json(run_directory / "latest_checkpoint.json", {{"step", step},
              {"checkpoint", fs::absolute(checkpoint).string()}});
          summary["elapsed_seconds"] = seconds_since(run_start);
          write_json(run_directory / "summary.json", summary);
        }
      }
    }
    summary["status"] = "schedule_completed";
    summary["executed_steps"] = step;
    summary["elapsed_seconds"] = seconds_since(run_start);
    summary["interpretation"] = "Simple-update screening with finite-chi CTMRG; a completed schedule does not establish ground-state or order convergence";
    write_json(run_directory / "summary.json", summary);
    std::cout << "Completed " << step << " sweeps; results: " << (run_directory / "summary.json") << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Bilayer iPEPS: " << error.what() << '\n';
    if (created_run_directory) {
      try { write_json(run_directory / "failure.json", {{"status", "failed"}, {"error", error.what()}}); }
      catch (...) {}  // Preserve the original numerical or input error.
    }
    return 1;
  }
}
