// SPDX-License-Identifier: MIT
#ifndef LNO_IPEPS_BILAYER_TWO_ORBITAL_MODEL_H
#define LNO_IPEPS_BILAYER_TWO_ORBITAL_MODEL_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "qlpeps/api/config/parameter_reader.h"
#include "qlpeps/algorithm/local_update/imaginary_time_gate.h"
#include "qlpeps/two_dim_tn/ipeps/unit_cell_geometry.h"
#include "qlten/qlten.h"

namespace bilayer_ipeps {

#ifndef LNO_IPEPS_SPIN_U1
#define LNO_IPEPS_SPIN_U1 1
#endif

#if LNO_IPEPS_SPIN_U1
using QNT = qlten::special_qn::fU1U1QN;  // Unshifted (N, 2 Sz).
inline constexpr bool kConserveSpin = true;
inline constexpr const char *kSymmetryName = "fU1U1QN(N,2Sz)";
#else
using QNT = qlten::special_qn::fU1QN;  // Only unshifted N; parity follows N mod2.
inline constexpr bool kConserveSpin = false;
inline constexpr const char *kSymmetryName = "fU1QN(N)";
#endif
using Scalar = double;
using IndexT = qlten::Index<QNT>;
using Tensor = qlten::QLTensor<double, QNT>;
using Json = qlpeps::config::Json;
constexpr size_t kPhysicalDimension = 81;

inline int particle_charge(const QNT &qn) {
#if LNO_IPEPS_SPIN_U1
  return qn.GetQNVal(0).GetVal();
#else
  // fU1QN exposes serialization but no public value accessor.
  std::ostringstream encoded;
  qn.StreamWrite(encoded);
  std::istringstream input(encoded.str());
  int charge;
  if (!(input >> charge)) throw std::runtime_error("Cannot decode particle quantum number.");
  return charge;
#endif
}

struct PhysicsParams {
  double epsilon_z = 0, epsilon_x = 0;
  double v = 0, jh = 0;
  double tz_perp = 0, jz_perp = 0;
  double tx1 = 0, tx2 = 0;
  double txz1 = 0, txz2 = 0;
  double jx1 = 0, jx2 = 0;
  double jxz1 = 0, jxz2 = 0;
  double mu = 0;

  static PhysicsParams from_json(const Json &values) {
    if (!values.is_object()) throw std::invalid_argument("Physics must be a JSON object.");
    const std::set<std::string> keys{
        "EpsilonZ", "EpsilonX", "V", "JH", "TzPerp", "JzPerp",
        "Tx1", "Tx2", "Txz1", "Txz2", "Jx1", "Jx2", "Jxz1", "Jxz2", "Mu"};
    for (auto it = values.begin(); it != values.end(); ++it) {
      if (!keys.count(it.key())) {
        throw std::invalid_argument("Unknown Physics parameter: " + it.key());
      }
    }
    using qlpeps::config::ReadDouble;
    PhysicsParams p;
    p.epsilon_x = 0.558;
    p.v = 3; p.jh = 1;
    p.tx1 = 0.403; p.tx2 = 0.398;
    p.txz1 = 0.211; p.txz2 = 0.190;
    p.jx1 = 0.0650; p.jx2 = 0.0633;
    p.jxz1 = 0.0385; p.jxz2 = 0.0268;
    p.epsilon_z = ReadDouble(values, "EpsilonZ", p.epsilon_z);
    p.epsilon_x = ReadDouble(values, "EpsilonX", p.epsilon_x);
    p.v = ReadDouble(values, "V", p.v);
    p.jh = ReadDouble(values, "JH", p.jh);
    p.tz_perp = ReadDouble(values, "TzPerp", p.tz_perp);
    p.jz_perp = ReadDouble(values, "JzPerp", p.jz_perp);
    p.tx1 = ReadDouble(values, "Tx1", p.tx1);
    p.tx2 = ReadDouble(values, "Tx2", p.tx2);
    p.txz1 = ReadDouble(values, "Txz1", p.txz1);
    p.txz2 = ReadDouble(values, "Txz2", p.txz2);
    p.jx1 = ReadDouble(values, "Jx1", p.jx1);
    p.jx2 = ReadDouble(values, "Jx2", p.jx2);
    p.jxz1 = ReadDouble(values, "Jxz1", p.jxz1);
    p.jxz2 = ReadDouble(values, "Jxz2", p.jxz2);
    p.mu = ReadDouble(values, "Mu", p.mu);
    return p;
  }
};

inline void validate_physics(const PhysicsParams &p) {
  for (double value : {p.epsilon_z, p.epsilon_x, p.v, p.jh, p.tz_perp,
                      p.jz_perp, p.tx1, p.tx2, p.txz1, p.txz2, p.jx1,
                      p.jx2, p.jxz1, p.jxz2, p.mu}) {
    if (!std::isfinite(value)) throw std::invalid_argument("All Hamiltonian couplings must be finite.");
  }
}

struct BasisState {
  uint16_t bits;
  size_t canonical_label;
  int particles;
  int sz2;
};

// Ordered Fock modes: tz-up,tz-down,tx-up,tx-down,bz-up,bz-down,bx-up,bx-down.
// Canonical label is base 3 with orbital digits (hole,up,down)=(0,1,2),
// top-z the least significant digit. Every orbital is projected separately.
inline const std::array<BasisState, kPhysicalDimension> &basis() {
  static const auto basis = [] {
    std::array<BasisState, kPhysicalDimension> result{};
    for (size_t label = 0; label < result.size(); ++label) {
      size_t digits = label;
      BasisState state{0, label, 0, 0};
      for (size_t orbital = 0; orbital < 4; ++orbital) {
        const size_t occupation = digits % 3;
        digits /= 3;
        if (occupation) {
          state.bits |= uint16_t(1u << (2 * orbital + occupation - 1));
          ++state.particles;
          state.sz2 += occupation == 1 ? 1 : -1;
        }
      }
      result[label] = state;
    }
    std::sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
      return std::tie(a.particles, a.sz2, a.canonical_label) <
             std::tie(b.particles, b.sz2, b.canonical_label);
    });
    return result;
  }();
  return basis;
}

inline size_t basis_coordinate(uint16_t bits) {
  static const auto coordinates = [] {
    std::array<int, 256> result{};
    result.fill(-1);
    for (size_t i = 0; i < basis().size(); ++i) result[basis()[i].bits] = int(i);
    return result;
  }();
  if (bits >= coordinates.size() || coordinates[bits] < 0) {
    throw std::invalid_argument("Fock state violates the no-double-occupancy projection.");
  }
  return size_t(coordinates[bits]);
}

inline const IndexT &physical_index() {
  static const IndexT physical = [] {
    std::vector<qlten::QNSector<QNT>> sectors;
    for (size_t i = 0; i < basis().size();) {
      size_t end = i + 1;
      while (end < basis().size() && basis()[end].particles == basis()[i].particles &&
             (!kConserveSpin || basis()[end].sz2 == basis()[i].sz2)) ++end;
#if LNO_IPEPS_SPIN_U1
      sectors.emplace_back(QNT(basis()[i].particles, basis()[i].sz2), end - i);
#else
      sectors.emplace_back(QNT(basis()[i].particles), end - i);
#endif
      i = end;
    }
    return IndexT(sectors, qlten::IN);
  }();
  return physical;
}

namespace detail {

inline int popcount(uint32_t bits) {
  int count = 0;
  while (bits) { bits &= bits - 1; ++count; }
  return count;
}

inline bool annihilate(uint32_t &bits, size_t mode, double &coefficient) {
  const uint32_t mask = uint32_t(1) << mode;
  if (!(bits & mask)) return false;
  if (popcount(bits & (mask - 1)) % 2) coefficient = -coefficient;
  bits ^= mask;
  return true;
}

inline bool create(uint32_t &bits, size_t mode, double &coefficient) {
  const uint32_t mask = uint32_t(1) << mode;
  // Gutzwiller projection: the partner spin mode may not be occupied either.
  if (bits & (uint32_t(3) << (mode / 2 * 2))) return false;
  if (popcount(bits & (mask - 1)) % 2) coefficient = -coefficient;
  bits |= mask;
  return true;
}

inline int number(uint32_t bits, size_t orbital) {
  return int((bits >> (2 * orbital)) & 1) + int((bits >> (2 * orbital + 1)) & 1);
}

inline double spin_z(uint32_t bits, size_t orbital) {
  return 0.5 * (int((bits >> (2 * orbital)) & 1) -
                int((bits >> (2 * orbital + 1)) & 1));
}

template<class Emit>
void append_hopping(uint32_t bits, size_t orbital_a, size_t orbital_b, double t, Emit emit) {
  if (t == 0) return;
  for (size_t spin = 0; spin < 2; ++spin) {
    for (int direction = 0; direction < 2; ++direction) {
      const size_t from = 2 * (direction ? orbital_a : orbital_b) + spin;
      const size_t to = 2 * (direction ? orbital_b : orbital_a) + spin;
      uint32_t output = bits;
      double coefficient = -t;
      if (annihilate(output, from, coefficient) && create(output, to, coefficient)) {
        emit(output, coefficient);
      }
    }
  }
}

template<class Emit>
void append_exchange(uint32_t bits, size_t orbital_a, size_t orbital_b, double j, Emit emit) {
  if (j == 0) return;
  const double diagonal = j * spin_z(bits, orbital_a) * spin_z(bits, orbital_b);
  if (diagonal != 0) emit(bits, diagonal);
  for (size_t spin = 0; spin < 2; ++spin) {
    uint32_t output = bits;
    double coefficient = j / 2;
    // S_a^(+/-) S_b^(-/+), applying the rightmost operator first.
    if (annihilate(output, 2 * orbital_b + spin, coefficient) &&
        create(output, 2 * orbital_b + (1 - spin), coefficient) &&
        annihilate(output, 2 * orbital_a + (1 - spin), coefficient) &&
        create(output, 2 * orbital_a + spin, coefficient)) {
      emit(output, coefficient);
    }
  }
}

}  // namespace detail

struct Transition { size_t output; double value; };
struct BondTransition { size_t output_first, output_second; double value; };

inline std::vector<Transition> local_transitions(size_t input, const PhysicsParams &p) {
  validate_physics(p);
  if (input >= basis().size()) throw std::out_of_range("Local input coordinate out of range.");
  const uint32_t bits = basis()[input].bits;
  std::map<size_t, double> values;
  auto emit = [&](uint32_t output, double coefficient) {
    values[basis_coordinate(uint16_t(output))] += coefficient;
  };
  double diagonal = 0;
  for (size_t layer = 0; layer < 2; ++layer) {
    const size_t z = 2 * layer, x = z + 1;
    const int nz = detail::number(bits, z), nx = detail::number(bits, x);
    diagonal += p.epsilon_z * nz + p.epsilon_x * nx - p.mu * (nz + nx) +
                (p.v - p.jh / 2) * nz * nx;
    detail::append_exchange(bits, z, x, -2 * p.jh, emit);
  }
  emit(bits, diagonal);
  detail::append_hopping(bits, 0, 2, p.tz_perp, emit);
  detail::append_exchange(bits, 0, 2, p.jz_perp, emit);
  std::vector<Transition> result;
  for (const auto &[output, value] : values) if (value != 0) result.push_back({output, value});
  return result;
}

// Ordinary ordered-Fock matrix elements, before the tensor gate encoding chi.
// Both endpoint onsite terms occur with weight 1/4, once per each of four NN bonds.
inline std::vector<BondTransition> bond_transitions(
    size_t input_first, size_t input_second, const PhysicsParams &p, int bond_type) {
  validate_physics(p);
  if (input_first >= basis().size() || input_second >= basis().size()) {
    throw std::out_of_range("Bond input coordinate out of range.");
  }
  if (bond_type != 1 && bond_type != 2) throw std::invalid_argument("Bond type must be 1 or 2.");
  const uint32_t bits = basis()[input_first].bits | (uint32_t(basis()[input_second].bits) << 8);
  std::map<std::pair<size_t, size_t>, double> values;
  auto emit = [&](uint32_t output, double coefficient) {
    values[{basis_coordinate(uint16_t(output & 255)), basis_coordinate(uint16_t(output >> 8))}] += coefficient;
  };
  for (const auto &term : local_transitions(input_first, p)) {
    values[{term.output, input_second}] += term.value / 4;
  }
  for (const auto &term : local_transitions(input_second, p)) {
    values[{input_first, term.output}] += term.value / 4;
  }
  const double tx = bond_type == 1 ? p.tx1 : p.tx2;
  const double txz = bond_type == 1 ? p.txz1 : p.txz2;
  const double jx = bond_type == 1 ? p.jx1 : p.jx2;
  const double jxz = bond_type == 1 ? p.jxz1 : p.jxz2;
  for (size_t layer = 0; layer < 2; ++layer) {
    const size_t za = 2 * layer, xa = za + 1, zb = za + 4, xb = zb + 1;
    detail::append_hopping(bits, xa, xb, tx, emit);
    detail::append_hopping(bits, xa, zb, txz, emit);
    detail::append_hopping(bits, za, xb, txz, emit);
    detail::append_exchange(bits, xa, xb, jx, emit);
    detail::append_exchange(bits, xa, zb, jxz, emit);
    detail::append_exchange(bits, za, xb, jxz, emit);
  }
  std::vector<BondTransition> result;
  for (const auto &[output, value] : values) {
    if (value != 0) result.push_back({output.first, output.second, value});
  }
  return result;
}

using LocalMatrix = std::array<std::array<double, kPhysicalDimension>, kPhysicalDimension>;

inline LocalMatrix onsite_matrix(const PhysicsParams &p) {
  LocalMatrix result{};
  for (size_t input = 0; input < basis().size(); ++input) {
    for (const auto &term : local_transitions(input, p)) result[term.output][input] += term.value;
  }
  return result;
}

inline Tensor build_bond_hamiltonian(const IndexT &first, const IndexT &second,
                                   const PhysicsParams &p, int bond_type) {
  if (first != physical_index() || second != physical_index()) {
    throw std::invalid_argument("Bilayer Hamiltonian requires the model's exact IN physical index.");
  }
  Tensor result({qlten::InverseIndex(first), qlten::InverseIndex(second), first, second});
  for (size_t u = 0; u < basis().size(); ++u) {
    for (size_t v = 0; v < basis().size(); ++v) {
      const double chi = qlpeps::local_update_detail::PhysicalOperatorInputSign(first, u, second, v);
      for (const auto &term : bond_transitions(u, v, p, bond_type)) {
        result.SetElem({u, v, term.output_first, term.output_second}, chi * term.value);
      }
    }
  }
  return result;
}

inline int bond_type(size_t row, size_t col, qlpeps::BondOrientation orientation) {
  if (orientation != qlpeps::HORIZONTAL && orientation != qlpeps::VERTICAL) {
    throw std::invalid_argument("Invalid NN bond orientation.");
  }
  return ((orientation == qlpeps::HORIZONTAL) == ((row + col) % 2 == 0)) ? 1 : 2;
}

// This convenience builds one tensor per orbit. For production, cache the two
// Hamiltonians/gates instead; all bonds of the same type have identical couplings.
inline std::vector<qlpeps::NearestNeighborBondHamiltonian<double, QNT>> build_terms(
    size_t rows, size_t cols, const PhysicsParams &p) {
  if (rows % 2 || cols % 2) throw std::invalid_argument("Checkerboard cell dimensions must be even.");
  const auto one = build_bond_hamiltonian(physical_index(), physical_index(), p, 1);
  const auto two = build_bond_hamiltonian(physical_index(), physical_index(), p, 2);
  std::vector<qlpeps::NearestNeighborBondHamiltonian<double, QNT>> result;
  for (const auto &bond : qlpeps::UnitCellGeometry(rows, cols).Bonds()) {
    result.push_back({bond.first, bond.orientation,
                      bond_type(bond.first.row(), bond.first.col(), bond.orientation) == 1 ? one : two});
  }
  return result;
}

// Ordinary [output,input] local matrix; for one-site CTM density matrices use
// sum rho[input,output]*matrix[output,input]/trace(rho). No two-site chi applies.
inline Tensor build_one_site_operator(const LocalMatrix &matrix) {
  const auto &physical = physical_index();
  Tensor result({qlten::InverseIndex(physical), physical});
  for (size_t input = 0; input < basis().size(); ++input) {
    for (size_t output = 0; output < basis().size(); ++output) {
      if (matrix[output][input] != 0) result.SetElem({input, output}, matrix[output][input]);
    }
  }
  return result;
}

inline std::vector<double> diagonal_observable(const std::string &name) {
  static const std::map<std::string, std::vector<size_t>> orbitals{
      {"total", {0, 1, 2, 3}}, {"z", {0, 2}}, {"x", {1, 3}},
      {"top", {0, 1}}, {"bottom", {2, 3}}, {"top_z", {0}},
      {"top_x", {1}}, {"bottom_z", {2}}, {"bottom_x", {3}}};
  const bool spin = name.rfind("sz_", 0) == 0;
  const bool number = name.rfind("n_", 0) == 0;
  if (!spin && !number) throw std::invalid_argument("Unknown diagonal observable: " + name);
  const auto it = orbitals.find(name.substr(spin ? 3 : 2));
  if (it == orbitals.end()) throw std::invalid_argument("Unknown diagonal observable: " + name);
  std::vector<double> result(basis().size(), 0);
  for (size_t i = 0; i < basis().size(); ++i) {
    for (size_t orbital : it->second) {
      result[i] += spin ? detail::spin_z(basis()[i].bits, orbital) :
                          detail::number(basis()[i].bits, orbital);
    }
  }
  return result;
}

inline Tensor build_one_site_operator(const std::string &name) {
  const auto diagonal = diagonal_observable(name);
  LocalMatrix matrix{};
  for (size_t i = 0; i < diagonal.size(); ++i) matrix[i][i] = diagonal[i];
  return build_one_site_operator(matrix);
}

// Ordinary real Sx matrix. With N-only tensors, transverse spin coherences
// must be measured as well as Sz; real wavefunctions have <Sy>=0.
inline LocalMatrix transverse_spin_matrix(const std::string &name) {
  if (name.rfind("sx_", 0) != 0) throw std::invalid_argument("Expected an sx_ observable.");
  const auto selected_sz = diagonal_observable("sz_" + name.substr(3));
  LocalMatrix result{};
  for (size_t input = 0; input < basis().size(); ++input) {
    const auto bits = basis()[input].bits;
    for (size_t orbital = 0; orbital < 4; ++orbital) {
      if (detail::number(bits, orbital) == 0) continue;
      const auto flipped = uint16_t(bits ^ (3u << (2 * orbital)));
      const auto output = basis_coordinate(flipped);
      if (std::abs(selected_sz[output] - selected_sz[input]) > 0.5) result[output][input] += 0.5;
    }
  }
  return result;
}

// All seeds have three electrons/rung: one z electron on each layer, and one
// x electron alternating between layers. A common period-four charge seed is
// used on 4x4 comparison cells so every magnetic seed has total Sz=0 for AF
// layer alignment. A checkerboard charge seed would accidentally polarize all
// x spins in the Neel initializer and lock it into a different conserved sector.
// Each layer averages 1.5 electrons/Ni.
// The layer registry is a seed choice; it is degenerate if both interlayer terms vanish.
inline std::vector<std::vector<size_t>> build_seed_labels(
    size_t rows, size_t cols, const std::string &name, int layer_alignment = -1,
    const std::string &charge_pattern = "auto", size_t charge_offset = 0,
    size_t spin_offset = 0) {
  qlpeps::UnitCellGeometry geometry(rows, cols);
  if (rows % 2 || cols % 2) throw std::invalid_argument("Seed cell dimensions must be even.");
  if (layer_alignment != 1 && layer_alignment != -1) {
    throw std::invalid_argument("Layer alignment must be +1 or -1.");
  }
  if (spin_offset > 3) throw std::invalid_argument("SpinOffset must be 0..3.");
  if (charge_offset > 3 || (charge_pattern != "auto" &&
      charge_pattern != "stripe_x" && charge_pattern != "stripe_y")) {
    throw std::invalid_argument("ChargePattern must be auto, stripe_x or stripe_y; ChargeOffset must be 0..3.");
  }
  if ((charge_pattern == "stripe_x" && cols % 4) ||
      (charge_pattern == "stripe_y" && rows % 4)) {
    throw std::invalid_argument("Charge stripes require a period-four cell in their direction.");
  }
  const bool plus = name == "diagonal_plus";
  const bool minus = name == "diagonal_minus" || name == "stripe_pi2pi2";
  const bool xstripe = name == "stripe_x" || name == "stripe_pi0" || name == "pi0";
  const bool ystripe = name == "stripe_y";
  const bool xquarter = name == "stripe_pi2_x";
  const bool yquarter = name == "stripe_pi2_y";
  const bool neel = name == "neel";
  const bool ferro = name == "ferro" || name == "fm";
  if (!(plus || minus || xstripe || ystripe || xquarter || yquarter || neel || ferro)) {
    throw std::invalid_argument("Unknown initial spin pattern: " + name);
  }
  if ((plus || minus) && (rows % 4 || cols % 4)) {
    throw std::invalid_argument("Axis-aligned diagonal stripe cells require rows and cols divisible by 4.");
  }
  if ((xquarter && cols % 4) || (yquarter && rows % 4)) {
    throw std::invalid_argument("Quarter-axis spin stripes require period four in their direction.");
  }
  // For AF layer alignment, matching charge/spin period-four stripes in phase
  // polarize the x electrons. Auto chooses a perpendicular charge pattern.
  const std::string effective_charge = charge_pattern == "auto" && xquarter ?
      "stripe_y" : charge_pattern;
  if (effective_charge == "stripe_y" && rows % 4) {
    throw std::invalid_argument("This charge seed requires rows divisible by four; use a balanced explicit phase for a smaller cell.");
  }
  std::vector<std::vector<size_t>> result(rows, std::vector<size_t>(cols));
  int total_sz2 = 0;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      int top_spin = 1;
      if (plus || minus) {
        const auto phase = (static_cast<long>(row) + (plus ? 1 : -1) * static_cast<long>(col) +
                            static_cast<long>(spin_offset)) % 4;
        top_spin = ((phase + 4) % 4 < 2) ? 1 : -1;
      } else if (xstripe) top_spin = (col + spin_offset) % 2 ? -1 : 1;
      else if (ystripe) top_spin = (row + spin_offset) % 2 ? -1 : 1;
      else if (xquarter || yquarter) {
        const size_t phase = ((xquarter ? col : row) + spin_offset) % 4;
        top_spin = phase < 2 ? 1 : -1;
      }
      else if (neel) top_spin = (row + col + spin_offset) % 2 ? -1 : 1;
      uint16_t bits = 0;
      const size_t occupied_x_layer = effective_charge == "stripe_y" ?
          ((row + charge_offset) / 2) % 2 :
          ((effective_charge == "stripe_x" || cols % 4 == 0) ?
           ((col + charge_offset) / 2) % 2 :
           (neel ? (row + charge_offset) % 2 : (row + col + charge_offset) % 2));
      for (size_t layer = 0; layer < 2; ++layer) {
        const int spin = layer ? layer_alignment * top_spin : top_spin;
        const size_t offset = spin > 0 ? 0 : 1;
        bits |= uint16_t(1u << (4 * layer + offset));
        if (occupied_x_layer == layer) bits |= uint16_t(1u << (4 * layer + 2 + offset));
      }
      result[row][col] = basis_coordinate(bits);
      total_sz2 += basis()[result[row][col]].sz2;
    }
  }
  if (kConserveSpin && (xquarter || yquarter) && layer_alignment == -1 && total_sz2 != 0) {
    throw std::invalid_argument("Quarter-axis seed has nonzero total Sz: use perpendicular charge stripes or shift the relative spin/charge phase.");
  }
  return result;
}

}  // namespace bilayer_ipeps
#endif  // LNO_IPEPS_BILAYER_TWO_ORBITAL_MODEL_H
