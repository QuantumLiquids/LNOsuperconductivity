// SPDX-License-Identifier: MIT
#ifndef LNO_IPEPS_SINGLE_LAYER_TWO_ORBITAL_MODEL_H
#define LNO_IPEPS_SINGLE_LAYER_TWO_ORBITAL_MODEL_H

#include "model.h"

namespace single_layer_ipeps {

static_assert(!bilayer_ipeps::kConserveSpin, "The single-layer iPEPS uses N-only tensors.");

using bilayer_ipeps::BasisState;
using bilayer_ipeps::BondTransition;
using bilayer_ipeps::IndexT;
using bilayer_ipeps::Json;
using bilayer_ipeps::PhysicsParams;
using bilayer_ipeps::QNT;
using bilayer_ipeps::Tensor;
using bilayer_ipeps::Transition;
using bilayer_ipeps::bond_type;
using bilayer_ipeps::gate_orientation;
using bilayer_ipeps::gate_slot;
using bilayer_ipeps::gate_type;
using bilayer_ipeps::kGateSlots;
using bilayer_ipeps::kSymmetryName;
using bilayer_ipeps::particle_charge;

inline constexpr size_t kPhysicalDimension = 9;
using LocalMatrix = std::array<std::array<double, kPhysicalDimension>, kPhysicalDimension>;

inline void validate_physics(const PhysicsParams &physics) {
  bilayer_ipeps::validate_physics(physics);
  if (physics.tz_perp != 0 || physics.jz_perp != 0) {
    throw std::invalid_argument("Single-layer iPEPS requires TzPerp = JzPerp = 0.");
  }
}

// The bottom Ni is a vacuum spectator. The remaining ordered Fock modes
// are (z up, z down, x up, x down), with both orbitals projected.
inline const std::array<BasisState, kPhysicalDimension> &basis() {
  static const auto states = [] {
    std::array<BasisState, kPhysicalDimension> result{};
    size_t index = 0;
    for (const auto &state : bilayer_ipeps::basis()) {
      if ((state.bits >> 4) == 0) result.at(index++) = state;
    }
    return result;
  }();
  return states;
}

inline size_t basis_coordinate(uint16_t bits) {
  static const auto coordinates = [] {
    std::array<int, 16> result{};
    result.fill(-1);
    for (size_t i = 0; i < basis().size(); ++i) result[basis()[i].bits] = int(i);
    return result;
  }();
  if (bits >= coordinates.size() || coordinates[bits] < 0) {
    throw std::invalid_argument("Fock state violates the projected single-layer basis.");
  }
  return size_t(coordinates[bits]);
}

inline const IndexT &physical_index() {
  static const IndexT index = [] {
    std::vector<qlten::QNSector<QNT>> sectors;
    for (size_t begin = 0; begin < basis().size();) {
      size_t end = begin + 1;
      while (end < basis().size() && basis()[end].particles == basis()[begin].particles) ++end;
      sectors.emplace_back(QNT(basis()[begin].particles), end - begin);
      begin = end;
    }
    return IndexT(sectors, qlten::IN);
  }();
  return index;
}

inline std::vector<Transition> local_transitions(size_t input, const PhysicsParams &physics) {
  single_layer_ipeps::validate_physics(physics);
  auto terms = bilayer_ipeps::local_transitions(
      bilayer_ipeps::basis_coordinate(basis().at(input).bits), physics);
  for (auto &term : terms) term.output = basis_coordinate(bilayer_ipeps::basis()[term.output].bits);
  return terms;
}

inline std::vector<BondTransition> bond_transitions(
    size_t first, size_t second, const PhysicsParams &physics, int type,
    qlpeps::BondOrientation orientation = qlpeps::HORIZONTAL) {
  single_layer_ipeps::validate_physics(physics);
  // Removing the two empty spectators leaves the occupied-mode order
  // unchanged, so all ordered-Fock matrix elements keep their signs.
  auto terms = bilayer_ipeps::bond_transitions(
      bilayer_ipeps::basis_coordinate(basis().at(first).bits),
      bilayer_ipeps::basis_coordinate(basis().at(second).bits), physics, type, orientation);
  for (auto &term : terms) {
    term.output_first = basis_coordinate(bilayer_ipeps::basis()[term.output_first].bits);
    term.output_second = basis_coordinate(bilayer_ipeps::basis()[term.output_second].bits);
  }
  return terms;
}

inline LocalMatrix onsite_matrix(const PhysicsParams &physics) {
  LocalMatrix matrix{};
  for (size_t input = 0; input < basis().size(); ++input) {
    for (const auto &term : single_layer_ipeps::local_transitions(input, physics)) {
      matrix[term.output][input] = term.value;
    }
  }
  return matrix;
}

inline Tensor build_bond_hamiltonian(
    const IndexT &first, const IndexT &second, const PhysicsParams &physics, int type,
    qlpeps::BondOrientation orientation = qlpeps::HORIZONTAL) {
  if (first != physical_index() || second != physical_index()) {
    throw std::invalid_argument("Single-layer Hamiltonian requires the model's exact IN physical index.");
  }
  Tensor tensor({qlten::InverseIndex(first), qlten::InverseIndex(second), first, second});
  for (size_t u = 0; u < basis().size(); ++u) {
    for (size_t v = 0; v < basis().size(); ++v) {
      const double sign = qlpeps::local_update_detail::PhysicalOperatorInputSign(first, u, second, v);
      for (const auto &term : single_layer_ipeps::bond_transitions(u, v, physics, type, orientation)) {
        tensor.SetElem({u, v, term.output_first, term.output_second}, sign * term.value);
      }
    }
  }
  return tensor;
}

namespace detail {

inline std::pair<size_t, size_t> orbital_range(const std::string &suffix) {
  if (suffix == "total") return {0, 2};
  if (suffix == "z") return {0, 1};
  if (suffix == "x") return {1, 2};
  throw std::invalid_argument("Unknown single-layer observable orbital: " + suffix);
}

}  // namespace detail

inline std::vector<double> diagonal_observable(const std::string &name) {
  const bool spin = name.rfind("sz_", 0) == 0;
  const bool number = name.rfind("n_", 0) == 0;
  if (!spin && !number) throw std::invalid_argument("Unknown diagonal observable: " + name);
  const auto [begin, end] = detail::orbital_range(name.substr(spin ? 3 : 2));
  std::vector<double> values(basis().size(), 0);
  for (size_t i = 0; i < basis().size(); ++i) {
    const auto bits = basis()[i].bits;
    for (size_t orbital = begin; orbital < end; ++orbital) {
      values[i] += spin ? bilayer_ipeps::detail::spin_z(bits, orbital) :
                          bilayer_ipeps::detail::number(bits, orbital);
    }
  }
  return values;
}

inline LocalMatrix transverse_spin_matrix(const std::string &name) {
  if (name.rfind("sx_", 0) != 0) throw std::invalid_argument("Expected an sx_ observable.");
  const auto [begin, end] = detail::orbital_range(name.substr(3));
  LocalMatrix matrix{};
  for (size_t input = 0; input < basis().size(); ++input) {
    const auto bits = basis()[input].bits;
    for (size_t orbital = begin; orbital < end; ++orbital) {
      if (bilayer_ipeps::detail::number(bits, orbital) == 0) continue;
      const auto flipped = uint16_t(bits ^ (3u << (2 * orbital)));
      matrix[basis_coordinate(flipped)][input] = 0.5;
    }
  }
  return matrix;
}

}  // namespace single_layer_ipeps

#endif  // LNO_IPEPS_SINGLE_LAYER_TWO_ORBITAL_MODEL_H
