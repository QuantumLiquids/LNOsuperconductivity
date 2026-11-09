// SPDX-License-Identifier: MIT
#ifndef LNO_IPEPS_SINGLE_LAYER_TWO_ORBITAL_SEED_H
#define LNO_IPEPS_SINGLE_LAYER_TWO_ORBITAL_SEED_H

#include "single_layer_model.h"
#include "uniform_seed.h"

namespace single_layer_ipeps {
namespace detail {

// Prepare one x electron delocalized over a dimer. Each endpoint has a
// fixed z spin, and the occupied x spin aligns with it in either component.
inline Tensor build_dimer_gate() {
  const auto &physical = physical_index();
  const auto dual = qlten::InverseIndex(physical);
  Tensor gate({dual, dual, physical, physical});
  const double amplitude = 1.0 / std::sqrt(2.0);
  for (size_t first_spin = 0; first_spin < 2; ++first_spin) {
    for (size_t second_spin = 0; second_spin < 2; ++second_spin) {
      const uint16_t first = (1u << first_spin) | (1u << (2 + first_spin));
      const uint16_t second = 1u << second_spin;
      const size_t u = basis_coordinate(first), v = basis_coordinate(second);
      uint32_t output = first | (uint32_t(second) << 4);
      double sign = 1.0;
      if (!bilayer_ipeps::detail::annihilate(output, 2 + first_spin, sign) ||
          !bilayer_ipeps::detail::create(output, 6 + second_spin, sign)) {
        throw std::logic_error("Invalid charge transfer in the single-layer seed.");
      }
      const double input_sign =
          qlpeps::local_update_detail::PhysicalOperatorInputSign(physical, u, physical, v);
      gate.SetElem({u, v, u, v}, input_sign * amplitude);
      gate.SetElem({u, v, basis_coordinate(uint16_t(output & 0xf)),
                    basis_coordinate(uint16_t(output >> 4))},
                   input_sign * sign * amplitude);
    }
  }
  return gate;
}

}  // namespace detail

inline qlpeps::InfiniteSimpleUpdateState<double, QNT> build_seed_state(
    size_t rows, size_t cols, const std::string &seed, const std::string &charge_pattern,
    size_t charge_offset, size_t spin_offset) {
  using State = qlpeps::InfiniteSimpleUpdateState<double, QNT>;
  const bool uniform = bilayer_ipeps::is_uniform_charge_seed(charge_pattern);
  if (uniform && charge_offset > 1) {
    throw std::invalid_argument("Uniform charge dimer ChargeOffset must be 0 or 1.");
  }
  const bool horizontal = charge_pattern == "uniform_x" || charge_pattern == "uniform_type_x";
  const bool checkerboard = bilayer_ipeps::is_checkerboard_charge_seed(charge_pattern);
  const auto is_dimer_start = [&](size_t row, size_t col) {
    const size_t coordinate = checkerboard ? row + col : (horizontal ? col : row);
    return (coordinate + charge_offset) % 2 == 0;
  };
  // Use the original magnetic labels. For a uniform seed the charge bits
  // are replaced, so the temporary pattern need only fit the cell.
  const std::string temporary_charge = cols % 4 == 0 ? "stripe_x" : "auto";
  auto labels = bilayer_ipeps::build_seed_labels(
      rows, cols, seed, -1, uniform ? temporary_charge : charge_pattern,
      uniform ? 0 : charge_offset, spin_offset);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      uint16_t bits = bilayer_ipeps::basis()[labels[row][col]].bits & 0xf;
      if (uniform) {
        bits &= 0x3;
        if (is_dimer_start(row, col)) bits |= uint16_t(bits << 2);
      }
      labels[row][col] = basis_coordinate(bits);
    }
  }
  auto state = State::ProductState(physical_index(), rows, cols, labels);
  if (!uniform) return state;

  const auto orientation = horizontal ? qlpeps::HORIZONTAL : qlpeps::VERTICAL;
  const auto gate = detail::build_dimer_gate();
  const qlpeps::SimpleUpdateTruncateParams exact_projection(2, 2, 0.0, 1e-14);
  auto peps = state.GetPEPS();
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      if (!is_dimer_start(row, col)) continue;
      const auto result = qlpeps::NearestNeighborSiteProject(
          peps, gate, {row, col}, orientation, exact_projection);
      if (result.bond_dim != 2 || !(result.norm > 0) || !std::isfinite(result.norm) ||
          !std::isfinite(result.trunc_err) || std::abs(result.trunc_err) > 1e-12) {
        throw std::runtime_error("Single-layer uniform charge dimer preparation was not exact.");
      }
    }
  }
  return State(std::move(peps));
}

}  // namespace single_layer_ipeps

#endif  // LNO_IPEPS_SINGLE_LAYER_TWO_ORBITAL_SEED_H
