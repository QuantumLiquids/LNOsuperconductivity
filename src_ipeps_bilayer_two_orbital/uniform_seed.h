// SPDX-License-Identifier: MIT
#ifndef LNO_IPEPS_BILAYER_TWO_ORBITAL_UNIFORM_SEED_H
#define LNO_IPEPS_BILAYER_TWO_ORBITAL_UNIFORM_SEED_H

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "qlpeps/algorithm/simple_update/simple_update_projection.h"
#include "qlpeps/two_dim_tn/ipeps/infinite_simple_update_state.h"
#include "model.h"

namespace bilayer_ipeps {
namespace uniform_seed_detail {

// The input has one x electron at the first end and none at the second end
// in the addressed layer. Delocalize it with equal amplitudes, with the x spin
// aligned to the local z spin on either end. The other layer is a spectator.
inline Tensor build_dimer_gate(size_t layer) {
  const auto &physical = physical_index();
  const auto dual = qlten::InverseIndex(physical);
  Tensor gate({dual, dual, physical, physical});
  const double amplitude = 1.0 / std::sqrt(2.0);
  const size_t z = 2 * layer;
  const size_t x = z + 1;
  for (size_t u = 0; u < basis().size(); ++u) {
    const uint32_t first = basis()[u].bits;
    if (detail::number(first, z) != 1 ||
        detail::number(first, 2 * (1 - layer)) != 1) {
      continue;
    }
    const size_t from = 2 * x + (detail::spin_z(first, z) < 0);
    if ((first & (uint32_t(1) << from)) == 0) continue;
    for (size_t v = 0; v < basis().size(); ++v) {
      const uint32_t second = basis()[v].bits;
      if (detail::number(second, z) != 1 ||
          detail::number(second, 2 * (1 - layer)) != 1 ||
          detail::number(second, x) != 0) {
        continue;
      }
      uint32_t output = first | (second << 8);
      const size_t to = 8 + 2 * x + (detail::spin_z(second, z) < 0);
      double sign = 1.0;
      if (!detail::annihilate(output, from, sign) ||
          !detail::create(output, to, sign)) {
        throw std::logic_error("Invalid charge transfer in uniform-density seed");
      }
      const double input_sign = qlpeps::local_update_detail::PhysicalOperatorInputSign(
          physical, u, physical, v);
      gate.SetElem({u, v, u, v}, input_sign * amplitude);
      gate.SetElem({u, v, basis_coordinate(uint16_t(output & 255)),
                    basis_coordinate(uint16_t(output >> 8))},
                   input_sign * sign * amplitude);
    }
  }
  return gate;
}

}  // namespace uniform_seed_detail

inline qlpeps::InfiniteSimpleUpdateState<double, QNT> build_uniform_charge_state(
    size_t rows, size_t cols, const std::string &spin_seed, int layer_alignment,
    const std::string &charge_pattern, size_t charge_offset, size_t spin_offset) {
  if (kConserveSpin) {
    throw std::invalid_argument("Uniform charge dimers require the N-only build");
  }
  if (charge_pattern != "uniform_x" && charge_pattern != "uniform_y") {
    throw std::invalid_argument("Expected uniform_x or uniform_y charge dimers");
  }
  if (charge_offset > 1) {
    throw std::invalid_argument("Uniform charge dimer ChargeOffset must be 0 or 1");
  }
  const bool horizontal = charge_pattern == "uniform_x";
  const auto orientation = horizontal ? qlpeps::HORIZONTAL : qlpeps::VERTICAL;
  // Reuse the magnetic pattern only. Use an explicit temporary x stripe
  // where it fits, so an x-quarter spin seed also works on a two-row cell.
  auto labels = build_seed_labels(rows, cols, spin_seed, layer_alignment,
                                  cols % 4 == 0 ? "stripe_x" : "auto", 0, spin_offset);
  constexpr uint16_t z_mask = 0x33;  // Both spins of top-z and bottom-z.
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      uint16_t bits = basis()[labels[row][col]].bits & z_mask;
      const size_t coordinate = horizontal ? col : row;
      const size_t layer = (coordinate + charge_offset) % 2;
      const size_t spin = detail::spin_z(bits, 2 * layer) < 0;
      bits |= uint16_t(1u << (4 * layer + 2 + spin));
      labels[row][col] = basis_coordinate(bits);
    }
  }

  // Offset the two layer coverings by one bond so every virtual edge carries
  // at most one dimer. Each dimer has exactly one x electron and Schmidt rank 2.
  using State = qlpeps::InfiniteSimpleUpdateState<double, QNT>;
  auto peps = State::ProductState(physical_index(), rows, cols, labels).GetPEPS();
  const qlpeps::SimpleUpdateTruncateParams exact_projection(2, 2, 0.0, 1e-14);
  for (size_t layer = 0; layer < 2; ++layer) {
    const auto gate = uniform_seed_detail::build_dimer_gate(layer);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t col = 0; col < cols; ++col) {
        const size_t coordinate = horizontal ? col : row;
        if ((coordinate + charge_offset) % 2 != layer) continue;
        const auto result = qlpeps::NearestNeighborSiteProject(
            peps, gate, {row, col}, orientation, exact_projection);
        if (result.bond_dim != 2 || !(result.norm > 0) || !std::isfinite(result.norm) ||
            !std::isfinite(result.trunc_err) || std::abs(result.trunc_err) > 1e-12) {
          throw std::runtime_error("Uniform charge dimer preparation was not exact");
        }
      }
    }
  }
  return State(std::move(peps));
}

}  // namespace bilayer_ipeps
#endif  // LNO_IPEPS_BILAYER_TWO_ORBITAL_UNIFORM_SEED_H
