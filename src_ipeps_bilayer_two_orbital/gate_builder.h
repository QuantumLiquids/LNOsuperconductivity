// SPDX-License-Identifier: MIT
#ifndef LNO_IPEPS_BILAYER_TWO_ORBITAL_GATE_BUILDER_H
#define LNO_IPEPS_BILAYER_TWO_ORBITAL_GATE_BUILDER_H

#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "model.h"

namespace bilayer_ipeps {

// The Hamiltonian conserves Sz even when the variational tensors conserve only
// N. Exploit that exact operator property during exponentiation, then forget
// the auxiliary spin labels when encoding the gate in the ansatz's index.
// This does not restrict the physical or virtual spaces of the N-only PEPS.
class EvolutionGateBuilder {
 public:
  explicit EvolutionGateBuilder(const PhysicsParams &physics) : physics_(physics) {
    validate_physics(physics_);
  }

  std::array<Tensor, kGateSlots> build(double tau) {
    if (!std::isfinite(tau) || tau < 0) {
      throw std::invalid_argument("Imaginary-time step must be finite and nonnegative.");
    }
    if (tau == 0) {
      // Identity has an exact graded encoding; no eigensolve is necessary.
      auto identity = identity_gate();
      return {identity, identity, identity, std::move(identity)};
    }
    if (!hamiltonians_) {
      hamiltonians_.emplace();
      for (size_t slot = 0; slot < kGateSlots; ++slot) {
        // Legacy vertical Hamiltonians are identical to the horizontal ones.
        if (slot >= 2 && physics_.txz_y_sign == 1) continue;
        (*hamiltonians_)[slot] = build_spin_hamiltonian(gate_type(slot), gate_orientation(slot));
      }
    }
    std::array<Tensor, kGateSlots> result;
    for (size_t slot = 0; slot < result.size(); ++slot) {
      if (slot >= 2 && physics_.txz_y_sign == 1) {
        result[slot] = result[slot - 2];
        continue;
      }
      auto evolution = qlpeps::BuildImaginaryTimeGate((*hamiltonians_)[slot], tau);
#if !defined(LNO_IPEPS_SPIN_U1) || LNO_IPEPS_SPIN_U1
      result[slot] = std::move(evolution.gate);
#else
      result[slot] = forget_spin_label(evolution.gate);
#endif
    }
    return result;
  }

 private:
  using SpinQNT = qlten::special_qn::fU1U1QN;
  using SpinIndex = qlten::Index<SpinQNT>;
  using SpinTensor = qlten::QLTensor<double, SpinQNT>;
  using Pair = std::pair<size_t, size_t>;
  using Charge = std::pair<int, int>;

  static const SpinIndex &spin_physical_index() {
    static const SpinIndex physical = [] {
      std::vector<qlten::QNSector<SpinQNT>> sectors;
      for (size_t i = 0; i < basis().size();) {
        size_t end = i + 1;
        while (end < basis().size() &&
               basis()[end].particles == basis()[i].particles &&
               basis()[end].sz2 == basis()[i].sz2) {
          ++end;
        }
        sectors.emplace_back(SpinQNT(basis()[i].particles, basis()[i].sz2), end - i);
        i = end;
      }
      return SpinIndex(sectors, qlten::IN);
    }();
    return physical;
  }

  SpinTensor build_spin_hamiltonian(int bond_type, qlpeps::BondOrientation orientation) const {
    const auto &physical = spin_physical_index();
    SpinTensor result({qlten::InverseIndex(physical), qlten::InverseIndex(physical),
                       physical, physical});
    for (size_t u = 0; u < basis().size(); ++u) {
      for (size_t v = 0; v < basis().size(); ++v) {
        const double chi = qlpeps::local_update_detail::PhysicalOperatorInputSign(
            physical, u, physical, v);
        for (const auto &term : bond_transitions(u, v, physics_, bond_type, orientation)) {
          result.SetElem({u, v, term.output_first, term.output_second}, chi * term.value);
        }
      }
    }
    return result;
  }

  static Tensor identity_gate() {
    const auto &physical = physical_index();
    Tensor result({qlten::InverseIndex(physical), qlten::InverseIndex(physical),
                   physical, physical});
    for (size_t u = 0; u < basis().size(); ++u) {
      for (size_t v = 0; v < basis().size(); ++v) {
        result.SetElem({u, v, u, v},
            qlpeps::local_update_detail::PhysicalOperatorInputSign(physical, u, physical, v));
      }
    }
    return result;
  }

#if defined(LNO_IPEPS_SPIN_U1) && !LNO_IPEPS_SPIN_U1
  static const std::map<Charge, std::vector<Pair>> &spin_pair_groups() {
    static const auto groups = [] {
      std::map<Charge, std::vector<Pair>> result;
      for (size_t u = 0; u < basis().size(); ++u) {
        for (size_t v = 0; v < basis().size(); ++v) {
          result[{basis()[u].particles + basis()[v].particles,
                  basis()[u].sz2 + basis()[v].sz2}].push_back({u, v});
        }
      }
      return result;
    }();
    return groups;
  }

  static Tensor forget_spin_label(const SpinTensor &source) {
    const auto &physical = physical_index();
    Tensor result({qlten::InverseIndex(physical), qlten::InverseIndex(physical),
                   physical, physical});
    // Both physical layouts use the same (N,Sz,canonical) coordinate order.
    // Their parity and IN arrows agree, so the raw graded coefficients agree
    // without any additional sign or permutation. Spin-forbidden entries stay
    // exactly zero inside the larger N-only dense blocks.
    for (const auto &[charge, pairs] : spin_pair_groups()) {
      (void) charge;
      for (const auto &[u, v] : pairs) {
        for (const auto &[out_u, out_v] : pairs) {
          const double value = source.GetElem({u, v, out_u, out_v});
          if (!std::isfinite(value)) {
            throw std::runtime_error("Nonfinite spin-resolved gate coefficient.");
          }
          if (value != 0) result.SetElem({u, v, out_u, out_v}, value);
        }
      }
    }
    return result;
  }
#endif

  PhysicsParams physics_;
  std::optional<std::array<SpinTensor, kGateSlots>> hamiltonians_;
};

}  // namespace bilayer_ipeps
#endif  // LNO_IPEPS_BILAYER_TWO_ORBITAL_GATE_BUILDER_H
