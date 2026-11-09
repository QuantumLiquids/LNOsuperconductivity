// SPDX-License-Identifier: MIT
#include "single_layer_model.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace sl = single_layer_ipeps;
namespace bl = bilayer_ipeps;

constexpr size_t kSiteDim = 9;
constexpr size_t kPairDim = kSiteDim * kSiteDim;
using Column = std::map<uint32_t, double>;
using Matrix = std::array<std::array<double, kPairDim>, kPairDim>;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void compare(double actual, double expected, double &max_error, const char *message) {
  const double error = std::abs(actual - expected);
  require(std::isfinite(error) && error < 2e-12, message);
  max_error = std::max(max_error, error);
}

unsigned orbital_bits(uint32_t state, unsigned orbital) {
  return (state >> (2 * orbital)) & 3U;
}

// Independent ordered-Fock reference: a hop's Jordan–Wigner sign counts
// occupied modes strictly between its endpoints.
void hop(Column &column, uint32_t state, unsigned a, unsigned b, double hopping) {
  for (unsigned spin = 0; spin < 2; ++spin) {
    for (bool reverse : {false, true}) {
      const unsigned from = 2 * (reverse ? b : a) + spin;
      const unsigned to = 2 * (reverse ? a : b) + spin;
      const uint32_t from_bit = 1U << from;
      const uint32_t to_bit = 1U << to;
      if (!(state & from_bit) || orbital_bits(state, to / 2)) continue;
      const uint32_t between = std::max(from_bit, to_bit) - 2 * std::min(from_bit, to_bit);
      const double parity = (std::popcount(state & between) % 2) ? -1.0 : 1.0;
      column[state ^ from_bit ^ to_bit] -= parity * hopping;
    }
  }
}

// On the occupied spin-1/2 pair, S_a.S_b is +1/4 for parallel spins
// and has diagonal -1/4 and spin-flip amplitude 1/2 for opposite spins.
void exchange(Column &column, uint32_t state, unsigned a, unsigned b, double coupling) {
  const unsigned spin_a = orbital_bits(state, a), spin_b = orbital_bits(state, b);
  if (spin_a == 0 || spin_b == 0) return;
  const bool parallel = spin_a == spin_b;
  column[state] += (parallel ? 0.25 : -0.25) * coupling;
  if (!parallel) {
    column[state ^ (3U << (2 * a)) ^ (3U << (2 * b))] += 0.5 * coupling;
  }
}

void onsite(Column &column, uint32_t state, unsigned z, const sl::PhysicsParams &p,
            double weight) {
  const unsigned x = z + 1;
  const int nz = orbital_bits(state, z) != 0, nx = orbital_bits(state, x) != 0;
  column[state] += weight * ((p.epsilon_z - p.mu) * nz + (p.epsilon_x - p.mu) * nx +
                             (p.v - 0.5 * p.jh) * nz * nx);
  exchange(column, state, z, x, -2 * weight * p.jh);
}

Matrix reference_hamiltonian(const sl::PhysicsParams &p, int type,
                             qlpeps::BondOrientation orientation) {
  const double tx = type == 1 ? p.tx1 : p.tx2;
  const double txz = (type == 1 ? p.txz1 : p.txz2) *
      (orientation == qlpeps::VERTICAL ? p.txz_y_sign : 1);
  const double jx = type == 1 ? p.jx1 : p.jx2;
  const double jxz = type == 1 ? p.jxz1 : p.jxz2;
  Matrix h{};
  for (size_t u = 0; u < kSiteDim; ++u) {
    for (size_t v = 0; v < kSiteDim; ++v) {
      const uint32_t state = sl::basis()[u].bits | (uint32_t(sl::basis()[v].bits) << 4);
      Column column;
      onsite(column, state, 0, p, 0.25);
      onsite(column, state, 2, p, 0.25);
      hop(column, state, 1, 3, tx);
      hop(column, state, 1, 2, txz);
      hop(column, state, 0, 3, txz);
      exchange(column, state, 1, 3, jx);
      exchange(column, state, 1, 2, jxz);
      exchange(column, state, 0, 3, jxz);
      for (const auto &[output, coefficient] : column) {
        const size_t a = sl::basis_coordinate(output & 15U);
        const size_t b = sl::basis_coordinate(output >> 4);
        h[a * kSiteDim + b][u * kSiteDim + v] = coefficient;
      }
    }
  }
  return h;
}

Matrix reference_exponential(const Matrix &h, double tau) {
  Matrix result{}, term{};
  for (size_t i = 0; i < kPairDim; ++i) result[i][i] = term[i][i] = 1.0;
  for (unsigned order = 1; order <= 24; ++order) {
    Matrix next{};
    for (size_t row = 0; row < kPairDim; ++row) {
      for (size_t k = 0; k < kPairDim; ++k) {
        if (h[row][k] == 0) continue;
        const double factor = -tau * h[row][k] / order;
        for (size_t col = 0; col < kPairDim; ++col) {
          next[row][col] += factor * term[k][col];
        }
      }
    }
    for (size_t row = 0; row < kPairDim; ++row) {
      for (size_t col = 0; col < kPairDim; ++col) result[row][col] += next[row][col];
    }
    term = next;
  }
  return result;
}

// The qlpeps two-input encoding is (-1)^(p_u*p_v+p_u+p_v) for IN indices.
double input_sign(size_t u, size_t v) {
  return (sl::basis()[u].particles % 2 || sl::basis()[v].particles % 2) ? -1.0 : 1.0;
}

void check_basis_and_onsite(const sl::PhysicsParams &p) {
  const auto &basis = sl::basis();
  const auto &index = sl::physical_index();
  require(basis.size() == kSiteDim && index.dim() == kSiteDim &&
              index.GetDir() == qlten::IN && index.GetQNScts().size() == 3,
          "Expected nine physical states in three N-only sectors.");
  std::array<size_t, 3> multiplicities{};
  const auto h = sl::onsite_matrix(p);
  double max_error = 0;
  for (size_t u = 0; u < kSiteDim; ++u) {
    const auto bits = basis[u].bits;
    const int n = std::popcount(bits);
    require(bits < 16 && orbital_bits(bits, 0) != 3 && orbital_bits(bits, 1) != 3,
            "Basis state violates the orbital projection.");
    require(sl::basis_coordinate(bits) == u, "Basis is not a bijection.");
    require(basis[u].particles == n, "Basis particle charge mismatch.");
    const auto qn = index.GetQNSctFromActualCoor(u).GetQn();
    require(bl::particle_charge(qn) == n && qn.IsFermionParityOdd() == bool(n % 2),
            "Index charge or fermionic parity mismatch.");
    ++multiplicities[n];
    Column reference;
    onsite(reference, bits, 0, p, 1);
    for (size_t v = 0; v < kSiteDim; ++v) {
      compare(h[v][u], reference[basis[v].bits], max_error, "Independent onsite matrix.");
    }
  }
  require(multiplicities == std::array<size_t, 3>{1, 4, 4}, "Incorrect charge degeneracies.");
}

void check_gate(const sl::PhysicsParams &p, int type, qlpeps::BondOrientation orientation,
                const Matrix &reference, double &max_h_error, double &max_gate_error) {
  const auto &index = sl::physical_index();
  const auto h = sl::build_bond_hamiltonian(index, index, p, type, orientation);
  constexpr double tau = 0.13;
  const auto gate = qlpeps::BuildImaginaryTimeGate(h, tau);
  const auto exact_gate = reference_exponential(reference, tau);
  const double factor = std::exp(gate.log_amplitude_factor);
  for (size_t u = 0; u < kPairDim; ++u) {
    const size_t a = u / kSiteDim, b = u % kSiteDim;
    const double sign = input_sign(a, b);
    for (size_t v = 0; v < kPairDim; ++v) {
      const std::vector<size_t> coordinate{a, b, v / kSiteDim, v % kSiteDim};
      compare(sign * h.GetElem(coordinate), reference[v][u], max_h_error,
              "Independent ordered-Fock Hamiltonian or graded encoding.");
      compare(sign * factor * gate.gate.GetElem(coordinate), exact_gate[v][u],
              max_gate_error, "Independent exponential of the two-site Hamiltonian.");
    }
  }
}

uint32_t rung_bits(size_t top, size_t bottom) {
  return sl::basis()[top].bits | (uint32_t(sl::basis()[bottom].bits) << 4);
}

// Reorder the two-rung Fock modes from (a_top,a_bottom,b_top,b_bottom) to
// (a_top,b_top,a_bottom,b_bottom). Moving b_top past a_bottom gives this sign.
double layer_order_sign(size_t a_bottom, size_t b_top) {
  return (sl::basis()[a_bottom].particles * sl::basis()[b_top].particles % 2) ? -1.0 : 1.0;
}

void check_bilayer_factorization(const sl::PhysicsParams &p, int type,
                                 qlpeps::BondOrientation orientation, const Matrix &h,
                                 double &max_error) {
  for (size_t top = 0; top < kPairDim; ++top) {
    const size_t at = top / kSiteDim, bt = top % kSiteDim;
    for (size_t bottom = 0; bottom < kPairDim; ++bottom) {
      const size_t ab = bottom / kSiteDim, bb = bottom % kSiteDim;
      const double input = layer_order_sign(ab, bt);
      Column difference;
      for (size_t out = 0; out < kPairDim; ++out) {
        const size_t a = out / kSiteDim, b = out % kSiteDim;
        if (h[out][top] != 0) {
          const uint32_t bits = rung_bits(a, ab) | (rung_bits(b, bb) << 8);
          difference[bits] += input * layer_order_sign(ab, b) * h[out][top];
        }
        if (h[out][bottom] != 0) {
          const uint32_t bits = rung_bits(at, a) | (rung_bits(bt, b) << 8);
          difference[bits] += input * layer_order_sign(a, bt) * h[out][bottom];
        }
      }
      for (const auto &entry : bl::bond_transitions(bl::basis_coordinate(rung_bits(at, ab)),
               bl::basis_coordinate(rung_bits(bt, bb)), p, type, orientation)) {
        const uint32_t bits = bl::basis()[entry.output_first].bits |
            (uint32_t(bl::basis()[entry.output_second].bits) << 8);
        difference[bits] -= entry.value;
      }
      for (const auto &[bits, error] : difference) {
        compare(error, 0, max_error, "Two-rung Hamiltonian fails fermionic layer factorization.");
        (void) bits;
      }
    }
  }
}

void check_rejects_interlayer_terms(const sl::PhysicsParams &physics) {
  for (auto term : {&sl::PhysicsParams::tz_perp, &sl::PhysicsParams::jz_perp}) {
    auto p = physics;
    p.*term = 1e-5;
    bool rejected = false;
    try {
      sl::validate_physics(p);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "Single-layer input silently discarded nonzero interlayer coupling.");
  }
}

void check_axis_exchange(const sl::PhysicsParams &p) {
  const qlpeps::UnitCellGeometry cell(4, 4);
  const auto rotate = [](const qlpeps::SiteIdx &site) {
    return qlpeps::SiteIdx{(site.col() + 1) % 4, site.row()};
  };
  for (const auto &bond : cell.Bonds()) {
    const auto orientation = bond.orientation == qlpeps::HORIZONTAL ?
        qlpeps::VERTICAL : qlpeps::HORIZONTAL;
    const qlpeps::UnitCellBond image{rotate(bond.first), orientation};
    require(cell.Second(image).site == rotate(cell.Second(bond).site) &&
                sl::bond_type(bond.first.row(), bond.first.col(), bond.orientation) ==
                    sl::bond_type(image.first.row(), image.first.col(), image.orientation),
            "Axis exchange with checkerboard offset must preserve the directed bond types.");
  }
  const auto orbital_sign = [](size_t pair) {
    const unsigned a = orbital_bits(sl::basis()[pair / kSiteDim].bits, 0);
    const unsigned b = orbital_bits(sl::basis()[pair % kSiteDim].bits, 0);
    return (bool(a) != bool(b)) ? -1.0 : 1.0;
  };
  double max_error = 0;
  for (int type : {1, 2}) {
    const auto horizontal = reference_hamiltonian(p, type, qlpeps::HORIZONTAL);
    const auto vertical = reference_hamiltonian(p, type, qlpeps::VERTICAL);
    for (size_t u = 0; u < kPairDim; ++u) {
      for (size_t v = 0; v < kPairDim; ++v) {
        compare(horizontal[v][u], orbital_sign(v) * orbital_sign(u) * vertical[v][u],
                max_error, "Orbital gauge fails to restore the rotated hopping signs.");
      }
    }
  }
}

}  // namespace

int main() {
  try {
    qlten::hp_numeric::SetCpuNumericsThreads(1);
    sl::PhysicsParams p;
    p.epsilon_z = -0.13; p.epsilon_x = 0.428; p.mu = 0.31;
    p.v = 3.0; p.jh = 1.0;
    p.tx1 = 0.403; p.tx2 = 0.398;
    p.txz1 = 0.211; p.txz2 = -0.190;
    p.jx1 = 0.065; p.jx2 = 0.0633;
    p.jxz1 = 0.0385; p.jxz2 = 0.0268;
    check_basis_and_onsite(p);
    check_rejects_interlayer_terms(p);
    p.txz_y_sign = -1;
    check_axis_exchange(p);
    double h_error = 0, gate_error = 0, layer_error = 0;
    for (int y_sign : {-1, 1}) {
      p.txz_y_sign = y_sign;
      for (int type : {1, 2}) {
        for (auto orientation : {qlpeps::HORIZONTAL, qlpeps::VERTICAL}) {
          const auto h = reference_hamiltonian(p, type, orientation);
          check_gate(p, type, orientation, h, h_error, gate_error);
          check_bilayer_factorization(p, type, orientation, h, layer_error);
        }
      }
    }
    std::cout << "PASS: nine-state N-only basis, projected onsite and H/V bonds, "
                 "graded imaginary-time gates, fermionic layer factorization, and axial symmetry.\n"
              << "Maximum Fock H, exp(-tau H), and bilayer factorization errors: "
              << h_error << ", " << gate_error << ", " << layer_error << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Single-layer model validation: " << error.what() << '\n';
    return 1;
  }
}
