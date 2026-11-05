// SPDX-License-Identifier: LGPL-3.0-only
// Independent checks of the projected bilayer Hamiltonian and its graded encoding.
#include "model.h"
#include "gate_builder.h"
#include "qlpeps/two_dim_tn/ipeps/ipeps_checkpoint.h"

#include <bit>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {
using namespace bilayer_ipeps;
using MatrixColumn = std::map<uint32_t, double>;

void require(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}

void require_near(double actual, double expected, const std::string &message) {
  if (std::abs(actual - expected) > 2e-12) {
    std::ostringstream detail;
    detail << message << ": actual=" << actual << ", expected=" << expected;
    throw std::runtime_error(detail.str());
  }
}

// Reference construction uses a 16-mode ordered Fock space, independently of
// the model's local coordinate ordering and transition visitors.
bool annihilate(uint32_t &bits, unsigned mode, double &coefficient) {
  const uint32_t flag = uint32_t(1) << mode;
  if ((bits & flag) == 0) return false;
  if (std::popcount(bits & (flag - 1)) % 2) coefficient = -coefficient;
  bits ^= flag;
  return true;
}

bool create(uint32_t &bits, unsigned mode, double &coefficient) {
  const uint32_t orbital = uint32_t(3) << (2 * (mode / 2));
  if (bits & orbital) return false;  // Projection is applied to each c operator.
  const uint32_t flag = uint32_t(1) << mode;
  if (std::popcount(bits & (flag - 1)) % 2) coefficient = -coefficient;
  bits |= flag;
  return true;
}

void transfer(MatrixColumn &column, uint32_t input, unsigned to,
              unsigned from, double amplitude) {
  double coefficient = amplitude;
  if (annihilate(input, from, coefficient) && create(input, to, coefficient)) {
    column[input] += coefficient;
  }
}

int occupation(uint32_t bits, unsigned orbital) {
  return std::popcount((bits >> (2 * orbital)) & 3U);
}

double reference_spin_z(uint32_t bits, unsigned orbital) {
  return 0.5 * (int((bits >> (2 * orbital)) & 1U) -
                int((bits >> (2 * orbital + 1)) & 1U));
}

void spin_dot(MatrixColumn &column, uint32_t bits, unsigned first,
             unsigned second, double coupling) {
  column[bits] += coupling * reference_spin_z(bits, first) * reference_spin_z(bits, second);
  // Apply the rightmost spin operator first. Spin operators are even, so the
  // tensor product is the ordinary physical S_i.S_j even for separated modes.
  for (bool first_raise : {false, true}) {
    uint32_t output = bits;
    double coefficient = coupling / 2;
    const unsigned first_from = 2 * first + (first_raise ? 1 : 0);
    const unsigned second_from = 2 * second + (first_raise ? 0 : 1);
    if (annihilate(output, second_from, coefficient) &&
        create(output, second_from ^ 1U, coefficient) &&
        annihilate(output, first_from, coefficient) &&
        create(output, first_from ^ 1U, coefficient)) {
      column[output] += coefficient;
    }
  }
}

void append_hopping(MatrixColumn &column, uint32_t bits, unsigned first,
             unsigned second, double t) {
  for (unsigned spin = 0; spin < 2; ++spin) {
    transfer(column, bits, 2 * first + spin, 2 * second + spin, -t);
    transfer(column, bits, 2 * second + spin, 2 * first + spin, -t);
  }
}

void reference_onsite(MatrixColumn &column, uint32_t bits, unsigned rung,
                     const PhysicsParams &p, double fraction) {
  const unsigned offset = 4 * rung;
  double diagonal = 0;
  for (unsigned layer = 0; layer < 2; ++layer) {
    const unsigned z = offset + 2 * layer, x = z + 1;
    const int nz = occupation(bits, z), nx = occupation(bits, x);
    diagonal += p.epsilon_z * nz + p.epsilon_x * nx - p.mu * (nz + nx);
    diagonal += (p.v - p.jh / 2) * nz * nx;
    spin_dot(column, bits, z, x, -2 * p.jh * fraction);
  }
  column[bits] += fraction * diagonal;
  append_hopping(column, bits, offset, offset + 2, p.tz_perp * fraction);
  spin_dot(column, bits, offset, offset + 2, p.jz_perp * fraction);
}

MatrixColumn reference_bond(uint32_t bits, const PhysicsParams &p, int type) {
  MatrixColumn column;
  reference_onsite(column, bits, 0, p, 0.25);
  reference_onsite(column, bits, 1, p, 0.25);
  const double tx = type == 1 ? p.tx1 : p.tx2;
  const double txz = type == 1 ? p.txz1 : p.txz2;
  const double jx = type == 1 ? p.jx1 : p.jx2;
  const double jxz = type == 1 ? p.jxz1 : p.jxz2;
  for (unsigned layer = 0; layer < 2; ++layer) {
    const unsigned az = 2 * layer, ax = az + 1;
    const unsigned bz = 4 + az, bx = bz + 1;
    append_hopping(column, bits, ax, bx, tx);
    append_hopping(column, bits, ax, bz, txz);
    append_hopping(column, bits, az, bx, txz);
    spin_dot(column, bits, ax, bx, jx);
    spin_dot(column, bits, ax, bz, jxz);
    spin_dot(column, bits, az, bx, jxz);
  }
  return column;
}

PhysicsParams nontrivial_physics() {
  PhysicsParams p{};
  p.epsilon_z = -0.13;
  p.epsilon_x = 0.428;
  p.v = 3;
  p.jh = 1;
  p.tz_perp = 0.17;
  p.jz_perp = 0.19;
  p.tx1 = 0.403;
  p.tx2 = 0.398;
  p.txz1 = 0.211;
  p.txz2 = -0.190;  // Also test hopping signs beyond the example table.
  p.jx1 = 0.065;
  p.jx2 = 0.0633;
  p.jxz1 = 0.0385;
  p.jxz2 = 0.0268;
  p.mu = 0.31;
  return p;
}

void check_basis_and_projected_algebra() {
  require(basis().size() == 81 && physical_index().dim() == 81,
          "Projected rung must have 3^4=81 states.");
  std::set<unsigned> masks;
  for (size_t coordinate = 0; coordinate < 81; ++coordinate) {
    const auto &state = basis()[coordinate];
    masks.insert(state.bits);
    require(basis_coordinate(state.bits) == coordinate, "Basis inverse mismatch.");
    require(state.particles == std::popcount(state.bits), "Wrong particle label.");
    int sz2 = 0;
    for (unsigned orbital = 0; orbital < 4; ++orbital) {
      require(((state.bits >> (2 * orbital)) & 3) != 3,
              "Forbidden orbital double occupancy.");
      sz2 += int(2 * reference_spin_z(state.bits, orbital));
    }
    require(state.sz2 == sz2, "Wrong spin label.");
    const auto qn = physical_index().GetQNSctFromActualCoor(coordinate).GetQn();
    require(particle_charge(qn) == state.particles, "Physical particle charge mismatch.");
    require(particle_charge(-qn) == -state.particles, "Negative particle charge decoding mismatch.");
    require(qn.IsFermionParityOdd() == bool(state.particles % 2),
            "Physical quantum number has incorrect fermion parity.");
    // Projected algebra {c_sigma,c_sigma^dagger}=1-n_opposite.
    for (unsigned mode = 0; mode < 8; ++mode) {
      double diagonal = 0;
      uint32_t output = state.bits;
      double coefficient = 1;
      if (create(output, mode, coefficient) && annihilate(output, mode, coefficient)) {
        require(output == state.bits, "Projected CAR changed basis state.");
        diagonal += coefficient;
      }
      output = state.bits;
      coefficient = 1;
      if (annihilate(output, mode, coefficient) && create(output, mode, coefficient)) {
        require(output == state.bits, "Projected CAR changed basis state.");
        diagonal += coefficient;
      }
      require_near(diagonal, 1 - ((state.bits >> (mode ^ 1U)) & 1U), "Projected CAR");
    }
  }
  require(masks.size() == 81, "Duplicate Fock basis states.");
  require(physical_index().GetQNScts().size() == (kConserveSpin ? 15 : 5),
          "Physical sectors must merge spin labels only in the N-only ansatz.");
  const auto up = physical_index().GetQNSctFromActualCoor(basis_coordinate(1U)).GetQn();
  const auto down = physical_index().GetQNSctFromActualCoor(basis_coordinate(2U)).GetQn();
  require((up == down) == !kConserveSpin, "Opposite-spin states have incorrect symmetry labels.");
  require(qlpeps::IPEPSCheckpointSymmetry<QNT>::Id() ==
          (kConserveSpin ? "fU1U1QN" : "fU1QN"), "Checkpoint symmetry schema mismatch.");
}

void check_transverse_spin() {
  const auto sx = transverse_spin_matrix("sx_top_z");
  const auto up = basis_coordinate(1U), down = basis_coordinate(2U);
  require_near(sx[up][down], 0.5, "Sx raises an isolated top-z spin with coefficient1/2");
  require_near(sx[down][up], 0.5, "Sx lowers an isolated top-z spin with coefficient1/2");
  require_near(sx[up][up], 0, "Sx has zero diagonal");
  require_near(sx[basis_coordinate(0)][basis_coordinate(0)], 0, "Sx vanishes on an empty orbital");
  for (size_t input = 0; input < basis().size(); ++input) {
    for (size_t output = 0; output < basis().size(); ++output) {
      require_near(sx[output][input], sx[input][output], "Sx Hermiticity");
      if (sx[output][input] == 0) continue;
      require(basis()[output].particles == basis()[input].particles, "Sx particle conservation");
      require(std::abs(basis()[output].sz2 - basis()[input].sz2) == 2,
              "Sx must connect different Sz components.");
      require((basis()[input].bits ^ basis()[output].bits) == 3U,
              "sx_top_z must flip only the top-z orbital spin.");
      require_near(sx[output][input], 0.5, "Sx spectator-independent flip coefficient");
    }
  }
#if !LNO_IPEPS_SPIN_U1
  // This zero-divergence tensor explicitly holds coherences between distinct
  // Sz states. The ansatz has not inherited the gate builder's auxiliary Sz.
  const auto transverse = build_one_site_operator(sx);
  require(transverse.Div() == QNT::Zero(), "N-only Sx must be a zero-charge operator.");
  require_near(transverse.GetElem({up, down}), 0.5, "N-only tensor retains cross-Sz matrix element");
  require_near(transverse.GetElem({down, up}), 0.5, "N-only tensor retains reverse cross-Sz element");
#endif
}

void check_evolution_gates() {
  PhysicsParams physics{};
  physics.v = 3;
  physics.jh = 1;
  EvolutionGateBuilder builder(physics);
  const auto &physical = physical_index();
  const auto identity = builder.build(0);
  for (size_t u = 0; u < basis().size(); ++u) {
    for (size_t v = 0; v < basis().size(); ++v) {
      const double chi = qlpeps::local_update_detail::PhysicalOperatorInputSign(physical, u, physical, v);
      for (const auto &gate : identity) require_near(gate.GetElem({u, v, u, v}), chi,
          "Exact graded identity gate");
    }
  }
  for (const auto &gate : identity) require_near(gate.GetQuasi2Norm() * gate.GetQuasi2Norm() /
      double(kPhysicalDimension * kPhysicalDimension), 1, "Identity gate has no extra coefficients");

  constexpr double tau = 0.07;
  const auto gates = builder.build(tau);
  const size_t vacuum = basis_coordinate(0);
  const size_t up_down = basis_coordinate(0b1001), down_up = basis_coordinate(0b0110);
  const size_t parallel = basis_coordinate(0b0101);
  const size_t isolated_up = basis_coordinate(1U), isolated_down = basis_coordinate(2U);
  const double triplet_weight = std::exp(-tau / 2), singlet_weight = std::exp(-tau);
  const double both_triplet_weight = std::exp(-tau);
  const double singlet_triplet_weight = std::exp(-1.5 * tau);
  for (const auto &gate : gates) {
    // In this nonnegative Hamiltonian the common minimum energy is zero.
    // A single occupied Ni has triplet/singlet energies2/4 and contributes
    // one quarter of its onsite Hamiltonian to this planar bond endpoint.
    require_near(gate.GetElem({up_down, vacuum, up_down, vacuum}),
        (triplet_weight + singlet_weight) / 2, "Hund gate diagonal spectral weights");
    require_near(gate.GetElem({up_down, vacuum, down_up, vacuum}),
        (triplet_weight - singlet_weight) / 2, "Hund spin-flip exponential");
    require_near(gate.GetElem({parallel, vacuum, parallel, vacuum}), triplet_weight,
        "Parallel Hund triplet gate weight");
    require_near(gate.GetElem({up_down, parallel, up_down, parallel}),
        (both_triplet_weight + singlet_triplet_weight) / 2, "Both bond endpoints carry onsite quarters");
    require_near(gate.GetElem({up_down, parallel, down_up, parallel}),
        (both_triplet_weight - singlet_triplet_weight) / 2, "Two-endpoint Hund spin-flip weight");
    require_near(gate.GetElem({isolated_up, vacuum, isolated_down, vacuum}), 0,
        "Intrinsic gate Sz conservation leaves spin-forbidden coefficients zero");
    const double one_layer_trace = 5 + 3 * std::exp(-tau) + std::exp(-2 * tau);
    const double expected_squared_norm = std::pow(one_layer_trace, 4);
    require_near(gate.GetQuasi2Norm() * gate.GetQuasi2Norm() / expected_squared_norm, 1,
        "Analytic full Hund exponential norm excludes missing or extra gate coefficients");
  }
}

void check_hund_and_exchange_spectra() {
  PhysicsParams p{};
  p.v = 3;
  p.jh = 1;
  const auto matrix = onsite_matrix(p);
  const auto up_up = basis_coordinate(0b0101), down_down = basis_coordinate(0b1010);
  const auto up_down = basis_coordinate(0b1001), down_up = basis_coordinate(0b0110);
  require_near(matrix[up_up][up_up], 2, "Parallel Hund triplet E=V-JH");
  require_near(matrix[down_down][down_down], 2, "Parallel Hund triplet E=V-JH");
  const double triplet = (matrix[up_down][up_down] + matrix[down_up][down_up] +
                         matrix[up_down][down_up] + matrix[down_up][up_down]) / 2;
  const double singlet = (matrix[up_down][up_down] + matrix[down_up][down_up] -
                         matrix[up_down][down_up] - matrix[down_up][up_down]) / 2;
  require_near(triplet, 2, "m=0 Hund triplet E=V-JH");
  require_near(singlet, 4, "Hund singlet E=V+JH");

  p = PhysicsParams{};
  p.jx1 = 1;
  const size_t first = basis_coordinate(1U << 2), second = basis_coordinate(1U << 2);
  double parallel_exchange = 0;
  for (const auto &term : bond_transitions(first, second, p, 1)) {
    if (term.output_first == first && term.output_second == second) parallel_exchange += term.value;
  }
  require_near(parallel_exchange, 0.25, "Literal J S.S must not include -J n.n/4");

  // Hand-worked signs: moving through one occupied intervening mode changes
  // the bare -t coefficient to +t. These examples do not use the reference
  // ladder helpers, so a shared sign error in both visitors is also detected.
  p = PhysicsParams{};
  p.tz_perp = 0.37;
  const size_t rung_in = basis_coordinate((1U << 0) | (1U << 2));
  const size_t rung_out = basis_coordinate((1U << 2) | (1U << 4));
  double rung_hop = 0;
  for (const auto &term : local_transitions(rung_in, p)) {
    if (term.output == rung_out) rung_hop += term.value;
  }
  require_near(rung_hop, p.tz_perp, "Rung hopping sign from intervening x electron");
  p = PhysicsParams{};
  p.txz1 = 0.23;
  const size_t pair_in = basis_coordinate((1U << 2) | (1U << 4));
  const size_t pair_out = basis_coordinate(1U << 4), vacuum = basis_coordinate(0);
  const size_t neighbor_out = basis_coordinate(1U << 0);
  double cross_hop = 0;
  for (const auto &term : bond_transitions(pair_in, vacuum, p, 1)) {
    if (term.output_first == pair_out && term.output_second == neighbor_out) cross_hop += term.value;
  }
  require_near(cross_hop, p.txz1, "Inter-rung cross hopping sign from intervening electron");
}

void check_all_matrix_elements() {
  const auto physics = nontrivial_physics();
  const auto physical = physical_index();
  for (int type : {1, 2}) {
    const auto tensor = build_bond_hamiltonian(physical, physical, physics, type);
    double expected_squared_norm = 0;
    for (size_t first = 0; first < 81; ++first) {
      for (size_t second = 0; second < 81; ++second) {
        const uint32_t mask = basis()[first].bits | (uint32_t(basis()[second].bits) << 8);
        const auto expected = reference_bond(mask, physics, type);
        MatrixColumn actual;
        for (const auto &entry : bond_transitions(first, second, physics, type)) {
          const uint32_t output = basis()[entry.output_first].bits |
              (uint32_t(basis()[entry.output_second].bits) << 8);
          actual[output] += entry.value;
        }
        std::set<uint32_t> outputs;
        for (const auto &[output, value] : expected) outputs.insert(output);
        for (const auto &[output, value] : actual) outputs.insert(output);
        const bool odd_first = basis()[first].particles % 2;
        const bool odd_second = basis()[second].particles % 2;
        const double input_chi = ((odd_first && odd_second) ^ odd_first ^ odd_second) ? -1 : 1;
        for (uint32_t output : outputs) {
          const double reference = expected.contains(output) ? expected.at(output) : 0;
          const double got = actual.contains(output) ? actual.at(output) : 0;
          require_near(got, reference, "Independent 16-mode Fock matrix element");
          const size_t out_first = basis_coordinate(output & 255U);
          const size_t out_second = basis_coordinate(output >> 8);
          require_near(input_chi * tensor.GetElem({first, second, out_first, out_second}),
               reference, "Graded input-column tensor encoding");
          if (std::abs(got) <= 1e-15) continue;
          require(std::popcount(mask) == std::popcount(output), "Particle conservation");
          require(basis()[first].sz2 + basis()[second].sz2 ==
                  basis()[out_first].sz2 + basis()[out_second].sz2, "Sz conservation");
          const auto reverse = reference_bond(output, physics, type);
          require_near(reverse.contains(mask) ? reverse.at(mask) : 0, got, "Hermiticity");
          expected_squared_norm += reference * reference;
        }
      }
    }
    // Matching all nonzero reference elements plus the norm excludes unexpected
    // extra tensor entries, without scanning 81^4 mostly zero coordinates.
    require_near((tensor.GetQuasi2Norm() * tensor.GetQuasi2Norm()) / expected_squared_norm, 1,
         "Tensor norm excludes spurious coefficients");
  }
  for (size_t input = 0; input < 81; ++input) {
    MatrixColumn expected;
    reference_onsite(expected, basis()[input].bits, 0, physics, 1);
    MatrixColumn actual;
    for (const auto &entry : local_transitions(input, physics)) actual[basis()[entry.output].bits] += entry.value;
    for (const auto &[output, value] : expected) {
      require_near(actual.contains(output) ? actual.at(output) : 0, value, "Independent onsite reference");
    }
    for (const auto &[output, value] : actual) {
      require_near(expected.contains(output) ? expected.at(output) : 0, value, "Extra onsite element");
    }
  }
}

void check_seeds(size_t spin_offset) {
  constexpr size_t rows = 4, cols = 4;
  constexpr double pi = 3.14159265358979323846;
  for (const auto &[name, qx, qy] : {
      std::tuple{"diagonal_plus", pi / 2, pi / 2},
      std::tuple{"diagonal_minus", pi / 2, -pi / 2},
      std::tuple{"stripe_x", pi, 0.0},
      std::tuple{"stripe_y", 0.0, pi},
      std::tuple{"neel", pi, pi},
      std::tuple{"fm", 0.0, 0.0}}) {
    for (const auto &charge_pattern : {"stripe_x", "stripe_y"}) {
      for (size_t charge_offset = 0; charge_offset < 4; ++charge_offset) {
        const auto labels = build_seed_labels(rows, cols, name, -1, charge_pattern, charge_offset, spin_offset);
        size_t particles = 0;
        int total_sz2 = 0;
        std::complex<double> fourier = 0;
        for (size_t row = 0; row < rows; ++row) {
          for (size_t col = 0; col < cols; ++col) {
            const auto &state = basis()[labels[row][col]];
            require(state.particles == 3, "Seed must have three electrons per rung.");
            particles += state.particles;
            total_sz2 += state.sz2;
            fourier += reference_spin_z(state.bits, 0) *
                std::exp(std::complex<double>(0, -(qx * col + qy * row)));
          }
        }
        require(particles == 48, "Wrong 4x4 seed filling.");
        require(total_sz2 == 0, "Competing AF-layer seeds must share total Sz=0: " + std::string(name));
        require(std::abs(fourier) / (rows * cols) > 0.3, "Seed wavevector mismatch: " + std::string(name));
      }
    }
  }
}

void check_zigzag_spin_phase() {
  PhysicsParams physics{};
  physics.jxz1 = 1;
  const qlpeps::UnitCellGeometry geometry(4, 4);
  for (const auto &name : {"diagonal_plus", "diagonal_minus", "stripe_x", "stripe_y"}) {
    for (size_t spin_offset = 0; spin_offset < 2; ++spin_offset) {
      const auto labels = build_seed_labels(4, 4, name, -1, "stripe_x", 0, spin_offset);
      double energy = 0;
      for (const auto &bond : geometry.Bonds()) {
        const auto next = geometry.Second(bond).site;
        const size_t a = labels[bond.first.row()][bond.first.col()];
        const size_t b = labels[next.row()][next.col()];
        for (const auto &term : bond_transitions(a, b, physics,
            bond_type(bond.first.row(), bond.first.col(), bond.orientation))) {
          if (term.output_first == a && term.output_second == b) energy += term.value;
        }
      }
      // A bond has two x-z products and one x electron per rung: each strong
      // bond contributes +/-1/2; one such bond/rung and two Ni/rung gives1/4.
      const double expected = std::string(name) == "diagonal_minus" ?
          (spin_offset == 0 ? -0.25 : 0.25) : 0;
      require_near(energy / 32, expected, "Zigzag phase energy per Ni");
    }
  }
}

void check_quarter_axis_seeds() {
  constexpr double pi = 3.14159265358979323846;
  for (const auto &name : {"stripe_pi2_x", "stripe_pi2_y"}) {
    const bool x_direction = std::string(name) == "stripe_pi2_x";
    for (const auto &charge : {"stripe_x", "stripe_y"}) {
      const bool parallel = x_direction == (std::string(charge) == "stripe_x");
      for (size_t spin_offset = 0; spin_offset < 4; ++spin_offset) {
        for (size_t charge_offset = 0; charge_offset < 4; ++charge_offset) {
          const bool balanced = !parallel || (spin_offset + charge_offset) % 2 == 1;
          bool rejected = false;
          std::vector<std::vector<size_t>> labels;
          try {
            labels = build_seed_labels(4, 4, name, -1, charge, charge_offset, spin_offset);
          } catch (const std::invalid_argument &) {
            rejected = true;
          }
          const bool should_reject = kConserveSpin && !balanced;
          require(rejected == should_reject, "Quarter-axis seed rejection must follow the ansatz spin constraint.");
          if (rejected) continue;
          int particles = 0, top_particles = 0, bottom_particles = 0, sz2 = 0;
          std::complex<double> amplitude = 0;
          for (size_t row = 0; row < 4; ++row) for (size_t col = 0; col < 4; ++col) {
            const auto &state = basis()[labels[row][col]];
            particles += state.particles;
            top_particles += std::popcount(unsigned(state.bits & 15));
            bottom_particles += std::popcount(unsigned(state.bits >> 4));
            sz2 += state.sz2;
            amplitude += reference_spin_z(state.bits, 0) * std::exp(std::complex<double>(
                0, -pi / 2 * double(x_direction ? col : row)));
          }
          require(particles == 48 && top_particles == 24 && bottom_particles == 24,
                  "Quarter-axis seed must share the layer fillings of competitors.");
          const size_t relative_phase = (spin_offset + 4 - charge_offset) % 4;
          const int expected_sz2 = balanced ? 0 : (relative_phase == 0 ? 16 : -16);
          require(sz2 == expected_sz2, "Quarter-axis seed spin polarization mismatch.");
          require_near(std::abs(amplitude) / 16, std::sqrt(0.125), "Quarter-axis seed wavevector");
        }
      }
    }
    const auto automatic = build_seed_labels(4, 4, name);
    const auto perpendicular = build_seed_labels(4, 4, name, -1,
        x_direction ? "stripe_y" : "stripe_x");
    require(automatic == perpendicular, "Automatic quarter-axis charge seed must be balanced.");
  }
}
}  // namespace

int main() {
  try {
    qlten::hp_numeric::SetCpuNumericsThreads(1);
    check_basis_and_projected_algebra();
    check_transverse_spin();
    check_hund_and_exchange_spectra();
    check_all_matrix_elements();
    for (size_t spin_offset = 0; spin_offset < 4; ++spin_offset) check_seeds(spin_offset);
    check_zigzag_spin_phase();
    check_quarter_axis_seeds();
    check_evolution_gates();
    std::cout << "PASS: 81-state projected basis, Hund spectra, literal exchange, all two-rung "
                 "Fock coefficients, Hermiticity, graded gates, transverse coherences and seed wavevectors ("
              << kSymmetryName << ").\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Bilayer model validation: " << error.what() << '\n';
    return 1;
  }
}
