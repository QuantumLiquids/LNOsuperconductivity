// SPDX-License-Identifier: MIT
#ifndef LNO_IPEPS_BILAYER_TWO_ORBITAL_ENERGY_COMPONENTS_H
#define LNO_IPEPS_BILAYER_TWO_ORBITAL_ENERGY_COMPONENTS_H

#include <array>

#include "model.h"

namespace bilayer_ipeps {

inline constexpr std::array<const char *, 10> kEnergyComponentNames{
    "hopping_x", "hopping_xz", "exchange_x", "exchange_xz", "orbital_energy",
    "density_v", "hund", "interlayer_hopping", "interlayer_exchange", "chemical_potential"};

// PhysicsParams{} has zero couplings. Applying bond_transitions to each piece
// preserves the full Hamiltonian's fermion signs and endpoint onsite weight 1/4.
inline std::array<PhysicsParams, kEnergyComponentNames.size()> energy_component_physics(
    const PhysicsParams &p) {
  std::array<PhysicsParams, kEnergyComponentNames.size()> pieces{};
  for (auto &piece : pieces) piece.txz_y_sign = p.txz_y_sign;
  pieces[0].tx1 = p.tx1;
  pieces[0].tx2 = p.tx2;
  pieces[1].txz1 = p.txz1;
  pieces[1].txz2 = p.txz2;
  pieces[2].jx1 = p.jx1;
  pieces[2].jx2 = p.jx2;
  pieces[3].jxz1 = p.jxz1;
  pieces[3].jxz2 = p.jxz2;
  pieces[4].epsilon_z = p.epsilon_z;
  pieces[4].epsilon_x = p.epsilon_x;
  pieces[5].v = p.v;
  // Includes both -2 JH Sz.Sx and -JH nz*nx/2, as in the supplied model.
  pieces[6].jh = p.jh;
  pieces[7].tz_perp = p.tz_perp;
  pieces[8].jz_perp = p.jz_perp;
  pieces[9].mu = p.mu;
  return pieces;
}

}  // namespace bilayer_ipeps

#endif  // LNO_IPEPS_BILAYER_TWO_ORBITAL_ENERGY_COMPONENTS_H
