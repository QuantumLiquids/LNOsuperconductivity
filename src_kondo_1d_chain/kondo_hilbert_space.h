#ifndef KONDO_HILBERT_SPACE_H
#define KONDO_HILBERT_SPACE_H

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"

using QNT = qlten::special_qn::U1U1QN;

using qlten::U1QNVal;
using qlten::QNCard;
using qlten::TenIndexDirType;

#ifdef USE_REAL
using TenElemT = qlten::QLTEN_Double;
#else
using TenElemT = qlten::QLTEN_Complex;
#endif

using Tensor = qlten::QLTensor<TenElemT, QNT>;
using QNSctT = qlten::QNSector<QNT>;
using IndexT = qlten::Index<QNT>;

const auto qn0 = QNT(
    {QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}
);

qlmps::sites::HubbardSite<QNT> hubbard_site;

const IndexT pb_outE = hubbard_site.phys_bond_out;
const auto pb_inE = qlten::InverseIndex(pb_outE);

struct SpinOneHalfSiteU1U1 : public qlmps::sites::ModelSiteBase<QNT> {
  using QNSctT = qlten::QNSector<QNT>;
  using IndexT = qlten::Index<QNT>;

  SpinOneHalfSiteU1U1() {
    // Spin Sz conservation
    this->phys_bond_out = IndexT({QNSctT(QNT({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(1))}), 1),
                                  QNSctT(QNT({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(-1))}), 1)},
                                 TenIndexDirType::OUT);

    this->phys_bond_in = InverseIndex(this->phys_bond_out);
  }

  // Define the order of the basis, map to the numbers 0 and 1
  const size_t spin_up = 0;
  const size_t spin_down = 1;

}; // SpinOneHalfSite
struct SpinOneHalfOperatorsU1U1 {
  using QNSctT = qlten::QNSector<QNT>;
  using IndexT = qlten::Index<QNT>;
  using Tensor = qlten::QLTensor<TenElemT, QNT>;

  SpinOneHalfOperatorsU1U1() : SpinOneHalfOperatorsU1U1(SpinOneHalfSiteU1U1()) {};

  SpinOneHalfOperatorsU1U1(const SpinOneHalfSiteU1U1 &site) : sz({site.phys_bond_in, site.phys_bond_out}),
                                                              sp({site.phys_bond_in, site.phys_bond_out}),
                                                              sm({site.phys_bond_in, site.phys_bond_out}),
                                                              id({site.phys_bond_in, site.phys_bond_out}) {
    const size_t spin_up = site.spin_up;
    const size_t spin_down = site.spin_down;

    // Spin operators
    sz({spin_up, spin_up}) = 0.5;         // ⟨↑| S_z |↑⟩ = 0.5
    sz({spin_down, spin_down}) = -0.5;    // ⟨↓| S_z |↓⟩ = -0.5
    sp({spin_down, spin_up}) = 1.0;       // ⟨↑| S^+ |↓⟩ = 1.0
    sm({spin_up, spin_down}) = 1.0;       // ⟨↓| S^- |↑⟩ = 1.0
    id({spin_up, spin_up}) = 1.0;         // ⟨↑| I |↑⟩ = 1
    id({spin_down, spin_down}) = 1.0;     // ⟨↓| I |↓⟩ = 1
  }

  // Spin operators
  Tensor sz;
  Tensor sp;
  Tensor sm;
  Tensor id;
 private:
};

auto local_spin_site = SpinOneHalfSiteU1U1();
auto pb_outL = local_spin_site.phys_bond_out;
auto pb_inL = InverseIndex(pb_outL);

#endif // KONDO_HILBERT_SPACE_H
