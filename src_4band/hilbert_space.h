#ifndef HILBERT_SPACE_H
#define HILBERT_SPACE_H
#include "qlten/qlten.h"

using qlten::QNCard;
using qlten::U1QNVal;
using qlten::TenIndexDirType;

#ifdef USE_REAL
using TenElemT = qlten::QLTEN_Double;
#else
using TenElemT = qlten::QLTEN_Complex;
#endif

#ifndef SYM_LEVEL
#define SYM_LEVEL 1
#endif

#if SYM_LEVEL == 1
using qlten::special_qn::U1U1QN;
using QNT = U1U1QN;
using Tensor = qlten::QLTensor<TenElemT, QNT>;
using QNSctT = qlten::QNSector<QNT>;
using IndexT = qlten::Index<QNT>;

const auto qn0 = QNT(
    {QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}
);

const IndexT pb_out = IndexT({QNSctT(QNT({QNCard("N", U1QNVal(2)), QNCard("Sz", U1QNVal(0))}), 1),
                              QNSctT(QNT({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(1))}), 1),
                              QNSctT(QNT({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(-1))}), 1),
                              QNSctT(QNT({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}), 1)},
                             TenIndexDirType::OUT
);
#elif SYM_LEVEL == 0  // no symmetry
using qlten::special_qn::TrivialRepQN;
using QNT = TrivialRepQN;
using Tensor = qlten::QLTensor<TenElemT, TrivialRepQN>;
using QNSctT = qlten::QNSector<TrivialRepQN>;
using IndexT = qlten::Index<TrivialRepQN>;

const auto qn0 = TrivialRepQN();

const IndexT pb_out = IndexT({QNSctT(TrivialRepQN(), 4)},
                             TenIndexDirType::OUT);
#endif
const auto pb_in = qlten::InverseIndex(pb_out);

void OperatorInitial();

#endif //HILBERT_SPACE_H