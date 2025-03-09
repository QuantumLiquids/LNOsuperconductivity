#ifndef TJMODEL_SRC_QLDOUBLE_H
#define TJMODEL_SRC_QLDOUBLE_H

#include "qlten/qlten.h"
#ifndef SYMMETRY_LEVLE
#define SYMMETRY_LEVLE 0
#endif

#if SYMMETRY_LEVEL == 0
using QNT = qlten::special_qn::U1U1QN;
#elif SYMMETRY_LEVEL == 1

#include "./u1u1u1qn.h"

using QNT = qlten::special_qn::U1QNT;
#else
#error Unsupported SYMMETRY_LEVEL
#endif

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

#if SYMMETRY_LEVEL == 0
const auto qn0 = QNT(
    {QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}
);

const IndexT pb_out = IndexT({
                                 QNSctT(QNT({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(1))}), 1),
                                 QNSctT(QNT({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(-1))}), 1),
                                 QNSctT(QNT({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}), 1)},
                             TenIndexDirType::OUT
);
const auto pb_in = qlten::InverseIndex(pb_out);

const IndexT pb_out_layer1 = pb_out;
const IndexT pb_out_layer2 = pb_out;
const auto pb_in_layer1 = pb_in;
const auto pb_in_layer2 = pb_in;
#elif SYMMETRY_LEVEL == 1
const auto qn0 = QNT("N1", 0, "N2", 0, "Sz", 0);
const IndexT pb_out_layer1 = IndexT({
                                        QNSctT(QNT("N1", 1, "N2", 0, "Sz", 1), 1),
                                        QNSctT(QNT("N1", 1, "N2", 0, "Sz", -1), 1),
                                        QNSctT(QNT("N1", 0, "N2", 0, "Sz", 0), 1)},
                                    QLTenIndexDirType::OUT
);

const IndexT pb_out_layer2 = IndexT({
                                        QNSctT(QNT("N1", 0, "N2", 1, "Sz", 1), 1),
                                        QNSctT(QNT("N1", 0, "N2", -1, "Sz", -1), 1),
                                        QNSctT(QNT("N1", 0, "N2", 0, "Sz", 0), 1)},
                                    QLTenIndexDirType::OUT
);
const auto pb_in_layer1 = qlten::InverseIndex(pb_out_layer1);
const auto pb_in_layer2 = qlten::InverseIndex(pb_out_layer2);
#endif
#endif // TJMODEL_SRC_QLDOUBLE_H
