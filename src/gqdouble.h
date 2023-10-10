#ifndef TJMODEL_SRC_GQDOUBLE_H
#define TJMODEL_SRC_GQDOUBLE_H

#include "boost/serialization/complex.hpp"
#include "gqten/gqten.h"

#if SYMMETRY_LEVLE == 0
using QNT = gqten::special_qn::U1U1QN;
#elif SYMMETRY_LEVLE == 1

#include "./u1u1u1qn.h"

using QNT = gqten::special_qn::U1U1U1QN;
#else
#error Unsupported choice setting
#endif

using gqten::U1QNVal;
using gqten::QNCard;
using gqten::GQTenIndexDirType;
using TenElemT = gqten::GQTEN_Complex;


using Tensor = gqten::GQTensor<TenElemT, QNT>;
using QNSctT = gqten::QNSector<QNT>;
using IndexT = gqten::Index<QNT>;

#if SYMMETRY_LEVLE == 0
const auto qn0 = QNT(
    {QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}
);


const IndexT pb_out = IndexT({
                                 QNSctT(QNT({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(1))}), 1),
                                 QNSctT(QNT({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(-1))}), 1),
                                 QNSctT(QNT({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}), 1)},
                             GQTenIndexDirType::OUT
);
const auto pb_in = gqten::InverseIndex(pb_out);\
const IndexT pb_out_layer1 = pb_out;
const IndexT pb_out_layer2 = pb_out;
const auto pb_in_layer1 = pb_in;
const auto pb_in_layer2 = pb_in;
#elif SYMMETRY_LEVLE == 1
const auto qn0 = QNT("N1", 0, "N2", 0, "Sz", 0);
const IndexT pb_out_layer1 = IndexT({
                                        QNSctT(QNT("N1", 1, "N2", 0, "Sz", 1), 1),
                                        QNSctT(QNT("N1", 1, "N2", 0, "Sz", -1), 1),
                                        QNSctT(QNT("N1", 0, "N2", 0, "Sz", 0), 1)},
                                    GQTenIndexDirType::OUT
);

const IndexT pb_out_layer2 = IndexT({
                                        QNSctT(QNT("N1", 0, "N2", 1, "Sz", 1), 1),
                                        QNSctT(QNT("N1", 0, "N2", -1, "Sz", -1), 1),
                                        QNSctT(QNT("N1", 0, "N2", 0, "Sz", 0), 1)},
                                    GQTenIndexDirType::OUT
);
const auto pb_in_layer1 = gqten::InverseIndex(pb_out_layer1);
const auto pb_in_layer2 = gqten::InverseIndex(pb_out_layer2);
#endif
#endif // TJMODEL_SRC_GQDOUBLE_H
