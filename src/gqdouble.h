#ifndef TJMODEL_SRC_GQDOUBLE_H
#define TJMODEL_SRC_GQDOUBLE_H
#include "boost/serialization/complex.hpp"
#include "gqten/gqten.h"

using gqten::QNCard;
using gqten::U1QNVal;
using gqten::GQTenIndexDirType;

using TenElemT = gqten::GQTEN_Complex;
using U1U1QN = gqten::special_qn::U1U1QN;
using Tensor = gqten::GQTensor<TenElemT, U1U1QN>;

using QNSctT = gqten::QNSector<U1U1QN>;
using IndexT = gqten::Index<U1U1QN>;


// possibly use particle number U1 x U1 & spin U1, if interlayer hopping = 0

const auto qn0 = U1U1QN(
	{QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal(0))}
    );


const IndexT pb_out = IndexT({
      QNSctT(U1U1QN({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal( 1))}), 1),
      QNSctT(U1U1QN({QNCard("N", U1QNVal(1)), QNCard("Sz", U1QNVal(-1))}), 1),
      QNSctT(U1U1QN({QNCard("N", U1QNVal(0)), QNCard("Sz", U1QNVal( 0))}), 1) },
      GQTenIndexDirType::OUT
    );
const auto pb_in = gqten::InverseIndex(pb_out);

#endif // TJMODEL_SRC_GQDOUBLE_H
