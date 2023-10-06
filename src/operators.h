
#ifndef TJMODEL_SRC_OPERATORS_H
#define TJMODEL_SRC_OPERATORS_H


#include "gqten/gqten.h"
#include "gqdouble.h"

extern Tensor sz, sp,sm, id;
extern Tensor f, bupc, bupa, bdnc, bdna;
extern Tensor  nf, nup, ndn;
extern Tensor bdnc_multi_bupa, bupc_multi_bdna;

void OperatorInitial();

#endif // TJMODEL_SRC_OPERATORS_H