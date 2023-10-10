
#ifndef TJMODEL_SRC_OPERATORS_H
#define TJMODEL_SRC_OPERATORS_H


#include "gqten/gqten.h"
#include "gqdouble.h"


#if SYMMETRY_LEVLE == 0
extern Tensor sz, sp,sm, id;
extern Tensor f, bupc, bupa, bdnc, bdna;
extern Tensor  nf, nup, ndn;
extern Tensor bdnc_multi_bupa, bupc_multi_bdna;
#endif


extern Tensor sz1, sp1, sm1, id1;
extern Tensor f1, bupc1, bupa1, bdnc1, bdna1;
extern Tensor nf1, nup1, ndn1;
extern Tensor bdnc_multi_bupa1, bupc_multi_bdna1;


extern Tensor sz2, sp2, sm2, id2;
extern Tensor f2, bupc2, bupa2, bdnc2, bdna2;
extern Tensor nf2, nup2, ndn2;
extern Tensor bdnc_multi_bupa2, bupc_multi_bdna2;

void OperatorInitial();

#endif // TJMODEL_SRC_OPERATORS_H