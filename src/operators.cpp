#include "gqdouble.h"

#if SYMMETRY_LEVEL == 0
Tensor sz = Tensor({pb_in, pb_out});
Tensor sp = Tensor({pb_in, pb_out});
Tensor sm = Tensor({pb_in, pb_out});
Tensor id = Tensor({pb_in, pb_out});

Tensor f = Tensor({pb_in, pb_out}); //fermion's insertion operator

Tensor bupc = Tensor({pb_in, pb_out}); //hardcore boson, b_up^creation, used for JW transformation
Tensor bupa = Tensor({pb_in, pb_out}); //hardcore boson, b_up^annihilation
Tensor bdnc = Tensor({pb_in, pb_out}); //hardcore boson, b_down^creation
Tensor bdna = Tensor({pb_in, pb_out}); //hardcore boson, b_down^annihilation



Tensor nf = Tensor({pb_in, pb_out}); // nup+ndown, fermion number
Tensor nup = Tensor({pb_in, pb_out}); // fermion number of spin up
Tensor ndn = Tensor({pb_in, pb_out}); // ndown

Tensor bdnc_multi_bupa = Tensor({pb_in, pb_out});
Tensor bupc_multi_bdna = Tensor({pb_in, pb_out});

Tensor sz1 = sz;
Tensor sp1 = sp;
Tensor sm1 = sm;
Tensor id1 = id;

Tensor f1 = f;
Tensor bupc1 = bupc;
Tensor bupa1 = bupa;
Tensor bdnc1 = bdnc;
Tensor bdna1 = bdna;

Tensor nf1 = nf;
Tensor nup1 = nup;
Tensor ndn1 = ndn;

Tensor bdnc_multi_bupa1 = bdnc_multi_bupa;
Tensor bupc_multi_bdna1 = bupc_multi_bdna;

Tensor sz2 = sz;
Tensor sp2 = sp;
Tensor sm2 = sm;
Tensor id2 = id;

Tensor f2 = f;
Tensor bupc2 = bupc;
Tensor bupa2 = bupa;
Tensor bdnc2 = bdnc;
Tensor bdna2 = bdna;

Tensor nf2 = nf;
Tensor nup2 = nup;
Tensor ndn2 = ndn;

Tensor bdnc_multi_bupa2 = bdnc_multi_bupa;
Tensor bupc_multi_bdna2 = bupc_multi_bdna;

void OperatorInitial() {
  static bool initialized = false;
  if (!initialized) {
    sz({0, 0}) = 0.5;
    sz({1, 1}) = -0.5;
    sp({0, 1}) = 1.0;
    sm({1, 0}) = 1.0;
    id({0, 0}) = 1;
    id({1, 1}) = 1;
    id({2, 2}) = 1;

    f({0, 0}) = -1;
    f({1, 1}) = -1;
    f({2, 2}) = 1;


    bupc({0, 2}) = 1;
    bdnc({1, 2}) = 1;
    bupa({2, 0}) = 1;
    bdna({2, 1}) = 1;

    nf({0, 0}) = 1;
    nf({1, 1}) = 1;


    nup({0, 0}) = 1;
    ndn({1, 1}) = 1;

    bdnc_multi_bupa({1, 0}) = 1;
    bupc_multi_bdna({0, 1}) = 1;

    sz1 = sz;
    sp1 = sp;
    sm1 = sm;
    id1 = id;
    f1 = f;
    bupc1 = bupc;
    bupa1 = bupa;
    bdnc1 = bdnc;
    bdna1 = bdna;
    nf1 = nf;
    nup1 = nup;
    ndn1 = ndn;
    bdnc_multi_bupa1 = bdnc_multi_bupa;
    bupc_multi_bdna1 = bupc_multi_bdna;
    sz2 = sz;
    sp2 = sp;
    sm2 = sm;
    id2 = id;
    f2 = f;
    bupc2 = bupc;
    bupa2 = bupa;
    bdnc2 = bdnc;
    bdna2 = bdna;
    nf2 = nf;
    nup2 = nup;
    ndn2 = ndn;
    bdnc_multi_bupa2 = bdnc_multi_bupa;
    bupc_multi_bdna2 = bupc_multi_bdna;
    initialized = true;
  }
}

#elif SYMMETRY_LEVEL == 1

Tensor sz1 = Tensor({pb_in_layer1, pb_out_layer1});
Tensor sp1 = Tensor({pb_in_layer1, pb_out_layer1});
Tensor sm1 = Tensor({pb_in_layer1, pb_out_layer1});
Tensor id1 = Tensor({pb_in_layer1, pb_out_layer1});

Tensor f1 = Tensor({pb_in_layer1, pb_out_layer1}); //fermion's insertion operator

Tensor bupc1 = Tensor({pb_in_layer1, pb_out_layer1}); //hardcore boson, b_up^creation, used for JW transformation
Tensor bupa1 = Tensor({pb_in_layer1, pb_out_layer1}); //hardcore boson, b_up^annihilation
Tensor bdnc1 = Tensor({pb_in_layer1, pb_out_layer1}); //hardcore boson, b_down^creation
Tensor bdna1 = Tensor({pb_in_layer1, pb_out_layer1}); //hardcore boson, b_down^annihilation

Tensor nf1 = Tensor({pb_in_layer1, pb_out_layer1}); // nup+ndown, fermion number
Tensor nup1 = Tensor({pb_in_layer1, pb_out_layer1}); // fermion number of spin up
Tensor ndn1 = Tensor({pb_in_layer1, pb_out_layer1}); // ndown

Tensor bdnc_multi_bupa1 = Tensor({pb_in_layer1, pb_out_layer1});
Tensor bupc_multi_bdna1 = Tensor({pb_in_layer1, pb_out_layer1});

Tensor sz2 = Tensor({pb_in_layer2, pb_out_layer2});
Tensor sp2 = Tensor({pb_in_layer2, pb_out_layer2});
Tensor sm2 = Tensor({pb_in_layer2, pb_out_layer2});
Tensor id2 = Tensor({pb_in_layer2, pb_out_layer2});

Tensor f2 = Tensor({pb_in_layer2, pb_out_layer2}); //fermion's insertion operator

Tensor bupc2 = Tensor({pb_in_layer2, pb_out_layer2}); //hardcore boson, b_up^creation, used for JW transformation
Tensor bupa2 = Tensor({pb_in_layer2, pb_out_layer2}); //hardcore boson, b_up^annihilation
Tensor bdnc2 = Tensor({pb_in_layer2, pb_out_layer2}); //hardcore boson, b_down^creation
Tensor bdna2 = Tensor({pb_in_layer2, pb_out_layer2}); //hardcore boson, b_down^annihilation

Tensor nf2 = Tensor({pb_in_layer2, pb_out_layer2}); // nup+ndown, fermion number
Tensor nup2 = Tensor({pb_in_layer2, pb_out_layer2}); // fermion number of spin up
Tensor ndn2 = Tensor({pb_in_layer2, pb_out_layer2}); // ndown

Tensor bdnc_multi_bupa2 = Tensor({pb_in_layer2, pb_out_layer2});
Tensor bupc_multi_bdna2 = Tensor({pb_in_layer2, pb_out_layer2});


void OperatorInitial() {
  static bool initialized = false;
  if (!initialized) {

    sz1({0, 0}) = 0.5;
    sz1({1, 1}) = -0.5;
    sp1({0, 1}) = 1.0;
    sm1({1, 0}) = 1.0;
    id1({0, 0}) = 1;
    id1({1, 1}) = 1;
    id1({2, 2}) = 1;

    f1({0, 0}) = -1;
    f1({1, 1}) = -1;
    f1({2, 2}) = 1;

    bupc1({0, 2}) = 1;
    bdnc1({1, 2}) = 1;
    bupa1({2, 0}) = 1;
    bdna1({2, 1}) = 1;

    nf1({0, 0}) = 1;
    nf1({1, 1}) = 1;

    nup1({0, 0}) = 1;
    ndn1({1, 1}) = 1;

    bdnc_multi_bupa1({1, 0}) = 1;
    bupc_multi_bdna1({0, 1}) = 1;


    sz2({0, 0}) = 0.5;
    sz2({1, 1}) = -0.5;
    sp2({0, 1}) = 1.0;
    sm2({1, 0}) = 1.0;
    id2({0, 0}) = 1;
    id2({1, 1}) = 1;
    id2({2, 2}) = 1;

    f2({0, 0}) = -1;
    f2({1, 1}) = -1;
    f2({2, 2}) = 1;

    bupc2({0, 2}) = 1;
    bdnc2({1, 2}) = 1;
    bupa2({2, 0}) = 1;
    bdna2({2, 1}) = 1;

    nf2({0, 0}) = 1;
    nf2({1, 1}) = 1;

    nup2({0, 0}) = 1;
    ndn2({1, 1}) = 1;

    bdnc_multi_bupa2({1, 0}) = 1;
    bupc_multi_bdna2({0, 1}) = 1;

    initialized = true;
  }
}

#endif
