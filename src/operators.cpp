#include "gqdouble.h"

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

    initialized = true;
  }
}
