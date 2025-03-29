#include "tJ_operators.h"

tJOperators::tJOperators()
    : sz({pb_in, pb_out}),
      sp({pb_in, pb_out}),
      sm({pb_in, pb_out}),
      id({pb_in, pb_out}),
      f({pb_in, pb_out}),
      bupc({pb_in, pb_out}),
      bupa({pb_in, pb_out}),
      bdnc({pb_in, pb_out}),
      bdna({pb_in, pb_out}),
      nf({pb_in, pb_out}),
      nup({pb_in, pb_out}),
      ndn({pb_in, pb_out}),
      bdnc_multi_bupa({pb_in, pb_out}),
      bupc_multi_bdna({pb_in, pb_out}) {
  // Initialize spin operators
  sz({0, 0}) = 0.5;
  sz({1, 1}) = -0.5;
  sp({1, 0}) = 1.0;
  sm({0, 1}) = 1.0;
  id({0, 0}) = 1;
  id({1, 1}) = 1;
  id({2, 2}) = 1;

  // Initialize the fermion operator
  f({0, 0}) = -1;
  f({1, 1}) = -1;
  f({2, 2}) = 1;

  // Initialize the boson operators
  bupc({2, 0}) = 1;
  bdnc({2, 1}) = 1;
  bupa({0, 2}) = 1;
  bdna({1, 2}) = 1;

  // Initialize number operators
  nf({0, 0}) = 1;
  nf({1, 1}) = 1;
  nup({0, 0}) = 1;
  ndn({1, 1}) = 1;
}
