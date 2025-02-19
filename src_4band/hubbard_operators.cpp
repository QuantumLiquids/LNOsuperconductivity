#include "hubbard_operators.h"

// The constructor uses an initializer list to create all the Tensors
// with the given dimensions {pb_in, pb_out} (assumed to be defined elsewhere).
HubbardOperators::HubbardOperators()
    : sz({pb_in, pb_out}),
      sp({pb_in, pb_out}),
      sm({pb_in, pb_out}),
      id({pb_in, pb_out}),
      sx({pb_in, pb_out}),
      sy({pb_in, pb_out}),
      f({pb_in, pb_out}),
      bupc({pb_in, pb_out}),
      bupa({pb_in, pb_out}),
      bdnc({pb_in, pb_out}),
      bdna({pb_in, pb_out}),
      bupcF({pb_in, pb_out}),
      bupaF({pb_in, pb_out}),
      Fbdnc({pb_in, pb_out}),
      Fbdna({pb_in, pb_out}),
      cupccdnc({pb_in, pb_out}),
      cdnacupa({pb_in, pb_out}),
      cupccdna({pb_in, pb_out}),
      cdnccupa({pb_in, pb_out}),
      Uterm({pb_in, pb_out}),
      nf({pb_in, pb_out}),
      nfsquare({pb_in, pb_out}),
      nup({pb_in, pb_out}),
      ndn({pb_in, pb_out}) {
  // Initialize the spin operators
  sz({1, 1}) = 0.5;
  sz({2, 2}) = -0.5;
  sp({1, 2}) = 1.0;
  sm({2, 1}) = 1.0;
  id({0, 0}) = 1;
  id({1, 1}) = 1;
  id({2, 2}) = 1;
  id({3, 3}) = 1;
  sx({1, 2}) = 0.5;
  sx({2, 1}) = 0.5;
  sy({1, 2}) = std::complex<double>(0, -0.5);
  sy({2, 1}) = std::complex<double>(0, 0.5);

  // Initialize the fermionic operator
  f({0, 0}) = 1;
  f({1, 1}) = -1;
  f({2, 2}) = -1;
  f({3, 3}) = 1;

  // Initialize the hardcore boson operators
  bupc({0, 2}) = 1;
  bupc({1, 3}) = 1;
  bdnc({0, 1}) = 1;
  bdnc({2, 3}) = 1;
  bupa({2, 0}) = 1;
  bupa({3, 1}) = 1;
  bdna({1, 0}) = 1;
  bdna({3, 2}) = 1;

  // Initialize the composite operators
  bupcF({0, 2}) = -1;
  bupcF({1, 3}) = 1;
  Fbdnc({0, 1}) = 1;
  Fbdnc({2, 3}) = -1;
  bupaF({2, 0}) = 1;
  bupaF({3, 1}) = -1;
  Fbdna({1, 0}) = -1;
  Fbdna({3, 2}) = 1;

  // Initialize the pairing operators
  cupccdnc({0, 3}) = 1;
  cdnacupa({3, 0}) = 1;
  // Define cupccdna and cdnccupa as aliases for sp and sm respectively
  cupccdna = sp;
  cdnccupa = sm;

  // Initialize the Hubbard U term operator
  Uterm({0, 0}) = 1;

  // Initialize the fermion number operators
  nf({0, 0}) = 2;
  nf({1, 1}) = 1;
  nf({2, 2}) = 1;
  nfsquare({0, 0}) = 4;
  nfsquare({1, 1}) = 1;
  nfsquare({2, 2}) = 1;
  nup({0, 0}) = 1;
  nup({1, 1}) = 1;
  ndn({0, 0}) = 1;
  ndn({2, 2}) = 1;
}
