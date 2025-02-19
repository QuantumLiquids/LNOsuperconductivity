#ifndef HUBBARD_OPERATORS_H
#define HUBBARD_OPERATORS_H

#include "hilbert_space.h"

class HubbardOperators {
 public:
  // Spin operators
  Tensor sz;
  Tensor sp;
  Tensor sm;
  Tensor id;
  Tensor sx;
  Tensor sy;

  // Fermionic operator
  Tensor f;

  // Hardcore boson operators
  Tensor bupc;  // b_up^creation
  Tensor bupa;  // b_up^annihilation
  Tensor bdnc;  // b_down^creation
  Tensor bdna;  // b_down^annihilation

  // Composite operators
  Tensor bupcF; // bupc * f
  Tensor bupaF;
  Tensor Fbdnc;
  Tensor Fbdna;

  // Pairing operators
  Tensor cupccdnc;  // c_up^creation * c_down^creation
  Tensor cdnacupa;  // onsite pair operator (c_down*c_up)
  Tensor cupccdna;  // alias for sp
  Tensor cdnccupa;  // alias for sm

  // Other operators
  Tensor Uterm;   // Hubbard U term
  Tensor nf;      // fermion number (nup + ndn)
  Tensor nfsquare;// square of fermion number
  Tensor nup;     // spin-up number operator
  Tensor ndn;     // spin-down number operator

  // Constructor (initializes all the operators)
  HubbardOperators();
};

#endif // HUBBARD_OPERATORS_H
