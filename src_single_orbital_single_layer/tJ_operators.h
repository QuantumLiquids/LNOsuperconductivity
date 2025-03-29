#ifndef TJ_OPERATORS_H
#define TJ_OPERATORS_H

#include "qlten/qlten.h"
#include "../src_single_orbital/tJ_type_hilbert_space.h"

class tJOperators {
 public:
  // Spin operators
  Tensor sz;
  Tensor sp;
  Tensor sm;
  Tensor id;

  // Fermion and boson operators
  Tensor f;
  Tensor bupc;
  Tensor bupa;
  Tensor bdnc;
  Tensor bdna;

  // Number operators
  Tensor nf;
  Tensor nup;
  Tensor ndn;

  // Composite operators
  Tensor bdnc_multi_bupa;
  Tensor bupc_multi_bdna;

  // Constructor that initializes all the operators
  tJOperators();
};

#endif // TJ_OPERATORS_H
