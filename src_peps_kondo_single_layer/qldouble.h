/*
 * Local types and local Hilbert space definition for PEPS Kondo lattice.
 *
 * IMPORTANT:
 * - Requires TENSOR_SYMMETRY_LEVEL to be defined at compile time:
 *   -DTENSOR_SYMMETRY_LEVEL=0|1|2|3
 *
 * Local Hilbert space is 8-dimensional = (Hubbard electron 4 states) x (local spin 2 states).
 *
 * Basis encoding (fixed):
 *   electron: 0=|up dn>, 1=|up>, 2=|dn>, 3=|0>
 *   spin    : 0=|Up>,    1=|Dn>
 *   idx = 2*electron + spin
 *
 * Fermion parity: depends only on electron number (local spin is bosonic).
 * Total Sz (when present) = Sz_electron + Sz_local.
 */

#ifndef LNO_PEPS_KONDO_SINGLE_LAYER_QLDOUBLE_H
#define LNO_PEPS_KONDO_SINGLE_LAYER_QLDOUBLE_H

#include "qlten/qlten.h"

#if !defined(TENSOR_SYMMETRY_LEVEL)
#error "Please define TENSOR_SYMMETRY_LEVEL via CMake (-DTENSOR_SYMMETRY_LEVEL=0|1|2|3)."
#endif

using TenElemT = qlten::QLTEN_Double;

#if TENSOR_SYMMETRY_LEVEL == 0
// Fermion parity only.
using QNT = qlten::special_qn::fZ2QN;
const QNT qn0 = QNT(0);
#elif TENSOR_SYMMETRY_LEVEL == 1
// Fermion particle number only (U1).
using QNT = qlten::special_qn::fU1QN;
const QNT qn0 = QNT(0);
#elif TENSOR_SYMMETRY_LEVEL == 2
// Fermion particle number + total Sz (U1 x U1).
using QNT = qlten::special_qn::fU1U1QN;  // N, Sz(2*Sz integer)
const QNT qn0 = QNT(0, 0);
#elif TENSOR_SYMMETRY_LEVEL == 3
// Fermion parity + total Sz.
using QNT = qlten::special_qn::fZ2U1QN;  // parity, Sz(2*Sz integer)
const QNT qn0 = QNT(0, 0);
#endif

using qlten::Index;
using qlten::QLTensor;
using qlten::QNSector;

using QNSctT = QNSector<QNT>;
using IndexT = Index<QNT>;
using Tensor = QLTensor<TenElemT, QNT>;

// Helper: build local 8-state index with appropriate quantum numbers.
// We use IN direction for ket space, consistent with the SU examples in finite-size_PEPS_tJ.
static inline IndexT BuildLocalHilbertKet() {
#if TENSOR_SYMMETRY_LEVEL == 0
  // QNT = (parity)
  // electron parity: D(0), U(1), d(1), 0(0); local spin doesn't change parity.
  // Order: (D⊗Up, D⊗Dn, U⊗Up, U⊗Dn, d⊗Up, d⊗Dn, 0⊗Up, 0⊗Dn)
  return IndexT({QNSctT(QNT(0), 2),  // D ⊗ {Up,Dn}
                 QNSctT(QNT(1), 2),  // U ⊗ {Up,Dn}
                 QNSctT(QNT(1), 2),  // d ⊗ {Up,Dn}
                 QNSctT(QNT(0), 2)}, // 0 ⊗ {Up,Dn}
                qlten::TenIndexDirType::IN);
#elif TENSOR_SYMMETRY_LEVEL == 1
  // QNT = (N)
  return IndexT({QNSctT(QNT(2), 2),  // D (N=2) ⊗ {Up,Dn}
                 QNSctT(QNT(1), 2),  // U (N=1)
                 QNSctT(QNT(1), 2),  // d (N=1)
                 QNSctT(QNT(0), 2)}, // 0 (N=0)
                qlten::TenIndexDirType::IN);
#elif TENSOR_SYMMETRY_LEVEL == 2
  // QNT = (N, Sz)
  // D: (2, 0) with local Sz ±1 => (2, ±1)
  // U: (1, +1) with local Sz ±1 => (1, +2) and (1, 0)
  // d: (1, -1) with local Sz ±1 => (1, 0) and (1, -2)
  // 0: (0, 0) with local Sz ±1 => (0, ±1)
  return IndexT({QNSctT(QNT(2, +1), 1),  // D⊗Up
                 QNSctT(QNT(2, -1), 1),  // D⊗Dn
                 QNSctT(QNT(1, +2), 1),  // U⊗Up
                 QNSctT(QNT(1,  0), 1),  // U⊗Dn
                 QNSctT(QNT(1,  0), 1),  // d⊗Up
                 QNSctT(QNT(1, -2), 1),  // d⊗Dn
                 QNSctT(QNT(0, +1), 1),  // 0⊗Up
                 QNSctT(QNT(0, -1), 1)}, // 0⊗Dn
                qlten::TenIndexDirType::IN);
#elif TENSOR_SYMMETRY_LEVEL == 3
  // QNT = (parity, Sz)
  return IndexT({QNSctT(QNT(0, +1), 1),  // D⊗Up
                 QNSctT(QNT(0, -1), 1),  // D⊗Dn
                 QNSctT(QNT(1, +2), 1),  // U⊗Up
                 QNSctT(QNT(1,  0), 1),  // U⊗Dn
                 QNSctT(QNT(1,  0), 1),  // d⊗Up
                 QNSctT(QNT(1, -2), 1),  // d⊗Dn
                 QNSctT(QNT(0, +1), 1),  // 0⊗Up
                 QNSctT(QNT(0, -1), 1)}, // 0⊗Dn
                qlten::TenIndexDirType::IN);
#endif
}

const IndexT loc_hilbert_ket = BuildLocalHilbertKet();
const IndexT loc_hilbert_bra = qlten::InverseIndex(loc_hilbert_ket);

// Default output directory for PEPS dumps.
static const std::string peps_path = "kondo_peps";

#endif  // LNO_PEPS_KONDO_SINGLE_LAYER_QLDOUBLE_H


