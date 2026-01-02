// SPDX-License-Identifier: MIT
/*
 * Simple Update skeleton for the single-layer Kondo lattice model on a square lattice (OBC).
 *
 * This is intentionally minimal and focused on *conventions*:
 * - local 8D basis encoding (electron x local spin)
 * - fermion parity / (optional) Sz quantum numbers
 * - on-site terms: U, JK * s·S, -mu * n
 * - NN hopping term for itinerant electrons (uniform t for now)
 *
 * Usage:
 *   ./peps_kondo_square_simple_update <physics_params.json> <simple_update_algo.json>
 *
 * Notes:
 * - JK convention matches existing DMRG/VMPS code in this repo:
 *     H_K = JK * (s · S).  FM corresponds to JK < 0.
 * - NN bond anisotropy (t vs t2 bond-order pattern) is not implemented yet in SU executor.
 */

#include <iostream>
#include <vector>

#include "qlpeps/qlpeps.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "qlpeps/api/conversions.h"
#include "qlmps/utilities.h"  // IsPathExist, CreatPath

#include "./common_params.h"
#include "./qldouble.h"

namespace {

enum ElectronState : size_t {
  E_D = 0,  // |up dn>
  E_U = 1,  // |up>
  E_d = 2,  // |dn>
  E_0 = 3   // |0>
};

enum LocalSpinState : size_t {
  S_U = 0,  // |Up>
  S_d = 1   // |Dn>
};

inline size_t Combine(const ElectronState e, const LocalSpinState s) {
  return 2 * size_t(e) + size_t(s);
}

inline ElectronState ElectronOf(const size_t combined) {
  return static_cast<ElectronState>(combined / 2);
}

inline LocalSpinState SpinOf(const size_t combined) {
  return static_cast<LocalSpinState>(combined % 2);
}

// Build a one-site operator tensor in index order (OUT, IN) expected by qlpeps onsite terms.
Tensor MakeOneSiteOpZero() {
  Tensor op(std::vector<IndexT>{loc_hilbert_bra, loc_hilbert_ket});
  return op;
}

Tensor MakeNumberOp() {
  Tensor n = MakeOneSiteOpZero();
  for (size_t a = 0; a < 8; ++a) {
    const auto e = ElectronOf(a);
    const double ne = (e == E_D) ? 2.0 : (e == E_U || e == E_d) ? 1.0 : 0.0;
    n({a, a}) = ne;
  }
  return n;
}

Tensor MakeDoublonOp() {
  Tensor d = MakeOneSiteOpZero();
  for (size_t a = 0; a < 8; ++a) {
    d({a, a}) = (ElectronOf(a) == E_D) ? 1.0 : 0.0;
  }
  return d;
}

Tensor MakeSdotSLocal() {
  // s · S = sz*Sz + 0.5*(s+ S- + s- S+)
  Tensor op = MakeOneSiteOpZero();

  auto sz_e = [](ElectronState e) -> double {
    if (e == E_U) return +0.5;
    if (e == E_d) return -0.5;
    return 0.0;
  };
  auto sz_l = [](LocalSpinState s) -> double { return (s == S_U) ? +0.5 : -0.5; };

  // Diagonal sz*Sz contribution
  for (size_t a = 0; a < 8; ++a) {
    const auto e = ElectronOf(a);
    const auto s = SpinOf(a);
    op({a, a}) = TenElemT(sz_e(e) * sz_l(s));
  }

  // Off-diagonal flip-flop:
  // s+ S- : |dn,Up> -> |up,Dn>
  op({Combine(E_U, S_d), Combine(E_d, S_U)}) = TenElemT(0.5);
  // s- S+ : |up,Dn> -> |dn,Up>
  op({Combine(E_d, S_U), Combine(E_U, S_d)}) = TenElemT(0.5);

  return op;
}

Tensor MakeHoppingBondHam(const double t) {
  // IMPORTANT (fermions):
  // Do NOT construct the hopping term via `Contract(c, c^dag)` here.
  // Upstream PEPS tests show this can be a footgun for fermionic operator algebra.
  //
  // We instead build the two-site hopping Hamiltonian by explicit matrix elements,
  // using a fixed *global mode ordering*:
  //   mode 0: site1 up, mode 1: site1 down, mode 2: site2 up, mode 3: site2 down.
  //
  // This matches the logic in:
  //   /Users/wanghaoxin/GitHub/PEPS/tests/test_algorithm/test_fermion_simple_update.cpp
  //
  // Raw storage convention (i-j-j-i) prior to transpose:
  //   ham(ket1, ket2, bra2, bra1) = <ket1,ket2|H|bra1,bra2>.
  Tensor ham = Tensor({loc_hilbert_ket, loc_hilbert_ket, loc_hilbert_bra, loc_hilbert_bra});

  auto e_to_bits = [&](const ElectronState e) -> std::pair<int, int> {
    switch (e) {
      case E_D: return {1, 1};
      case E_U: return {1, 0};
      case E_d: return {0, 1};
      case E_0: return {0, 0};
      default:  return {0, 0};
    }
  };
  auto bits_to_e = [&](const int nu, const int nd) -> ElectronState {
    if (nu == 1 && nd == 1) return E_D;
    if (nu == 1 && nd == 0) return E_U;
    if (nu == 0 && nd == 1) return E_d;
    return E_0;
  };

  auto popcount_prefix = [&](const std::array<int, 4> &b, const int mode) -> int {
    int cnt = 0;
    for (int i = 0; i < mode; ++i) cnt += b[i];
    return cnt;
  };
  auto apply_annihilate = [&](std::array<int, 4> &b, const int mode, double &sgn) -> bool {
    if (b[mode] == 0) return false;
    if (popcount_prefix(b, mode) & 1) sgn *= -1.0;
    b[mode] = 0;
    return true;
  };
  auto apply_create = [&](std::array<int, 4> &b, const int mode, double &sgn) -> bool {
    if (b[mode] == 1) return false;
    if (popcount_prefix(b, mode) & 1) sgn *= -1.0;
    b[mode] = 1;
    return true;
  };

  // Fill hopping matrix elements:
  // H = -t * Σ_σ ( c1σ^† c2σ + c2σ^† c1σ )
  for (size_t bra1 = 0; bra1 < 8; ++bra1) {
    for (size_t bra2 = 0; bra2 < 8; ++bra2) {
      const auto e1 = ElectronOf(bra1);
      const auto s1 = SpinOf(bra1);
      const auto e2 = ElectronOf(bra2);
      const auto s2 = SpinOf(bra2);
      const auto [n1u, n1d] = e_to_bits(e1);
      const auto [n2u, n2d] = e_to_bits(e2);
      const std::array<int, 4> bra_bits{n1u, n1d, n2u, n2d};

      for (int sigma = 0; sigma < 2; ++sigma) {
        const int mode1 = (sigma == 0) ? 0 : 1;  // site1 up/down
        const int mode2 = (sigma == 0) ? 2 : 3;  // site2 up/down

        // c1^dag c2 : move sigma from site2 -> site1
        {
          std::array<int, 4> b = bra_bits;
          double sgn = 1.0;
          if (apply_annihilate(b, mode2, sgn) && apply_create(b, mode1, sgn)) {
            const ElectronState ke1 = bits_to_e(b[0], b[1]);
            const ElectronState ke2 = bits_to_e(b[2], b[3]);
            const size_t ket1 = Combine(ke1, s1);
            const size_t ket2 = Combine(ke2, s2);
            ham({ket1, ket2, bra2, bra1}) = ham({ket1, ket2, bra2, bra1}) + TenElemT((-t) * sgn);
          }
        }
        // c2^dag c1 : move sigma from site1 -> site2
        {
          std::array<int, 4> b = bra_bits;
          double sgn = 1.0;
          if (apply_annihilate(b, mode1, sgn) && apply_create(b, mode2, sgn)) {
            const ElectronState ke1 = bits_to_e(b[0], b[1]);
            const ElectronState ke2 = bits_to_e(b[2], b[3]);
            const size_t ket1 = Combine(ke1, s1);
            const size_t ket2 = Combine(ke2, s2);
            ham({ket1, ket2, bra2, bra1}) = ham({ket1, ket2, bra2, bra1}) + TenElemT((-t) * sgn);
          }
        }
      }
    }
  }

  // Match qlpeps simple update convention (same permutation used in upstream tests).
  ham.Transpose({3, 0, 2, 1});
  return ham;
}

Tensor MakeOnsiteHam(const double U, const double JK, const double mu) {
  // H_onsite = U * n_up n_dn + JK * (s·S) - mu * n
  Tensor ham = MakeOneSiteOpZero();
  ham += MakeDoublonOp() * U;
  ham += MakeSdotSLocal() * JK;
  ham += MakeNumberOp() * (-mu);
  return ham;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " <physics_params.json> <simple_update_algo.json>\n";
    return 1;
  }

  peps_kondo_params::Params params(argv[1], argv[2]);
  const size_t Lx = params.physical.Lx;
  const size_t Ly = params.physical.Ly;

  const double t = params.physical.t;
  const double t2 = params.physical.t2;
  const double U = params.physical.U;
  const double JK = params.physical.JK;
  const double mu = params.physical.mu;
  const size_t Ne = params.physical.ElectronNum;
  const int Sz2e = params.physical.ElectronSz2;

  if (Lx == 0 || Ly == 0) {
    std::cerr << "Invalid lattice size.\n";
    return 1;
  }
  try {
    peps_kondo_params::EnforceRestrictedSectorOrDie(Lx, Ly, Ne, "simple_update");
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  if (t2 != 0.0 && t2 != t) {
    std::cerr << "NN bond anisotropy (t vs t2) is not implemented in SU executor yet.\n"
              << "For now, use t2=0 or t2=t.\n";
    return 1;
  }

  // Thread control: keep it simple and portable (do not depend on qlten hp_numeric here).
#ifdef _OPENMP
  if (params.algo.ThreadNum > 0) {
    omp_set_num_threads(static_cast<int>(params.algo.ThreadNum));
  }
#endif

  const Tensor ham_nn = MakeHoppingBondHam(t);
  const Tensor ham_onsite = MakeOnsiteHam(U, JK, mu);

  qlpeps::SimpleUpdatePara su_para(params.algo.Step, params.algo.Tau,
                                  params.algo.Dmin, params.algo.Dmax,
                                  params.algo.TruncErr);

  using PEPST = qlpeps::SquareLatticePEPS<TenElemT, QNT>;
  PEPST peps0(loc_hilbert_ket, Ly, Lx);
  std::optional<std::vector<std::vector<size_t>>> init_activates;

  if (qlmps::IsPathExist(peps_path)) {
    std::cout << "Continue by loading PEPS from directory: " << peps_path << "\n";
    peps0.Load(peps_path);
  } else {
    std::cout << "Initialize PEPS as a direct product state.\n";
    // IMPORTANT:
    // The Hamiltonian implemented in this SU driver conserves electron number.
    // So we must initialize the desired electron filling sector here.
    const size_t N = Lx * Ly;
    size_t Ne_clamped = Ne;
    if (Ne_clamped > 2 * N) Ne_clamped = 2 * N;
    long Ne_single = static_cast<long>(Ne_clamped); // we do not place doublons in the initializer
    long Nup = (Ne_single + static_cast<long>(Sz2e)) / 2;
    long Ndn = Ne_single - Nup;
    if (Nup < 0) Nup = 0;
    if (Ndn < 0) Ndn = 0;
    if (static_cast<size_t>(Nup + Ndn) > N) { // fallback: cap to one electron per site
      Nup = static_cast<long>(N / 2);
      Ndn = static_cast<long>(N / 2);
    }

    std::vector<size_t> e_labels;
    e_labels.reserve(N);
    for (size_t i = 0; i < static_cast<size_t>(Nup); ++i) e_labels.push_back(E_U);
    for (size_t i = 0; i < static_cast<size_t>(Ndn); ++i) e_labels.push_back(E_d);
    while (e_labels.size() < N) e_labels.push_back(E_0);
    e_labels.resize(N);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(e_labels.begin(), e_labels.end(), gen);

    // Enforce total Sz_total = 0 by choosing local spins such that:
    //   Sz2_total = Sz2e + Sz2_local = 0
    //   Sz2_local = 2*Nup_local - N
    // => Nup_local = (N - Sz2e) / 2
    if ((static_cast<long>(N) - static_cast<long>(Sz2e)) % 2 != 0) {
      std::cerr << "Initializer error: cannot satisfy Sz_total=0 due to parity mismatch "
                   "(need (N - Sz2e) even).\n";
      return 1;
    }
    const long Nup_local = (static_cast<long>(N) - static_cast<long>(Sz2e)) / 2;
    if (Nup_local < 0 || Nup_local > static_cast<long>(N)) {
      std::cerr << "Initializer error: cannot satisfy Sz_total=0 (invalid local spin magnetization).\n";
      return 1;
    }
    std::vector<LocalSpinState> s_labels;
    s_labels.reserve(N);
    if (Sz2e == 0) {
      // Prefer a clean Neel pattern when compatible with Sz_total=0.
      for (size_t y = 0; y < Ly; ++y) {
        for (size_t x = 0; x < Lx; ++x) {
          s_labels.push_back(((x + y) % 2 == 0) ? S_U : S_d);
        }
      }
    } else {
      for (size_t i = 0; i < static_cast<size_t>(Nup_local); ++i) s_labels.push_back(S_U);
      while (s_labels.size() < N) s_labels.push_back(S_d);
      std::shuffle(s_labels.begin(), s_labels.end(), gen);
    }

    std::vector<std::vector<size_t>> activates(Ly, std::vector<size_t>(Lx, 0));
    size_t idx = 0;
    for (size_t y = 0; y < Ly; ++y) {
      for (size_t x = 0; x < Lx; ++x) {
        const auto e = static_cast<ElectronState>(e_labels[idx]);
        const auto s = s_labels[idx];
        activates[y][x] = Combine(e, s);
        ++idx;
      }
    }
    peps0.Initial(activates);
    init_activates = activates; // dump after we dump tpsfinal/
  }

  qlpeps::SquareLatticeNNSimpleUpdateExecutor<TenElemT, QNT> exe(su_para, peps0, ham_nn, ham_onsite);
  std::cout << "[Kondo-PEPS-SU] Lx=" << Lx << " Ly=" << Ly
            << " t=" << t << " U=" << U << " JK=" << JK << " mu=" << mu
            << " Ne=" << Ne << " Sz2e=" << Sz2e
            << " steps=" << su_para.steps << " tau=" << su_para.tau
            << " Dmax=" << su_para.Dmax
            << "\n";
  exe.Execute();
  // Dump PEPS tensors
  exe.DumpResult(peps_path, /*release_mem=*/false);

  // Also dump SplitIndexTPS for VMC/measurement (convention: "tpsfinal/")
  // This matches the workflow used in finite-size_PEPS_tJ.
  auto tps = qlpeps::ToTPS<TenElemT, QNT>(exe.GetPEPS());
  for (auto &tensor : tps) {
    auto max_abs = tensor.GetMaxAbs();
    if (max_abs != 0.0) {
      tensor *= 1.0 / max_abs;
    }
  }
  auto sitps = qlpeps::ToSplitIndexTPS<TenElemT, QNT>(tps);
  sitps.Dump("tpsfinal");

  // Dump a compatible initial configuration into tpsfinal/ *after* sitps dump.
  // Some wavefunction dump paths recreate the directory, so dumping before sitps may be lost.
  if (init_activates.has_value()) {
    qlpeps::Configuration init_cfg(*init_activates);
    init_cfg.Dump("tpsfinal", 0);
  }
  return 0;
}


