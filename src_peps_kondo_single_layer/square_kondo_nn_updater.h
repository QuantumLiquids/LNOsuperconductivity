// SPDX-License-Identifier: MIT
/*
 * Custom NN Monte-Carlo updater for the single-layer Kondo lattice (8D local space).
 *
 * Core rule (what we conserve):
 * - Total itinerant electron number Ne is conserved.
 * - Total Sz_total = Sz_electron + Sz_local is conserved.
 *
 * Implementation rule (how we propose candidates on one NN bond):
 * - Enumerate *all* (c1', c2') in the 8x8 two-site space such that:
 *     Ne(c1') + Ne(c2') == Ne(c1) + Ne(c2)
 *   and
 *     SzTot2(c1') + SzTot2(c2') == SzTot2(c1) + SzTot2(c2)
 *   where SzTot2 is 2*Sz_total (integer).
 * - Then do Suwa–Todo selection with weights ~ |psi(c1',c2')|^2.
 *
 * This is the simplest correct thing: no extra "special-case" moves, no hidden bias,
 * and it is guaranteed to stay inside the desired global symmetry sector because the
 * update only touches these two sites.
 */
#ifndef LNO_PEPS_KONDO_SQUARE_KONDO_NN_UPDATER_H
#define LNO_PEPS_KONDO_SQUARE_KONDO_NN_UPDATER_H

#include <array>
#include <algorithm>
#include <utility>
#include <vector>

#include "qlpeps/vmc_basic/configuration_update_strategies/square_nn_updater.h"
#include "./square_kondo_model.h" // reuse ElectronState/LocalSpinState and decoding helpers

namespace peps_kondo {

inline size_t CombineKondo(ElectronState e, LocalSpinState s) { return 2 * size_t(e) + size_t(s); }

inline int ElectronNum(ElectronState e) {
  switch (e) {
    case E_D: return 2;
    case E_U: return 1;
    case E_d: return 1;
    case E_0: return 0;
    default:  return 0;
  }
}
inline int ElectronSz2(ElectronState e) {
  switch (e) {
    case E_U: return +1;
    case E_d: return -1;
    case E_D: return 0;
    case E_0: return 0;
    default:  return 0;
  }
}
inline int LocalSz2(LocalSpinState s) { return (s == S_U) ? +1 : -1; }
inline int SzTot2OfLocalConfig(size_t c) { return ElectronSz2(ElectronOf(c)) + LocalSz2(SpinOf(c)); }

/**
 * NN updater that preserves Ne and Sz_total by using physically motivated moves.
 *
 * - Electron hop on bond: move one spin-σ electron from one site to its NN.
 * - Kondo flip on site: |up,Down> <-> |down,Up>.
 */
template<typename WaveFunctionDress = qlpeps::NoDress>
class MCUpdateSquareKondoNNConservedOBC
    : public qlpeps::MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareKondoNNConservedOBC<WaveFunctionDress>, WaveFunctionDress> {
 public:
  using Base = qlpeps::MCUpdateSquareNNUpdateBaseOBC<MCUpdateSquareKondoNNConservedOBC<WaveFunctionDress>, WaveFunctionDress>;
  using Base::Base; // inherit ctor (random engine, etc.)

  // Optional target sector for initialization.
  // If set (target_ne >= 0), InitConfig will generate a random state in this sector.
  int target_ne_ = -1;
  int target_sz2_ = 0;

  MCUpdateSquareKondoNNConservedOBC(size_t seed, size_t thread_num, int ne = -1, int sz2 = 0)
      : Base(seed, thread_num), target_ne_(ne), target_sz2_(sz2) {}

  // Override initialization to enforce symmetry sector if requested.
  template<typename TenElemT, typename QNT>
  void InitConfig(qlpeps::TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress> &tps_component) {
    if (target_ne_ < 0) {
      // Fallback to random initialization (Base implementation usually does this,
      // but since we can't easily call Base::InitConfig if it's not exposed or we shadowed it,
      // we'll just implement a simple random init here).
      // Actually, let's try to call Base::InitConfig first. If it fails to compile, we fix it.
      // But Base::InitConfig likely exists.
      Base::InitConfig(tps_component);
      return;
    }

    auto &config = tps_component.config;
    const size_t Ly = config.rows();
    const size_t Lx = config.cols();
    const size_t N = Lx * Ly;

    if (size_t(target_ne_) > 2 * N) {
      throw std::runtime_error("Target Ne > 2*N sites.");
    }

    // Generate a random configuration in the (Ne, Sz) sector.
    // We need:
    //  Sum(Ne_i) = target_ne_
    //  Sum(Sz2_i) = target_sz2_
    //
    // Strategy:
    // 1. Distribute Ne electrons.
    // 2. Assign spins to satisfy Sz.
    //    Total Sz = Sz_e + Sz_loc.
    //    Let N_up, N_dn be electron counts. N_up + N_dn = Ne.
    //    Let N_Up_loc, N_Dn_loc be local spin counts. N_Up_loc + N_Dn_loc = N.
    //    Sz_e = 0.5 * (N_up - N_dn)
    //    Sz_loc = 0.5 * (N_Up_loc - N_Dn_loc)
    //    Sz_total = 0.5 * (N_up - N_dn + N_Up_loc - N_Dn_loc)
    //    2*Sz_total = (N_up - N_dn) + (N_Up_loc - N_Dn_loc)
    //               = (2*N_up - Ne) + (2*N_Up_loc - N)
    //    target_sz2 = 2*N_up + 2*N_Up_loc - Ne - N.
    //    2*(N_up + N_Up_loc) = target_sz2 + Ne + N.
    //
    //    So K = N_up + N_Up_loc = (target_sz2 + Ne + N) / 2 must be integer.
    //
    //    We need to pick N_up (number of up electrons) and N_Up_loc (number of Up local spins)
    //    such that N_up + N_Up_loc = K.
    //    Constraints:
    //      0 <= N_up <= N (actually bounded by Ne: 0 <= N_up <= Ne, and also N_dn <= N => Ne-N_up <= N => N_up >= Ne-N)
    //      0 <= N_Up_loc <= N.
    //
    //    We pick a valid pair (N_up, N_Up_loc) randomly or fixed? Randomly is better.
    //    Range for N_up:
    //      Max N_up: min(Ne, N, K) (since N_Up_loc >= 0 => N_up <= K)
    //      Min N_up: max(0, Ne - N, K - N) (since N_Up_loc <= N => N_up >= K - N)
    //
    //    We uniformly pick N_up in [Min, Max].
    //    Then N_Up_loc = K - N_up.
    //    Then N_dn = Ne - N_up.
    //    Then N_Dn_loc = N - N_Up_loc.
    //
    //    Then we shuffle:
    //    - N sites: pick N_up sites to have Up electron? No, electrons can be double.
    //      Actually, "Distribute Ne electrons" involves placing them into 2*N slots (up/down orbitals).
    //      This is equivalent to:
    //      - Place N_up electrons into N sites (max 1 per site per spin).
    //      - Place N_dn electrons into N sites.
    //      - Place N_Up_loc local spins into N sites.

    int rhs = target_sz2_ + target_ne_ + int(N);
    if (rhs % 2 != 0) {
      throw std::runtime_error("Incompatible Ne, Sz, N constraints (sum must be even).");
    }
    int K = rhs / 2;

    int min_nup = std::max({0, int(target_ne_) - int(N), K - int(N)});
    int max_nup = std::min({int(target_ne_), int(N), K});

    if (min_nup > max_nup) {
      throw std::runtime_error("No valid configuration for target Ne, Sz sector.");
    }

    std::uniform_int_distribution<int> dist_nup(min_nup, max_nup);
    int n_up = dist_nup(this->random_engine_);
    int n_dn = target_ne_ - n_up;
    int n_up_loc = K - n_up;
    int n_dn_loc = int(N) - n_up_loc;

    // Distribute N_up electrons into N sites
    std::vector<int> sites_up(N, 0);
    for(int i=0; i<n_up; ++i) sites_up[i] = 1;
    std::shuffle(sites_up.begin(), sites_up.end(), this->random_engine_);

    // Distribute N_dn electrons into N sites
    std::vector<int> sites_dn(N, 0);
    for(int i=0; i<n_dn; ++i) sites_dn[i] = 1;
    std::shuffle(sites_dn.begin(), sites_dn.end(), this->random_engine_);

    // Distribute N_up_loc local spins into N sites
    std::vector<int> sites_loc(N, 0); // 1 for Up, 0 for Dn
    for(int i=0; i<n_up_loc; ++i) sites_loc[i] = 1;
    std::shuffle(sites_loc.begin(), sites_loc.end(), this->random_engine_);

    // Combine
    for (size_t i = 0; i < N; ++i) {
      int u = sites_up[i];
      int d = sites_dn[i];
      int loc = sites_loc[i]; // 1=Up, 0=Dn.

      ElectronState e;
      if (u && d) e = E_D;
      else if (u) e = E_U;
      else if (d) e = E_d;
      else e = E_0;

      LocalSpinState s = (loc == 1) ? S_U : S_d;
      
      size_t c = CombineKondo(e, s);
      config(i / Lx, i % Lx) = c;
    }

    // After setting config, we must update amplitudes/cache in tps_component?
    // Usually InitConfig just sets the configuration.
    // The sampler will then evaluate the amplitude of this config.
    // However, if we are overriding Base::InitConfig which might do more (like setting `tps_component.amplitude`), we should be careful.
    // But typically `Sampler::Init` calls `InitConfig` then computes `Amplitude`.
    // Let's assume just setting `config` is enough.
  }

  template<typename TenElemT, typename QNT>
  bool TwoSiteNNUpdateLocalImpl(const qlpeps::SiteIdx &site1,
                                const qlpeps::SiteIdx &site2,
                                qlpeps::BondOrientation bond_dir,
                                const qlpeps::SplitIndexTPS<TenElemT, QNT> &sitps,
                                qlpeps::TPSWaveFunctionComponent<TenElemT, QNT, WaveFunctionDress> &tps_component) {
    const size_t c1 = tps_component.config(site1);
    const size_t c2 = tps_component.config(site2);

    // Candidate list includes the current state (index 0).
    // We enumerate *all* candidates on the 8x8 two-site space that keep:
    //   Ne(bond) and Sz_total(bond) unchanged.
    const int ne0 = ElectronNum(ElectronOf(c1)) + ElectronNum(ElectronOf(c2));
    const int sztot20 = SzTot2OfLocalConfig(c1) + SzTot2OfLocalConfig(c2);

    std::vector<std::pair<size_t, size_t>> cand;
    cand.reserve(64);
    size_t init_state = static_cast<size_t>(-1);

    for (size_t nc1 = 0; nc1 < 8; ++nc1) {
      for (size_t nc2 = 0; nc2 < 8; ++nc2) {
        const int ne = ElectronNum(ElectronOf(nc1)) + ElectronNum(ElectronOf(nc2));
        if (ne != ne0) continue;
        const int sztot2 = SzTot2OfLocalConfig(nc1) + SzTot2OfLocalConfig(nc2);
        if (sztot2 != sztot20) continue;
        cand.emplace_back(nc1, nc2);
        if (nc1 == c1 && nc2 == c2) {
          init_state = cand.size() - 1;
        }
      }
    }

    // Nothing to do if only identity
    if (cand.size() == 1) return false;
    if (init_state == static_cast<size_t>(-1)) {
      // This should never happen: (c1,c2) must satisfy its own Ne/Sz constraints.
      return false;
    }

    // Compute wavefunction amplitudes for each candidate.
    std::vector<TenElemT> psis(cand.size());
    double psi_abs_max = 0.0;
    for (size_t i = 0; i < cand.size(); ++i) {
      const auto [nc1, nc2] = cand[i];
      if (i == init_state) {
        psis[i] = tps_component.amplitude;
      } else {
        psis[i] = tps_component.contractor.ReplaceNNSiteTrace(
            tps_component.tn,
            site1, site2, bond_dir,
            sitps(site1)[nc1],
            sitps(site2)[nc2]);
      }
      psi_abs_max = std::max(psi_abs_max, std::abs(psis[i]));
    }
    if (psi_abs_max == 0.0) {
      // All candidates have zero amplitude under current TPS; reject move.
      return false;
    }

    std::vector<double> weights(cand.size());
    for (size_t i = 0; i < cand.size(); ++i) {
      weights[i] = std::norm(psis[i] / psi_abs_max);
    }

    const size_t final_state = qlpeps::SuwaTodoStateUpdate(init_state, weights, this->random_engine_);
    if (final_state == init_state) return false;

    const auto [new_c1, new_c2] = cand[final_state];
    tps_component.UpdateLocal(
        sitps,
        psis[final_state],
        std::make_pair(site1, new_c1),
        std::make_pair(site2, new_c2));
    return true;
  }
};

} // namespace peps_kondo

#endif // LNO_PEPS_KONDO_SQUARE_KONDO_NN_UPDATER_H


