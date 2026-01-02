// SPDX-License-Identifier: MIT
/*
 * Square-lattice single-layer Kondo lattice model (PEPS VMC/measurement solver).
 *
 * Local Hilbert space is 8D = (Hubbard electron 4 states) x (local spin 1/2 2 states).
 * The basis encoding MUST match `src_peps_kondo_single_layer/qldouble.h`.
 *
 * Hamiltonian pieces implemented here:
 * - NN hopping for itinerant electrons:  H_t = -t Σ_<i,j>,σ (c†_{iσ} c_{jσ} + h.c.)
 * - On-site: U * n_up n_dn  - mu * n
 * - On-site Kondo: JK * s·S = JK*(sz*Sz + 1/2(s+S- + s-S+))
 *
 * Notes:
 * - JK convention: H_K = JK * (s · S). FM corresponds to JK < 0.
 * - We implement the off-diagonal part of s·S in the measurement layer (energy key) by
 *   explicit ReplaceOneSiteTrace ratios, because SquareNNNModelEnergySolver's on-site hook
 *   is diagonal-only by design.
 */
#ifndef LNO_PEPS_KONDO_SQUARE_KONDO_MODEL_H
#define LNO_PEPS_KONDO_SQUARE_KONDO_MODEL_H

#include <optional>
#include <utility>
#include <vector>

#include "qlpeps/algorithm/vmc_update/model_energy_solver.h"  // ModelEnergySolver base (CRTP)
#include "qlpeps/algorithm/vmc_update/model_solvers/base/square_nn_model_measurement_solver.h"
#include "qlpeps/algorithm/vmc_update/model_solvers/base/bond_traversal_mixin.h"  // BondTraversalMixin
#include "qlpeps/two_dim_tn/tensor_network_2d/bmps_contractor.h"
#include "qlpeps/utility/helpers.h" // ComplexConjugate

namespace peps_kondo {

// Basis decoding consistent with qldouble.h:
// electron: 0=|up dn>, 1=|up>, 2=|dn>, 3=|0>
// spin    : 0=|Up>,    1=|Dn>
enum ElectronState : size_t { E_D = 0, E_U = 1, E_d = 2, E_0 = 3 };
enum LocalSpinState : size_t { S_U = 0, S_d = 1 };

inline ElectronState ElectronOf(const size_t combined) { return static_cast<ElectronState>(combined / 2); }
inline LocalSpinState SpinOf(const size_t combined) { return static_cast<LocalSpinState>(combined % 2); }

inline double ElectronDensity(const size_t cfg) {
  switch (ElectronOf(cfg)) {
    case E_D: return 2.0;
    case E_U: return 1.0;
    case E_d: return 1.0;
    case E_0: return 0.0;
    default:  return 0.0;
  }
}

inline double ElectronSz(const size_t cfg) {
  switch (ElectronOf(cfg)) {
    case E_U: return +0.5;
    case E_d: return -0.5;
    default:  return 0.0;
  }
}

inline double LocalSz(const size_t cfg) {
  return (SpinOf(cfg) == S_U) ? +0.5 : -0.5;
}

inline double KondoDiagSzSz(const size_t cfg) {
  return ElectronSz(cfg) * LocalSz(cfg);
}

/**
 * SquareKondoModel:
 * - Implements NN hopping bond energy (fermionic off-diagonal bond term).
 * - Provides diagonal on-site energy: U doublon - mu n + JK sz Sz.
 * - Extends measurement registry with electron/local spin resolved observables and
 *   adds the on-site JK flip-flop contribution into the "energy" observable.
 */
class SquareKondoModel : public qlpeps::ModelEnergySolver<SquareKondoModel>,
                         public qlpeps::SquareNNModelMeasurementSolver<SquareKondoModel> {
 public:
  // Import base class CalEnergyAndHoles (CRTP now targets SquareKondoModel directly)
  using qlpeps::ModelEnergySolver<SquareKondoModel>::CalEnergyAndHoles;
  static constexpr bool requires_density_measurement = true;   // charge
  static constexpr bool requires_spin_sz_measurement = true;   // we report total Sz as "spin_z"

  SquareKondoModel() = delete;
  SquareKondoModel(double t, double U, double JK, double mu) : t_(t), U_(U), JK_(JK), mu_(mu) {}

  // =====================================================================
  // CalEnergyAndHolesImpl: Full energy calculation including:
  // 1) NN hopping (fermionic off-diagonal bond term)
  // 2) Diagonal on-site: U doublon - mu n + JK sz*Sz
  // 3) Off-diagonal Kondo flip-flop: JK/2 (s+S- + s-S+)
  //
  // CRITICAL: We now inherit directly from ModelEnergySolver<SquareKondoModel>,
  // so CRTP correctly calls this method rather than a base class version.
  // =====================================================================
  template<typename TenElemT, typename QNT, bool calchols = true>
  TenElemT CalEnergyAndHolesImpl(
      const qlpeps::SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      qlpeps::TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      qlpeps::TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &psi_list
  ) {
    std::vector<TenElemT> bond_energy_set;
    bond_energy_set.reserve(2 * split_index_tps->size());
    
    // 1. Calculate horizontal bond energies (hopping) and holes
    CalHorizontalBondEnergyAndHoles<TenElemT, QNT, calchols>(
        split_index_tps, tps_sample, hole_res, bond_energy_set, psi_list);
    
    // 2. Calculate vertical bond energies (hopping)
    CalVerticalBondEnergy<TenElemT, QNT>(
        split_index_tps, tps_sample, bond_energy_set, psi_list);
    
    TenElemT bond_energy_total = std::reduce(bond_energy_set.begin(), bond_energy_set.end());
    
    // 3. Calculate diagonal on-site energy
    TenElemT energy_onsite = EvaluateTotalOnsiteEnergy(tps_sample->config);
    
    // 4. Calculate Kondo flip-flop energy
    TenElemT e_flip = EvaluateKondoFlipFlopEnergy<TenElemT, QNT>(split_index_tps, tps_sample);
    
    return bond_energy_total + energy_onsite + e_flip;
  }
  
  // =====================================================================
  // Helper: Calculate horizontal bond energies (hopping)
  // =====================================================================
  template<typename TenElemT, typename QNT, bool calchols>
  void CalHorizontalBondEnergyAndHoles(
      const qlpeps::SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      qlpeps::TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      qlpeps::TensorNetwork2D<TenElemT, QNT> &hole_res,
      std::vector<TenElemT> &bond_energy_set,
      std::vector<TenElemT> &psi_list
  ) {
    using RealT = typename qlten::RealTypeTrait<TenElemT>::type;
    auto &tn = tps_sample->tn;
    auto &contractor = tps_sample->contractor;
    const auto &config = tps_sample->config;
    const qlpeps::BMPSTruncateParams<RealT> &trunc_para = tps_sample->trun_para;
    
    contractor.GenerateBMPSApproach(tn, qlpeps::UP, trunc_para);
    psi_list.reserve(tn.rows() + tn.cols());
    
    for (size_t row = 0; row < tn.rows(); ++row) {
      contractor.InitBTen(tn, qlpeps::LEFT, row);
      contractor.GrowFullBTen(tn, qlpeps::RIGHT, row, 1, true);
      
      bool psi_added = false;
      for (size_t col = 0; col < tn.cols(); ++col) {
        const qlpeps::SiteIdx site1 = {row, col};
        
        // Calculate holes if needed
        if constexpr (calchols) {
          hole_res(site1) = Dag(contractor.PunchHole(tn, site1, qlpeps::HORIZONTAL));
        }
        
        if (col < tn.cols() - 1) {
          const qlpeps::SiteIdx site2 = {row, col + 1};
          std::optional<TenElemT> psi;
          TenElemT bond_energy = EvaluateBondEnergy(
              site1, site2, config(site1), config(site2),
              qlpeps::HORIZONTAL, tn, contractor,
              (*split_index_tps)(site1), (*split_index_tps)(site2), psi);
          bond_energy_set.push_back(bond_energy);
          
          if (!psi_added && psi.has_value()) {
            psi_list.push_back(psi.value());
            psi_added = true;
          }
          contractor.BTenMoveStep(tn, qlpeps::RIGHT);
        }
      }
      
      if (row < tn.rows() - 1) {
        contractor.BMPSMoveStep(tn, qlpeps::DOWN, trunc_para);
      }
    }
  }
  
  // =====================================================================
  // Helper: Calculate vertical bond energies (hopping)
  // =====================================================================
  template<typename TenElemT, typename QNT>
  void CalVerticalBondEnergy(
      const qlpeps::SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      qlpeps::TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample,
      std::vector<TenElemT> &bond_energy_set,
      std::vector<TenElemT> &psi_list
  ) {
    const auto &config = tps_sample->config;
    qlpeps::BondTraversalMixin::TraverseVerticalBonds(
        tps_sample->tn,
        tps_sample->contractor,
        tps_sample->trun_para,
        [&, split_index_tps](const qlpeps::SiteIdx &site1,
                             const qlpeps::SiteIdx &site2,
                             const qlpeps::BondOrientation bond_orient,
                             const TenElemT &inv_psi) {
          std::optional<TenElemT> fermion_psi;
          TenElemT bond_energy = EvaluateBondEnergy(
              site1, site2, config(site1), config(site2),
              bond_orient, tps_sample->tn, tps_sample->contractor,
              (*split_index_tps)(site1), (*split_index_tps)(site2), fermion_psi);
          bond_energy_set.push_back(bond_energy);
        },
        psi_list
    );
  }

  // Compute the Kondo flip-flop energy: JK/2 * (s+S- + s-S+)
  // CRITICAL: For fermionic systems, psi must be recalculated at EACH LOCAL SITE
  // using contractor.Trace(), following the same pattern as EvaluateBondEnergy
  // in Hubbard/t-J models. This avoids sign issues with ReplaceOneSiteTrace ratios.
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateKondoFlipFlopEnergy(
      const qlpeps::SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      qlpeps::TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample
  ) const {
    if (JK_ == 0.0) {
      return TenElemT(0);
    }

    auto &tn = tps_sample->tn;
    auto &contractor = tps_sample->contractor;
    const auto &config = tps_sample->config;
    const size_t ly = config.rows();
    const size_t lx = config.cols();
    const auto &trunc_para = tps_sample->trun_para;

    TenElemT e_flip = TenElemT(0);

    contractor.GenerateBMPSApproach(tn, qlpeps::UP, trunc_para);
    for (size_t row = 0; row < ly; ++row) {
      contractor.InitBTen(tn, qlpeps::LEFT, row);
      contractor.GrowFullBTen(tn, qlpeps::RIGHT, row, 1, true);

      for (size_t col = 0; col < lx; ++col) {
        const qlpeps::SiteIdx site{row, col};
        const size_t c = config(site);
        size_t c2 = c;
        TenElemT coeff = TenElemT(0);

        // Kondo flip-flop: |up,Dn> (idx=3) <-> |dn,Up> (idx=4)
        if (c == 3) {  // E_U=1, S_d=1 -> idx = 2*1+1 = 3
          c2 = 4;      // E_d=2, S_U=0 -> idx = 2*2+0 = 4
          coeff = TenElemT(JK_ * 0.5);
        } else if (c == 4) {
          c2 = 3;
          coeff = TenElemT(JK_ * 0.5);
        }

        if (coeff != TenElemT(0)) {
          // For single-site operations, use ReplaceOneSiteTrace for both psi and psi_ex.
          // This is safer than Trace(tn, site, orient) which requires a valid neighbor site.
          // psi = <bra|current_tensor> at this site
          // psi_ex = <bra|flipped_tensor> at this site
          TenElemT psi = contractor.ReplaceOneSiteTrace(
              tn, site, (*split_index_tps)(site)[c], qlpeps::HORIZONTAL);
          if (psi == TenElemT(0)) [[unlikely]] {
            // Skip if amplitude is zero at this site
            if (col + 1 < lx) {
              contractor.BTenMoveStep(tn, qlpeps::RIGHT);
            }
            continue;
          }
          TenElemT psi_ex = contractor.ReplaceOneSiteTrace(
              tn, site, (*split_index_tps)(site)[c2], qlpeps::HORIZONTAL);
          // Use local psi, calculated at this exact site
          TenElemT ratio = qlpeps::ComplexConjugate(psi_ex / psi);
          e_flip += coeff * ratio;
        }

        if (col + 1 < lx) {
          contractor.BTenMoveStep(tn, qlpeps::RIGHT);
        }
      }
      if (row + 1 < ly) {
        contractor.BMPSMoveStep(tn, qlpeps::DOWN, trunc_para);
      }
    }

    return e_flip;
  }

  // Base "charge" observable (itinerant electron density)
  double CalDensityImpl(const size_t config) const { return ElectronDensity(config); }

  // Base "spin_z" observable (total Sz = electron + local)
  double CalSpinSzImpl(const size_t config) const { return ElectronSz(config) + LocalSz(config); }

  // Diagonal on-site contributions only
  double EvaluateTotalOnsiteEnergy(const qlpeps::Configuration &config) const {
    double e = 0.0;
    for (auto &c : config) {
      const auto epart = ElectronOf(c);
      if (epart == E_D) e += U_;
      e += (-mu_) * ElectronDensity(c);
      e += JK_ * KondoDiagSzSz(c);
    }
    return e;
  }

  // Fermionic NN hopping bond energy
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateBondEnergy(
      const qlpeps::SiteIdx site1, const qlpeps::SiteIdx site2,
      const size_t config1, const size_t config2,
      const qlpeps::BondOrientation orient,
      const qlpeps::TensorNetwork2D<TenElemT, QNT> &tn,
      qlpeps::BMPSContractor<TenElemT, QNT> &contractor,
      const std::vector<qlten::QLTensor<TenElemT, QNT>> &split_index_tps_on_site1,
      const std::vector<qlten::QLTensor<TenElemT, QNT>> &split_index_tps_on_site2,
      std::optional<TenElemT> &psi
  ) const {
    // Hopping does not act on local spins; if both sites have identical electron occupation,
    // it may still have doublon/empty hops? We'll handle generally by enumerating allowed hops.
    psi = contractor.Trace(tn, site1, site2, orient);
    if (psi.value() == TenElemT(0)) {
      return TenElemT(0);
    }

    const auto s1 = SpinOf(config1);
    const auto s2 = SpinOf(config2);
    const auto e1 = ElectronOf(config1);
    const auto e2 = ElectronOf(config2);

    auto e_to_bits = [&](ElectronState e) -> std::pair<int, int> {
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

    const auto [n1u, n1d] = e_to_bits(e1);
    const auto [n2u, n2d] = e_to_bits(e2);
    const std::array<int, 4> bra_bits{n1u, n1d, n2u, n2d};

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

    TenElemT e_bond = TenElemT(0);
    // Enumerate the two spin species (up/down)
    for (int sigma = 0; sigma < 2; ++sigma) {
      const int mode1 = (sigma == 0) ? 0 : 1;
      const int mode2 = (sigma == 0) ? 2 : 3;

      // c1^dag c2: site2 -> site1
      {
        std::array<int, 4> b = bra_bits;
        double sgn = 1.0;
        if (apply_annihilate(b, mode2, sgn) && apply_create(b, mode1, sgn)) {
          const ElectronState ke1 = bits_to_e(b[0], b[1]);
          const ElectronState ke2 = bits_to_e(b[2], b[3]);
          const size_t ket1 = 2 * size_t(ke1) + size_t(s1);
          const size_t ket2 = 2 * size_t(ke2) + size_t(s2);
          TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                                         split_index_tps_on_site1[ket1],
                                                         split_index_tps_on_site2[ket2]);
          TenElemT ratio = qlpeps::ComplexConjugate(psi_ex / psi.value());
          e_bond += TenElemT((-t_) * sgn) * ratio;
        }
      }
      // c2^dag c1: site1 -> site2
      {
        std::array<int, 4> b = bra_bits;
        double sgn = 1.0;
        if (apply_annihilate(b, mode1, sgn) && apply_create(b, mode2, sgn)) {
          const ElectronState ke1 = bits_to_e(b[0], b[1]);
          const ElectronState ke2 = bits_to_e(b[2], b[3]);
          const size_t ket1 = 2 * size_t(ke1) + size_t(s1);
          const size_t ket2 = 2 * size_t(ke2) + size_t(s2);
          TenElemT psi_ex = contractor.ReplaceNNSiteTrace(tn, site1, site2, orient,
                                                         split_index_tps_on_site1[ket1],
                                                         split_index_tps_on_site2[ket2]);
          TenElemT ratio = qlpeps::ComplexConjugate(psi_ex / psi.value());
          e_bond += TenElemT((-t_) * sgn) * ratio;
        }
      }
    }
    return e_bond;
  }

  // NNN energy is disabled for this model, but the base measurement template still requires
  // the method to exist (it is referenced in a compiled lambda). Always return 0.
  template<typename TenElemT, typename QNT>
  TenElemT EvaluateNNNEnergy(
      const qlpeps::SiteIdx /*site1*/, const qlpeps::SiteIdx /*site2*/,
      const size_t /*config1*/, const size_t /*config2*/,
      const qlpeps::DIAGONAL_DIR /*dir*/,
      const qlpeps::TensorNetwork2D<TenElemT, QNT> & /*tn*/,
      qlpeps::BMPSContractor<TenElemT, QNT> & /*contractor*/,
      const std::vector<qlten::QLTensor<TenElemT, QNT>> & /*split_index_tps_on_site1*/,
      const std::vector<qlten::QLTensor<TenElemT, QNT>> & /*split_index_tps_on_site2*/,
      std::optional<TenElemT> & /*psi*/
  ) const {
    return TenElemT(0);
  }

  // Extend registry metadata (extra per-site observables)
  std::vector<qlpeps::ObservableMeta> DescribeObservables(size_t ly, size_t lx) const {
    auto base = qlpeps::SquareNNModelMeasurementSolver<SquareKondoModel>::DescribeObservables(ly, lx);
    // Replace shapes for base keys are already set.
    base.push_back({"spin_z_e", "Itinerant electron Sz per site (Ly,Lx)", {ly, lx}, {"y", "x"}});
    base.push_back({"spin_z_loc", "Localized spin Sz per site (Ly,Lx)", {ly, lx}, {"y", "x"}});
    base.push_back({"kondo_szSz", "On-site sz*Sz per site (Ly,Lx)", {ly, lx}, {"y", "x"}});
    return base;
  }

  // Extend registry evaluation:
  // - Add spin-resolved observables
  // - Add the JK flip-flop contribution into "energy" (off-diagonal onsite term)
  template<typename TenElemT, typename QNT>
  qlpeps::ObservableMap<TenElemT> EvaluateObservables(
      const qlpeps::SplitIndexTPS<TenElemT, QNT> *split_index_tps,
      qlpeps::TPSWaveFunctionComponent<TenElemT, QNT> *tps_sample) {
    // Start from base (computes bond energies + diagonal onsite + caches psi summary).
    auto out = qlpeps::SquareNNModelMeasurementSolver<SquareKondoModel>::template EvaluateObservables<TenElemT, QNT>(
        split_index_tps, tps_sample);

    const auto &config = tps_sample->config;
    const size_t ly = config.rows();
    const size_t lx = config.cols();

    // Extra site-local arrays
    std::vector<TenElemT> se; se.reserve(config.size());
    std::vector<TenElemT> sl; sl.reserve(config.size());
    std::vector<TenElemT> szsz; szsz.reserve(config.size());
    for (auto &c : config) {
      se.push_back(static_cast<TenElemT>(ElectronSz(c)));
      sl.push_back(static_cast<TenElemT>(LocalSz(c)));
      szsz.push_back(static_cast<TenElemT>(KondoDiagSzSz(c)));
    }
    out["spin_z_e"] = std::move(se);
    out["spin_z_loc"] = std::move(sl);
    out["kondo_szSz"] = std::move(szsz);

    // Add off-diagonal onsite Kondo energy: JK/2 (s+S- + s-S+)
    // Reuse the same function used in CalEnergyAndHolesImpl for consistency.
    if (JK_ != 0.0) {
      TenElemT e_flip = EvaluateKondoFlipFlopEnergy<TenElemT, QNT>(split_index_tps, tps_sample);
      // out["energy"] is a scalar stored as length-1 vector
      if (out.find("energy") != out.end() && !out["energy"].empty()) {
        out["energy"][0] += e_flip;
      }
    }
    return out;
  }

 private:
  double t_;
  double U_;
  double JK_;
  double mu_;
};

}  // namespace peps_kondo

#endif  // LNO_PEPS_KONDO_SQUARE_KONDO_MODEL_H


