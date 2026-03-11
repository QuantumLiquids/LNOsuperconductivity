/*
 * Standalone measurement executable for the two-layer zigzag Kondo model.
 *
 * Loads a pre-existing MPS from disk and runs all correlation measurements
 * WITHOUT performing any VMPS sweeps.  Single-process (no MPI required).
 *
 * Usage:
 *   ./kondo_two_layer_zigzag_measure params.json [D]
 *
 * Arguments:
 *   params.json  -- same parameter file used by the VMPS program
 *   D            -- (optional) bond dimension label for output filenames;
 *                   defaults to the last entry in params.Dmax
 *
 * The MPS must already exist at the standard kMpsPath location.
 */

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "./params_case.h"
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"
#include "../src_tj_double_layer_single_orbital_2d/my_measure.h"
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace qlmps;
using namespace qlten;
using namespace std;

namespace {

// ---------------------------------------------------------------------------
// Index helpers (matching vmps.cpp)
// ---------------------------------------------------------------------------

inline size_t ElectronIndex(const size_t geom_site, const size_t layer) {
  return 4 * geom_site + 2 * layer;
}

inline size_t LocalizedIndex(const size_t geom_site, const size_t layer) {
  return ElectronIndex(geom_site, layer) + 1;
}

inline size_t LayerFromGlobalIndex(const size_t global_site) {
  return (global_site % 4) / 2;
}

inline void DeallocAllSites(qlmps::FiniteMPS<TenElemT, QNT> &mps) {
  for (size_t i = 0; i < mps.size(); ++i) {
    mps.dealloc(i);
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " params.json [D]\n";
    return 1;
  }

  CaseParams params(argv[1]);
  const size_t Lx = params.Lx;
  const size_t Ly = params.Ly;
  const double t = params.t;
  const double t2 = params.t2;
  const double Jk = params.JK;
  const double Jperp = params.Jperp;
  const double U = params.U;

  // Bond dimension label for output filenames
  size_t bond_dim = params.Dmax.back();
  if (argc > 2) {
    bond_dim = static_cast<size_t>(std::stoul(argv[2]));
  }

  // --skip-load: skip Load/Centralize/Dump when MPS is already centralized
  // at site 0 on disk (e.g., after a completed VMPS run). Saves memory for
  // large-D MPS that cannot fit entirely in RAM.
  bool skip_load = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--skip-load") {
      skip_load = true;
      break;
    }
  }

  // N = 4 * Ly * Lx: 2 layers x 2 orbitals (itinerant + localized)
  const size_t num_geom_sites = Ly * Lx;
  const size_t N = 4 * num_geom_sites;

  cout << "Two-layer zigzag Kondo lattice -- measurement only\n";
  cout << "Lx = " << Lx << "\n";
  cout << "Ly = " << Ly << "\n";
  cout << "N  = " << N << "\n";
  cout << "t  = " << t << "\n";
  cout << "t2 = " << t2 << "\n";
  cout << "Jk = " << Jk << "\n";
  cout << "U  = " << U << "\n";
  cout << "Jperp = " << Jperp << "\n";
  cout << "Geometry = " << params.Geometry << "\n";
  cout << "D (label) = " << bond_dim << "\n";

  // -----------------------------------------------------------------------
  // Site index helpers (coordinate-based, for per-layer site lists)
  // -----------------------------------------------------------------------
  auto elec_site = [&](size_t x, size_t y, size_t layer) -> size_t {
    return 4 * (y + Ly * x) + 2 * layer;
  };
  auto loc_site = [&](size_t x, size_t y, size_t layer) -> size_t {
    return elec_site(x, y, layer) + 1;
  };

  // -----------------------------------------------------------------------
  // Physical basis
  // -----------------------------------------------------------------------
  std::vector<IndexT> pb_set(N);
  for (size_t i = 0; i < N; ++i) {
    pb_set[i] = (i % 2 == 0) ? pb_outE : pb_outL;
  }
  const SiteVec<TenElemT, QNT> sites(pb_set);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  auto &ops = hubbard_ops;
  SpinOneHalfOperatorsU1U1 local_spin_ops;

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
#endif

  // -----------------------------------------------------------------------
  // Load MPS
  // -----------------------------------------------------------------------
  const std::string mps_path = kMpsPath;

  if (!IsPathExist(mps_path)) {
    std::cerr << "ERROR: MPS not found at " << mps_path << "\n";
    return 1;
  }
  if (N != GetNumofMps()) {
    std::cerr << "ERROR: MPS site count mismatch (expected " << N
              << ", found " << GetNumofMps() << ")\n";
    return 1;
  }

  if (skip_load) {
    cout << "Skipping MPS Load/Centralize (--skip-load). "
         << "MPS must already be centralized at site 0 on disk." << endl;
  } else {
    cout << "Loading MPS from " << mps_path << " ..." << endl;
    mps.Load(mps_path);
    cout << "MPS loaded successfully." << endl;

    // Centralize to site 0 for measurement.
    // Dump back to disk because MeasureTwoSiteOpGroup reads from disk
    // and expects the MPS to be centralized at site 0.
    mps.Centralize(0);
    cout << "MPS centralized to site 0." << endl;
    mps.Dump(mps_path, true);
    cout << "MPS dumped back to disk." << endl;

    // Print entanglement entropy
    auto ee_list = mps.GetEntanglementEntropy(1);
    std::copy(ee_list.begin(), ee_list.end(),
              std::ostream_iterator<double>(std::cout, " "));
    cout << "\nmiddle EE = " << ee_list[N / 2] << endl;
  }

  // -----------------------------------------------------------------------
  // Reference sites and target lists
  // -----------------------------------------------------------------------
  const size_t s_ref = Ly * (Lx / 4);
  const size_t ref_elec = ElectronIndex(s_ref, 0);   // layer-0 itinerant
  const size_t ref_loc  = ref_elec + 1;               // layer-0 localized

  // Per-layer localized reference sites (for l0/l1/l01 measurements)
  const size_t ref_x = s_ref / Ly;
  const size_t ref_y = s_ref % Ly;
  const size_t ref_loc0 = loc_site(ref_x, ref_y, 0);
  const size_t ref_loc1 = loc_site(ref_x, ref_y, 1);

  // -----------------------------------------------------------------------
  // Precompute site lists for measurements
  // -----------------------------------------------------------------------
  // All even (itinerant) and odd (localized) sites
  std::vector<size_t> even_sites;
  even_sites.reserve(N / 2);
  for (size_t i = 0; i < N; i += 2) even_sites.push_back(i);

  std::vector<size_t> odd_sites;
  odd_sites.reserve(N / 2);
  for (size_t i = 1; i < N; i += 2) odd_sites.push_back(i);

  // Per-layer localized site lists
  std::vector<size_t> all_loc0_sites, all_loc1_sites;
  all_loc0_sites.reserve(num_geom_sites);
  all_loc1_sites.reserve(num_geom_sites);
  for (size_t x = 0; x < Lx; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      all_loc0_sites.push_back(loc_site(x, y, 0));
      all_loc1_sites.push_back(loc_site(x, y, 1));
    }
  }

  // Electron two-site targets: split into all, intralayer, interlayer
  std::vector<size_t> elec_targets_all;
  std::vector<size_t> elec_targets_intralayer;
  std::vector<size_t> elec_targets_interlayer;
  for (size_t i = ref_elec + 2; i < N; i += 2) {
    elec_targets_all.push_back(i);
    const size_t layer = LayerFromGlobalIndex(i);
    if (layer == 0) elec_targets_intralayer.push_back(i);
    if (layer == 1) elec_targets_interlayer.push_back(i);
  }

  // Localized two-site targets: split into all, intralayer, interlayer
  std::vector<size_t> loc_targets_all;
  std::vector<size_t> loc_targets_intralayer;
  std::vector<size_t> loc_targets_interlayer;
  for (size_t i = ref_loc + 2; i < N; i += 2) {
    loc_targets_all.push_back(i);
    const size_t layer = LayerFromGlobalIndex(i);
    if (layer == 0) loc_targets_intralayer.push_back(i);
    if (layer == 1) loc_targets_interlayer.push_back(i);
  }

  // Per-layer localized spin targets (same-layer only)
  auto targets_after_ref = [](const std::vector<size_t> &sites_vec, size_t ref_site) {
    std::vector<size_t> res;
    res.reserve(sites_vec.size());
    for (size_t s : sites_vec) {
      if (s > ref_site) res.push_back(s);
    }
    return res;
  };
  const auto loc0_targets = targets_after_ref(all_loc0_sites, ref_loc0);
  const auto loc1_targets = targets_after_ref(all_loc1_sites, ref_loc1);
  const auto interlayer_loc_targets = targets_after_ref(all_loc1_sites, ref_loc0);

  // SC target interlayer bonds (two electron sites across layers at same geom site)
  std::vector<std::array<size_t, 2>> target_sites_interlayer_bond_set;
  target_sites_interlayer_bond_set.reserve(num_geom_sites);
  for (size_t s = s_ref + 1; s < num_geom_sites; ++s) {
    target_sites_interlayer_bond_set.push_back({ElectronIndex(s, 0), ElectronIndex(s, 1)});
  }
  const std::array<size_t, 2> ref_sites_sc = {ElectronIndex(s_ref, 0), ElectronIndex(s_ref, 1)};

  // SC pairing operator building blocks
  const std::array<Tensor, 4> sc_phys_ops_a = {ops.bupcF, ops.Fbdnc, ops.bupaF, ops.Fbdna};
  const std::array<Tensor, 4> sc_phys_ops_b = {ops.bdnc, ops.bupc, ops.bupaF, ops.Fbdna};
  const std::array<Tensor, 4> sc_phys_ops_c = {ops.bupcF, ops.Fbdnc, ops.bdna, ops.bupa};
  const std::array<Tensor, 4> sc_phys_ops_d = {ops.bdnc, ops.bupc, ops.bdna, ops.bupa};
  const std::array<Tensor, 4> sc_phys_ops_e = {ops.bupcF, ops.bupc, ops.bupaF, ops.bupa};
  const std::array<Tensor, 4> sc_phys_ops_f = {ops.bdnc, ops.Fbdnc, ops.bdna, ops.Fbdna};

  struct SCTask {
    const std::array<Tensor, 4> &phys_ops;
    const char *label;
  };
  const SCTask sc_tasks[] = {
      {sc_phys_ops_a, "scs_a"},
      {sc_phys_ops_b, "scs_b"},
      {sc_phys_ops_c, "scs_c"},
      {sc_phys_ops_d, "scs_d"},
      {sc_phys_ops_e, "sct_e"},
      {sc_phys_ops_f, "sct_f"},
  };

  // -----------------------------------------------------------------------
  // Filename postfix (matching vmps.cpp convention)
  // -----------------------------------------------------------------------
  std::ostringstream oss;
  oss << "tilted_zigzag"
      << "Jk" << Jk
      << "Jperp" << Jperp
      << "U" << U
      << "t2" << t2
      << "Lx" << Lx
      << "Ly" << Ly
      << "D" << bond_dim
      << "_" << params.Geometry;
  const std::string file_postfix = oss.str();

  // -----------------------------------------------------------------------
  // Measurements
  // -----------------------------------------------------------------------
  using OpT = Tensor;

  // Itinerant two-site correlation operators
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> elec_two_site_ops = {
      {"szsz", ops.sz, ops.sz},
      {"spsm", ops.sp, ops.sm},
      {"smsp", ops.sm, ops.sp},
      {"nfnf", ops.nf, ops.nf},
  };

  // Localized spin two-site correlation operators (for all/intra/inter split)
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> loc_two_site_ops = {
      {"szsz", local_spin_ops.sz, local_spin_ops.sz},
      {"spsm", local_spin_ops.sp, local_spin_ops.sm},
      {"smsp", local_spin_ops.sm, local_spin_ops.sp},
  };

  // Per-layer localized spin correlations
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> loc0_corr_ops = {
      {"l0szsz", local_spin_ops.sz, local_spin_ops.sz},
      {"l0spsm", local_spin_ops.sp, local_spin_ops.sm},
      {"l0smsp", local_spin_ops.sm, local_spin_ops.sp},
  };
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> loc1_corr_ops = {
      {"l1szsz", local_spin_ops.sz, local_spin_ops.sz},
      {"l1spsm", local_spin_ops.sp, local_spin_ops.sm},
      {"l1smsp", local_spin_ops.sm, local_spin_ops.sp},
  };
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> interlayer_corr_ops = {
      {"l01szsz", local_spin_ops.sz, local_spin_ops.sz},
      {"l01spsm", local_spin_ops.sp, local_spin_ops.sm},
      {"l01smsp", local_spin_ops.sm, local_spin_ops.sp},
  };

  clock_t startTime = clock();

  // --- Itinerant electron two-point correlations (all/intra/inter-layer) ---
  for (const auto &[base_label, op1, op2] : elec_two_site_ops) {
    {
      auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_all);
      DumpMeasuRes(res, base_label + std::string("_elec_all_") + file_postfix);
      cout << "Measured " << base_label << "_elec_all at D=" << bond_dim << endl;
    }
    {
      auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_intralayer);
      DumpMeasuRes(res, base_label + std::string("_elec_intra_") + file_postfix);
      cout << "Measured " << base_label << "_elec_intra at D=" << bond_dim << endl;
    }
    {
      auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_interlayer);
      DumpMeasuRes(res, base_label + std::string("_elec_inter_") + file_postfix);
      cout << "Measured " << base_label << "_elec_inter at D=" << bond_dim << endl;
    }
  }

  // --- Localized spin two-point correlations: all/intra/inter-layer ---
  for (const auto &[base_label, op1, op2] : loc_two_site_ops) {
    {
      auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_all);
      DumpMeasuRes(res, base_label + std::string("_loc_all_") + file_postfix);
      cout << "Measured " << base_label << "_loc_all at D=" << bond_dim << endl;
    }
    {
      auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_intralayer);
      DumpMeasuRes(res, base_label + std::string("_loc_intra_") + file_postfix);
      cout << "Measured " << base_label << "_loc_intra at D=" << bond_dim << endl;
    }
    {
      auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_interlayer);
      DumpMeasuRes(res, base_label + std::string("_loc_inter_") + file_postfix);
      cout << "Measured " << base_label << "_loc_inter at D=" << bond_dim << endl;
    }
  }

  // --- Per-layer localized spin correlations (l0, l1, l01) ---
  for (const auto &[label, op1, op2] : loc0_corr_ops) {
    auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc0, loc0_targets);
    DumpMeasuRes(res, label + std::string("_") + file_postfix);
    cout << "Measured " << label << " at D=" << bond_dim << endl;
  }
  for (const auto &[label, op1, op2] : loc1_corr_ops) {
    auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc1, loc1_targets);
    DumpMeasuRes(res, label + std::string("_") + file_postfix);
    cout << "Measured " << label << " at D=" << bond_dim << endl;
  }
  for (const auto &[label, op1, op2] : interlayer_corr_ops) {
    auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc0, interlayer_loc_targets);
    DumpMeasuRes(res, label + std::string("_") + file_postfix);
    cout << "Measured " << label << " at D=" << bond_dim << endl;
  }

  // --- One-site observables ---
  // Itinerant: sz, nf on all even sites
  {
    const std::vector<QLTensor<TenElemT, QNT>> elec_one_site_ops = {ops.sz, ops.nf};
    const std::vector<std::string> elec_one_site_labels = {
        std::string("sz_elec_") + file_postfix,
        std::string("nf_elec_") + file_postfix,
    };
    MeasureOneSiteOp(mps, mps_path, elec_one_site_ops, even_sites, elec_one_site_labels);
    cout << "Measured one-site itinerant observables at D=" << bond_dim << endl;
  }

  // Localized: sz on all odd sites (both layers combined)
  {
    const std::vector<QLTensor<TenElemT, QNT>> loc_one_site_ops = {local_spin_ops.sz};
    const std::vector<std::string> loc_one_site_labels = {std::string("sz_loc_") + file_postfix};
    MeasureOneSiteOp(mps, mps_path, loc_one_site_ops, odd_sites, loc_one_site_labels);
    cout << "Measured one-site localized spin (all) at D=" << bond_dim << endl;
  }

  // Per-layer localized: sz_loc0, sz_loc1
  {
    const std::vector<QLTensor<TenElemT, QNT>> loc_ops = {local_spin_ops.sz};
    const std::vector<std::string> labels = {std::string("sz_loc0_") + file_postfix};
    MeasureOneSiteOp(mps, mps_path, loc_ops, all_loc0_sites, labels);
    cout << "Measured one-site layer-0 localized spin at D=" << bond_dim << endl;
  }
  {
    const std::vector<QLTensor<TenElemT, QNT>> loc_ops = {local_spin_ops.sz};
    const std::vector<std::string> labels = {std::string("sz_loc1_") + file_postfix};
    MeasureOneSiteOp(mps, mps_path, loc_ops, all_loc1_sites, labels);
    cout << "Measured one-site layer-1 localized spin at D=" << bond_dim << endl;
  }

  // --- Interlayer SC correlations (4-site operators) ---
  if (params.SkipSC) {
    cout << "Skipping interlayer SC correlations at D=" << bond_dim << endl;
  } else {
    DeallocAllSites(mps);
    for (size_t i = 0; i < sizeof(sc_tasks) / sizeof(sc_tasks[0]); ++i) {
      auto res = MeasureFourSiteOpGroupInKondoLattice(
          mps,
          mps_path,
          sc_tasks[i].phys_ops,
          ref_sites_sc,
          target_sites_interlayer_bond_set,
          ops.f);
      DumpMeasuRes(res, std::string(sc_tasks[i].label) + "_" + file_postfix);
      cout << "Measured " << sc_tasks[i].label << " at D=" << bond_dim << endl;
    }
  }

  clock_t endTime = clock();
  cout << "CPU Time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

  return 0;
}
