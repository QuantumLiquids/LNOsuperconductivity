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
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

using namespace qlmps;
using namespace qlten;
using namespace std;

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

  // N = 4 * Ly * Lx: 2 layers x 2 orbitals (itinerant + localized)
  const size_t N = 4 * Ly * Lx;

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
  // Site index helpers
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

  // -----------------------------------------------------------------------
  // Reference sites and target lists
  // -----------------------------------------------------------------------
  const size_t n_itinerant = 2 * Ly * Lx;
  const size_t ref_x = Lx / 4;
  const size_t ref_y = 0;

  // Itinerant reference (layer 0)
  const size_t ref_elec0 = elec_site(ref_x, ref_y, 0);
  // Localized reference (layer 0)
  const size_t ref_loc0  = loc_site(ref_x, ref_y, 0);
  // Localized reference (layer 1)
  const size_t ref_loc1  = loc_site(ref_x, ref_y, 1);

  // Collect all itinerant / localized sites in both layers
  std::vector<size_t> all_elec_sites, all_loc0_sites, all_loc1_sites;
  all_elec_sites.reserve(n_itinerant);
  all_loc0_sites.reserve(Ly * Lx);
  all_loc1_sites.reserve(Ly * Lx);
  for (size_t i = 0; i < N; i += 2) all_elec_sites.push_back(i);
  for (size_t x = 0; x < Lx; ++x)
    for (size_t y = 0; y < Ly; ++y) {
      all_loc0_sites.push_back(loc_site(x, y, 0));
      all_loc1_sites.push_back(loc_site(x, y, 1));
    }

  // Target site lists for two-site correlations
  auto targets_after_ref = [&](const std::vector<size_t> &sites_vec, size_t ref_site) {
    std::vector<size_t> res;
    res.reserve(sites_vec.size());
    for (size_t s : sites_vec) {
      if (s > ref_site) res.push_back(s);
    }
    return res;
  };
  const auto elec_targets0 = targets_after_ref(all_elec_sites, ref_elec0);
  const auto loc0_targets = targets_after_ref(all_loc0_sites, ref_loc0);
  const auto loc1_targets = targets_after_ref(all_loc1_sites, ref_loc1);
  const auto interlayer_loc_targets = targets_after_ref(all_loc1_sites, ref_loc0);

  // -----------------------------------------------------------------------
  // Filename postfix
  // -----------------------------------------------------------------------
  std::ostringstream oss;
  oss << "Jperp" << Jperp << "Jk" << Jk << "t2" << t2 << "U" << U
      << "Ly" << Ly << "Lx" << Lx << "D" << bond_dim
      << "_" << params.Geometry;
  const std::string file_postfix = oss.str();

  // -----------------------------------------------------------------------
  // Measurements
  // -----------------------------------------------------------------------
  using OpT = Tensor;

  // -- Itinerant two-site correlations (layer-0 reference) --
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> elec_corr_ops = {
      {"szsz", hubbard_ops.sz, hubbard_ops.sz},
      {"spsm", hubbard_ops.sp, hubbard_ops.sm},
      {"smsp", hubbard_ops.sm, hubbard_ops.sp},
      {"nfnf", hubbard_ops.nf, hubbard_ops.nf}
  };

  // -- Layer-0 localized-spin correlations --
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> loc0_corr_ops = {
      {"l0szsz", local_spin_ops.sz, local_spin_ops.sz},
      {"l0spsm", local_spin_ops.sp, local_spin_ops.sm},
      {"l0smsp", local_spin_ops.sm, local_spin_ops.sp}
  };

  // -- Layer-1 localized-spin correlations --
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> loc1_corr_ops = {
      {"l1szsz", local_spin_ops.sz, local_spin_ops.sz},
      {"l1spsm", local_spin_ops.sp, local_spin_ops.sm},
      {"l1smsp", local_spin_ops.sm, local_spin_ops.sp}
  };

  // -- Inter-layer localized-spin correlations (layer-0 ref -> layer-1 targets) --
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> interlayer_corr_ops = {
      {"l01szsz", local_spin_ops.sz, local_spin_ops.sz},
      {"l01spsm", local_spin_ops.sp, local_spin_ops.sm},
      {"l01smsp", local_spin_ops.sm, local_spin_ops.sp}
  };

  clock_t startTime = clock();

  for (const auto &item : elec_corr_ops) {
    const std::string &label = std::get<0>(item);
    const OpT &op1 = std::get<1>(item);
    const OpT &op2 = std::get<2>(item);
    auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec0, elec_targets0);
    DumpMeasuRes(res, label + file_postfix);
    cout << "Measured " << label << " at D=" << bond_dim << endl;
  }

  for (const auto &item : loc0_corr_ops) {
    const std::string &label = std::get<0>(item);
    const OpT &op1 = std::get<1>(item);
    const OpT &op2 = std::get<2>(item);
    auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc0, loc0_targets);
    DumpMeasuRes(res, label + file_postfix);
    cout << "Measured " << label << " at D=" << bond_dim << endl;
  }

  for (const auto &item : loc1_corr_ops) {
    const std::string &label = std::get<0>(item);
    const OpT &op1 = std::get<1>(item);
    const OpT &op2 = std::get<2>(item);
    auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc1, loc1_targets);
    DumpMeasuRes(res, label + file_postfix);
    cout << "Measured " << label << " at D=" << bond_dim << endl;
  }

  for (const auto &item : interlayer_corr_ops) {
    const std::string &label = std::get<0>(item);
    const OpT &op1 = std::get<1>(item);
    const OpT &op2 = std::get<2>(item);
    auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc0, interlayer_loc_targets);
    DumpMeasuRes(res, label + file_postfix);
    cout << "Measured " << label << " at D=" << bond_dim << endl;
  }

  // -- One-site measurements --
  {
    std::vector<QLTensor<TenElemT, QNT>> ops = {hubbard_ops.sz, hubbard_ops.nf};
    std::vector<std::string> labels = {"sz_elec" + file_postfix, "n_elec" + file_postfix};
    MeasureOneSiteOp(mps, mps_path, ops, all_elec_sites, labels);
    cout << "Measured one-site itinerant observables at D=" << bond_dim << endl;
  }

  {
    std::vector<QLTensor<TenElemT, QNT>> ops = {local_spin_ops.sz};
    std::vector<std::string> labels = {"sz_loc0" + file_postfix};
    MeasureOneSiteOp(mps, mps_path, ops, all_loc0_sites, labels);
    cout << "Measured one-site layer-0 localized spin at D=" << bond_dim << endl;
  }

  {
    std::vector<QLTensor<TenElemT, QNT>> ops = {local_spin_ops.sz};
    std::vector<std::string> labels = {"sz_loc1" + file_postfix};
    MeasureOneSiteOp(mps, mps_path, ops, all_loc1_sites, labels);
    cout << "Measured one-site layer-1 localized spin at D=" << bond_dim << endl;
  }

  clock_t endTime = clock();
  cout << "CPU Time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

  return 0;
}
