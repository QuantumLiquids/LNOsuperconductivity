//
// Two-layer Kondo lattice model on a 45-degree tilted (zig-zag) lattice.
//
// Mapping / data layout (good taste: eliminate special cases):
// - Start from the geometric ordering used by `TiltedZigZagLattice`:
//     geom_site s = y + Ly * x  (x = chain coordinate, y = zig-zag chain index)
// - Each geometric site expands to 4 MPS sites (contiguous block):
//     4*s + 0 : itinerant electron, layer 0
//     4*s + 1 : localized spin,    layer 0
//     4*s + 2 : itinerant electron, layer 1
//     4*s + 3 : localized spin,    layer 1
//   Thus:
//     - even indices are itinerant electrons (fermions)
//     - odd indices are localized spins (non-fermions)
//
// Hamiltonian (paper Eq.(1) spirit):
// - Intralayer hopping: t on intra-chain bonds; t2 on inter-chain (zig-zag NN) bonds
// - Onsite Hubbard U on itinerant sites
// - Onsite ferromagnetic Kondo/Hund JK between itinerant spin and localized spin
// - Interlayer AFM coupling Jperp between localized spins on the same geometric site
//
// Measurements:
// - Per D in params.Dmax: measure itinerant spin/density one-point + two-point (fixed reference),
//   including intra-layer vs inter-layer target splits.
// - Add localized spin one-point + two-point correlations (previously missing).
// - Measure interlayer onsite singlet/triplet pairing correlations (ref bond across layers).
//

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "params_case.h"
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"
#include "../src_tj_double_layer_single_orbital_2d/my_measure.h"
#include "finite_mps_extended.h"
#include "../src_kondo_zigzag_ladder/tilted_zigzag_lattice.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace qlmps;
using namespace qlten;
using namespace std;

namespace {

inline size_t GeomSiteFromGlobalEvenIndex(const size_t global_site) {
  // global_site is an itinerant site => even, and belongs to some geom site block
  return global_site / 4;
}

inline size_t LayerFromGlobalIndex(const size_t global_site) {
  // 0 for block offsets 0/1, 1 for offsets 2/3
  return (global_site % 4) / 2;
}

inline size_t ElectronIndex(const size_t geom_site, const size_t layer) {
  return 4 * geom_site + 2 * layer;
}

inline size_t LocalizedIndex(const size_t geom_site, const size_t layer) {
  return ElectronIndex(geom_site, layer) + 1;
}

inline vector<size_t> EvenIndicesBetween(const size_t a, const size_t b) {
  vector<size_t> res;
  if (a == b) return res;
  const size_t lo = std::min(a, b);
  const size_t hi = std::max(a, b);
  for (size_t k = lo + 2; k < hi; k += 2) {
    res.push_back(k);
  }
  return res;
}

inline void DeallocAllSites(qlmps::FiniteMPS<TenElemT, QNT> &mps) {
  for (size_t i = 0; i < mps.size(); ++i) {
    mps.dealloc(i);
  }
}

}  // namespace

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);

  if (argc < 2) {
    if (rank == 0) {
      std::cerr << "Usage: " << argv[0] << " params.json\n";
    }
    MPI_Finalize();
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

  const size_t num_geom_sites = Ly * Lx;
  const size_t N = 4 * num_geom_sites;

  if (rank == 0) {
    cout << "Lx = " << Lx << "\n";
    cout << "Ly = " << Ly << "\n";
    cout << "N = " << N << "\n";
    cout << "t = " << t << "\n";
    cout << "t2 = " << t2 << "\n";
    cout << "Jk = " << Jk << "\n";
    cout << "Jperp = " << Jperp << "\n";
    cout << "U = " << U << "\n";
    cout << "Geometry = " << params.Geometry << "\n";
  }

  clock_t startTime = clock();

  // Build sites: even=itinerant Hubbard site, odd=localized S=1/2 site.
  std::vector<IndexT> pb_set(N);
  for (size_t i = 0; i < N; ++i) {
    pb_set[i] = (i % 2 == 0) ? pb_outE : pb_outL;
  }
  const SiteVec<TenElemT, QNT> sites(pb_set);
  auto mpo_gen = MPOGenerator<TenElemT, QNT>(sites);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  auto &ops = hubbard_ops;
  SpinOneHalfOperatorsU1U1 local_spin_ops;

  // Hopping helper (fermionic string only across even itinerant sites).
  auto add_hop = [&](size_t site1, size_t site2, double coeff) {
    if (site1 > site2) std::swap(site1, site2);
    const auto inst_sites = EvenIndicesBetween(site1, site2);
    mpo_gen.AddTerm(-coeff, ops.bupcF, site1, ops.bupa, site2, ops.f, inst_sites);
    mpo_gen.AddTerm(coeff, ops.bupaF, site1, ops.bupc, site2, ops.f, inst_sites);
    mpo_gen.AddTerm(-coeff, ops.bdnc, site1, ops.Fbdna, site2, ops.f, inst_sites);
    mpo_gen.AddTerm(coeff, ops.bdna, site1, ops.Fbdnc, site2, ops.f, inst_sites);
  };

  // Lattice bond lists in "single-layer electron indices" (even indices 2*(y+Ly*x)).
  // We lift them to the bilayer by mapping electron indices to geom site id = (i/2),
  // and then applying layer offsets.
  TiltedZigZagLattice lattice(Ly, Lx);

  auto add_bond_set_to_bilayer = [&](const std::vector<std::pair<size_t, size_t>> &pairs, double coeff) {
    for (const auto &p : pairs) {
      const size_t s1 = (p.first / 2);
      const size_t s2 = (p.second / 2);
      for (size_t layer = 0; layer < 2; ++layer) {
        add_hop(ElectronIndex(s1, layer), ElectronIndex(s2, layer), coeff);
      }
    }
  };

  // Intra-zig-zag-chain hopping t
  add_bond_set_to_bilayer(lattice.IntraChainPairs(), t);
  // Inter-chain hopping t2 (OBC part)
  add_bond_set_to_bilayer(lattice.InterChainNNPairsOBC(), t2);
  // PBC-only diagonal winding along y
  if (params.Geometry == "PBC") {
    add_bond_set_to_bilayer(lattice.InterChainNNPairsPBC(), t2);
  }

  // Onsite Hubbard U on itinerant sites
  for (size_t i = 0; i < N; i += 2) {
    mpo_gen.AddTerm(U, ops.nupndn, i);
  }

  // Onsite Kondo/Hund coupling between itinerant and localized on the same layer+geom site.
  for (size_t i = 0; i + 1 < N; i += 2) {
    mpo_gen.AddTerm(Jk, ops.sz, i, local_spin_ops.sz, i + 1);
    mpo_gen.AddTerm(Jk / 2, ops.sp, i, local_spin_ops.sm, i + 1);
    mpo_gen.AddTerm(Jk / 2, ops.sm, i, local_spin_ops.sp, i + 1);
  }

  // Interlayer AFM coupling Jperp between localized spins on the same geometric site.
  for (size_t s = 0; s < num_geom_sites; ++s) {
    const size_t sl0 = LocalizedIndex(s, 0);
    const size_t sl1 = LocalizedIndex(s, 1);
    mpo_gen.AddTerm(Jperp, local_spin_ops.sz, sl0, local_spin_ops.sz, sl1);
    mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sp, sl0, local_spin_ops.sm, sl1);
    mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sm, sl0, local_spin_ops.sp, sl1);
  }

  qlmps::MPO<Tensor> mpo = mpo_gen.Gen();

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
#endif

  // Random direct-product initialization: itinerant quarter filling (n=1/2 per site) across both layers.
  std::vector<size_t> elec_labs(N / 2);
  const size_t num_electrons = (N / 2) / 2;  // total itinerant sites = N/2
  const size_t num_up = num_electrons / 2 + (num_electrons % 2);
  const size_t num_down = num_electrons - num_up;
  std::fill(elec_labs.begin(), elec_labs.begin() + num_up, hubbard_site.spin_up);
  std::fill(elec_labs.begin() + num_up, elec_labs.begin() + num_up + num_down, hubbard_site.spin_down);
  std::fill(elec_labs.begin() + num_up + num_down, elec_labs.end(), hubbard_site.empty);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(elec_labs.begin(), elec_labs.end(), g);

  std::vector<size_t> local_spin_labs(N / 2);
  for (size_t i = 0; i < local_spin_labs.size(); ++i) {
    local_spin_labs[i] = i % 2;
  }
  std::shuffle(local_spin_labs.begin(), local_spin_labs.end(), g);

  std::vector<size_t> stat_labs(N);
  for (size_t i = 0; i < N; i += 2) {
    stat_labs[i] = elec_labs[i / 2];
    stat_labs[i + 1] = local_spin_labs[i / 2];
  }

  // MPS I/O convention from existing codes.
  if (IsPathExist(kMpsPath)) {
    if (N == GetNumofMps()) {
      if (rank == 0) {
        cout << "The number of mps files is consistent with mps size.\n";
        cout << "Directly use mps from files.\n";
      }
    } else {
      qlmps::DirectStateInitMps(mps, stat_labs);
      if (rank == 0) {
        cout << "Initial mps as direct product state.\n";
        mps.Dump(kMpsPath, true);
      }
    }
  } else {
    qlmps::DirectStateInitMps(mps, stat_labs);
    if (rank == 0) {
      cout << "Initial mps as direct product state.\n";
      mps.Dump(kMpsPath, true);
    }
  }

  const std::string mps_path = kMpsPath;

  // Reference geometric site in the bulk-ish region.
  const size_t s_ref = num_geom_sites / 2;
  const size_t ref_elec = ElectronIndex(s_ref, 0);   // layer-0 itinerant site
  const size_t ref_loc = ref_elec + 1;               // layer-0 localized site

  // Precompute site lists (targets must be to the right of ref for disk-measure routines).
  std::vector<size_t> even_sites;
  even_sites.reserve(N / 2);
  for (size_t i = 0; i < N; i += 2) even_sites.push_back(i);

  std::vector<size_t> odd_sites;
  odd_sites.reserve(N / 2);
  for (size_t i = 1; i < N; i += 2) odd_sites.push_back(i);

  std::vector<size_t> elec_targets_all;
  std::vector<size_t> elec_targets_intralayer;
  std::vector<size_t> elec_targets_interlayer;
  for (size_t i = ref_elec + 2; i < N; i += 2) {
    elec_targets_all.push_back(i);
    const size_t layer = LayerFromGlobalIndex(i);
    if (layer == 0) elec_targets_intralayer.push_back(i);
    if (layer == 1) elec_targets_interlayer.push_back(i);
  }

  std::vector<size_t> loc_targets_all;
  std::vector<size_t> loc_targets_intralayer;
  std::vector<size_t> loc_targets_interlayer;
  for (size_t i = ref_loc + 2; i < N; i += 2) {
    loc_targets_all.push_back(i);
    const size_t layer = LayerFromGlobalIndex(i);
    if (layer == 0) loc_targets_intralayer.push_back(i);
    if (layer == 1) loc_targets_interlayer.push_back(i);
  }

  // SC target interlayer bonds (two electron sites across layers at the same geom site).
  std::vector<std::array<size_t, 2>> target_sites_interlayer_bond_set;
  target_sites_interlayer_bond_set.reserve(num_geom_sites);
  for (size_t s = s_ref + 1; s < num_geom_sites; ++s) {
    target_sites_interlayer_bond_set.push_back({ElectronIndex(s, 0), ElectronIndex(s, 1)});
  }
  const std::array<size_t, 2> ref_sites_sc = {ElectronIndex(s_ref, 0), ElectronIndex(s_ref, 1)};

  // Pairing operator building blocks (same as existing two-layer conventional-square code).
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

  // Per-D measurement runner (same policy as vmps_conventional_square.cpp).
  auto run_measurements = [&](size_t bond_dim) {
    std::ostringstream oss;
    oss << "tilted_zigzag"
        << "Jk" << Jk
        << "Jperp" << Jperp
        << "U" << U
        << "t2" << t2
        << "Lx" << Lx
        << "Ly" << Ly
        << "D" << bond_dim;
    const std::string file_postfix = oss.str();

    using OpT = Tensor;

    // --- Itinerant electron two-point correlations (fixed ref, different target sets) ---
    const std::vector<std::tuple<std::string, const OpT &, const OpT &>> elec_two_site_ops = {
        {"szsz", ops.sz, ops.sz},
        {"spsm", ops.sp, ops.sm},
        {"smsp", ops.sm, ops.sp},
        {"nfnf", ops.nf, ops.nf},
    };

    size_t job_idx = 0;
    for (const auto &[base_label, op1, op2] : elec_two_site_ops) {
      // all targets (both layers)
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_all);
        DumpMeasuRes(measu_res, base_label + std::string("_elec_all_") + file_postfix);
      }
      // intralayer (same layer as ref)
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_intralayer);
        DumpMeasuRes(measu_res, base_label + std::string("_elec_intra_") + file_postfix);
      }
      // interlayer (op2 on other layer)
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_interlayer);
        DumpMeasuRes(measu_res, base_label + std::string("_elec_inter_") + file_postfix);
      }
    }

    // --- Localized spin two-point correlations (fixed ref, different target sets) ---
    const std::vector<std::tuple<std::string, const OpT &, const OpT &>> loc_two_site_ops = {
        {"szsz", local_spin_ops.sz, local_spin_ops.sz},
        {"spsm", local_spin_ops.sp, local_spin_ops.sm},
        {"smsp", local_spin_ops.sm, local_spin_ops.sp},
    };
    for (const auto &[base_label, op1, op2] : loc_two_site_ops) {
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_all);
        DumpMeasuRes(measu_res, base_label + std::string("_loc_all_") + file_postfix);
      }
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_intralayer);
        DumpMeasuRes(measu_res, base_label + std::string("_loc_intra_") + file_postfix);
      }
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_interlayer);
        DumpMeasuRes(measu_res, base_label + std::string("_loc_inter_") + file_postfix);
      }
    }

    // --- One-site observables ---
    // Itinerant: sz, nf on all even sites
    const std::vector<QLTensor<TenElemT, QNT>> elec_one_site_ops = {ops.sz, ops.nf};
    const std::vector<std::string> elec_one_site_labels = {
        std::string("sz_elec_") + file_postfix,
        std::string("nf_elec_") + file_postfix,
    };
    if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
      MeasureOneSiteOp(mps, mps_path, elec_one_site_ops, even_sites, elec_one_site_labels);
    }

    // Localized: sz on all odd sites
    const std::vector<QLTensor<TenElemT, QNT>> loc_one_site_ops = {local_spin_ops.sz};
    const std::vector<std::string> loc_one_site_labels = {std::string("sz_loc_") + file_postfix};
    if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
      MeasureOneSiteOp(mps, mps_path, loc_one_site_ops, odd_sites, loc_one_site_labels);
    }

    // --- Interlayer SC correlations (onsite bond across layers) ---
    // The SC measurement helper asserts mps.empty(), so enforce that invariant here.
    DeallocAllSites(mps);
    for (size_t i = 0; i < sizeof(sc_tasks) / sizeof(sc_tasks[0]); ++i) {
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto measu_res = MeasureFourSiteOpGroupInKondoLattice(
            mps,
            mps_path,
            sc_tasks[i].phys_ops,
            ref_sites_sc,
            target_sites_interlayer_bond_set,
            ops.f);
        DumpMeasuRes(measu_res, std::string(sc_tasks[i].label) + "_" + file_postfix);
      }
    }
  };

  // Optional lattice visualization (rank 0).
  if (rank == 0) {
    std::ostringstream svg_name;
    svg_name << "figures/tilted_lattice_bilayer_Ly" << Ly << "_Lx" << Lx << ".svg";
    lattice.DumpSVG(svg_name.str());
  }

  for (size_t i = 0; i < params.Dmax.size(); i++) {
    if (rank == 0) {
      std::cout << "D_max = " << params.Dmax[i] << std::endl;
    }
    qlmps::FiniteVMPSSweepParams sweep_params(
        params.Sweeps,
        params.Dmin,
        params.Dmax[i],
        params.CutOff,
        qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter),
        params.noise);
    auto e0 = qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);
    MPI_Barrier(comm);
    run_measurements(params.Dmax[i]);
    MPI_Barrier(comm);
  }

  if (rank == 0) {
    const clock_t endTime = clock();
    cout << "CPU Time : " << static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC << "s\n";
  }

  MPI_Finalize();
  return 0;
}


