//
// Unified two-layer Kondo lattice model on a 45-degree tilted (zig-zag) lattice.
//
// This file merges features from:
//   - src_kondo_two_layer_2d/vmps_tilted_zigzag.cpp (TiltedZigZagLattice, SC pairing,
//     intra/inter-layer split measurements, SVG visualization, finite_mps_extended.h)
//   - src_kondo_two_layer_zigzag/vmps.cpp (product-state initialization: stripe_pi2pi2,
//     stripe_pi0, random; effective_dmin fix; per-layer localized spin measurements)
//
// Mapping / data layout:
// - Geometric site ordering from TiltedZigZagLattice:
//     geom_site s = y + Ly * x  (x = chain coordinate, y = zigzag chain index)
// - Each geometric site expands to 4 MPS sites (contiguous block):
//     4*s + 0 : itinerant electron, layer 0  (even -> pb_outE)
//     4*s + 1 : localized spin,    layer 0  (odd  -> pb_outL)
//     4*s + 2 : itinerant electron, layer 1  (even -> pb_outE)
//     4*s + 3 : localized spin,    layer 1  (odd  -> pb_outL)
//
// Hamiltonian (paper Eq.(1)):
// - Intralayer hopping: t on intra-chain bonds; t2 on inter-chain (zigzag NN) bonds
// - Onsite Hubbard U on itinerant sites
// - Onsite ferromagnetic Kondo/Hund JK between itinerant spin and localized spin
// - Interlayer AFM coupling Jperp between localized spins on the same geometric site
//
// Measurements (per D in params.Dmax):
// - Itinerant electron two-point correlations (szsz, spsm, smsp, nfnf):
//     all targets, intralayer targets, interlayer targets
// - Localized spin two-point correlations (szsz, spsm, smsp):
//     all targets, intralayer targets, interlayer targets
//     plus per-layer (l0, l1) and cross-layer (l01) measurements
// - One-site observables: sz_elec, nf_elec on even sites; sz_loc on odd sites;
//     sz_loc0, sz_loc1 per layer
// - Interlayer onsite singlet/triplet SC pairing correlations (4-site operators)
//

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "./params_case.h"
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"
#include "../src_tj_double_layer_single_orbital_2d/my_measure.h"
#include "../src_kondo_two_layer_2d/finite_mps_extended.h"
#include "../src_kondo_zigzag_ladder/tilted_zigzag_lattice.h"

#include <algorithm>
#include <functional>
#include <array>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
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
// Index helpers
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

/// Return all even indices strictly between a and b (exclusive).
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

/// Deallocate all loaded MPS tensors (required before SC measurement routines).
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

  if ((Lx * Ly) % 2 != 0) {
    if (rank == 0)
      std::cerr << "Lx*Ly must be even for quarter filling; got "
                << Lx << "*" << Ly << "=" << Lx * Ly << std::endl;
    MPI_Finalize();
    return 1;
  }

  if (rank == 0) {
    cout << "Two-layer zigzag Kondo lattice (unified)\n";
    cout << "Lx = " << Lx << "\n";
    cout << "Ly = " << Ly << "\n";
    cout << "N  = " << N << "\n";
    cout << "t  = " << t << "\n";
    cout << "t2 = " << t2 << "\n";
    cout << "Jk = " << Jk << "\n";
    cout << "Jperp = " << Jperp << "\n";
    cout << "U  = " << U << "\n";
    cout << "Geometry = " << params.Geometry << "\n";
    cout << "InitState = " << params.InitState << "\n";
    cout << "PinField = " << params.PinField << "\n";
    cout << "PinPattern = " << params.PinPattern << "\n";
  }

  clock_t startTime = clock();

  // Coordinate helpers (used for initialization)
  auto elec_site = [&](size_t x, size_t y, size_t layer) -> size_t {
    return ElectronIndex(y + Ly * x, layer);
  };
  auto loc_site = [&](size_t x, size_t y, size_t layer) -> size_t {
    return LocalizedIndex(y + Ly * x, layer);
  };

  // -----------------------------------------------------------------------
  // Physical basis: even = itinerant Hubbard site, odd = localized S=1/2
  // -----------------------------------------------------------------------
  std::vector<IndexT> pb_set(N);
  for (size_t i = 0; i < N; ++i) {
    pb_set[i] = (i % 2 == 0) ? pb_outE : pb_outL;
  }
  const SiteVec<TenElemT, QNT> sites(pb_set);
  auto mpo_gen = MPOGenerator<TenElemT, QNT>(sites);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  auto &ops = hubbard_ops;
  SpinOneHalfOperatorsU1U1 local_spin_ops;

  // Hopping helper (fermionic string only across even itinerant sites)
  auto add_hop = [&](size_t site1, size_t site2, double coeff) {
    if (site1 > site2) std::swap(site1, site2);
    const auto inst_sites = EvenIndicesBetween(site1, site2);
    mpo_gen.AddTerm(-coeff, ops.bupcF, site1, ops.bupa, site2, ops.f, inst_sites);
    mpo_gen.AddTerm(coeff, ops.bupaF, site1, ops.bupc, site2, ops.f, inst_sites);
    mpo_gen.AddTerm(-coeff, ops.bdnc, site1, ops.Fbdna, site2, ops.f, inst_sites);
    mpo_gen.AddTerm(coeff, ops.bdna, site1, ops.Fbdnc, site2, ops.f, inst_sites);
  };

  // -----------------------------------------------------------------------
  // Build Hamiltonian using TiltedZigZagLattice for clean bond generation
  // -----------------------------------------------------------------------
  TiltedZigZagLattice lattice(Ly, Lx);

  auto add_bond_set_to_bilayer = [&](const std::vector<std::pair<size_t, size_t>> &pairs, double coeff) {
    for (const auto &p : pairs) {
      const size_t s1 = (p.first / 2);   // geom site id
      const size_t s2 = (p.second / 2);
      for (size_t layer = 0; layer < 2; ++layer) {
        add_hop(ElectronIndex(s1, layer), ElectronIndex(s2, layer), coeff);
      }
    }
  };

  // Intra-zigzag-chain hopping t
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

  // Onsite Kondo/Hund coupling between itinerant and localized on the same layer+geom site
  for (size_t i = 0; i + 1 < N; i += 2) {
    mpo_gen.AddTerm(Jk, ops.sz, i, local_spin_ops.sz, i + 1);
    mpo_gen.AddTerm(Jk / 2, ops.sp, i, local_spin_ops.sm, i + 1);
    mpo_gen.AddTerm(Jk / 2, ops.sm, i, local_spin_ops.sp, i + 1);
  }

  // Interlayer AFM coupling Jperp between localized spins on the same geometric site
  for (size_t s = 0; s < num_geom_sites; ++s) {
    const size_t sl0 = LocalizedIndex(s, 0);
    const size_t sl1 = LocalizedIndex(s, 1);
    mpo_gen.AddTerm(Jperp, local_spin_ops.sz, sl0, local_spin_ops.sz, sl1);
    mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sp, sl0, local_spin_ops.sm, sl1);
    mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sm, sl0, local_spin_ops.sp, sl1);
  }

  // 7. Boundary pinning field on localized spins
  if (params.PinField != 0.0 && params.PinPattern != "none") {
    std::function<double(size_t, size_t, size_t)> pin_sign;

    if (params.PinPattern == "pi2pi2") {
      // FM within each chain, AFM between chains, reversed between layers
      // Matches stripe_pi2pi2 init: spin_up = y_even ? (layer==1) : (layer==0)
      pin_sign = [](size_t, size_t y, size_t layer) -> double {
        bool y_even = (y % 2 == 0);
        bool spin_up = y_even ? (layer == 1) : (layer == 0);
        return spin_up ? 1.0 : -1.0;
      };
    } else if (params.PinPattern == "pi0") {
      // Period-2 along x, uniform across chains, reversed between layers
      // Matches stripe_pi0 init: spin_up = x_even ? (layer==0) : (layer==1)
      pin_sign = [](size_t x, size_t, size_t layer) -> double {
        bool x_even = (x % 2 == 0);
        bool spin_up = x_even ? (layer == 0) : (layer == 1);
        return spin_up ? 1.0 : -1.0;
      };
    } else {
      throw std::runtime_error(
          "Unknown PinPattern '" + params.PinPattern +
          "'; expected 'none', 'pi2pi2', or 'pi0'");
    }

    // Apply to left (x=0) and right (x=Lx-1) boundary columns
    std::vector<size_t> boundary_xs = {0};
    if (Lx > 1) boundary_xs.push_back(Lx - 1);
    for (size_t x : boundary_xs) {
      for (size_t y = 0; y < Ly; ++y) {
        for (size_t layer = 0; layer < 2; ++layer) {
          double coeff = params.PinField * pin_sign(x, y, layer);
          mpo_gen.AddTerm(coeff, local_spin_ops.sz, loc_site(x, y, layer));
        }
      }
    }
    if (rank == 0) {
      cout << "Applied " << params.PinPattern << " pinning field (h="
           << params.PinField << ") on x=0";
      if (Lx > 1) cout << " and x=" << (Lx - 1);
      cout << endl;
    }
  }

  qlmps::MPO<Tensor> mpo = mpo_gen.Gen();

  // -----------------------------------------------------------------------
  // MPS initialization (supports random, stripe_pi2pi2, stripe_pi0)
  // -----------------------------------------------------------------------
  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
#endif

  const size_t n_itinerant = 2 * Ly * Lx;
  std::vector<size_t> stat_labs(N);

  if (params.InitState == "stripe_pi2pi2") {
    // (pi/2, pi/2) stripe: FM within each chain, AFM between chains,
    // reversed between layers.  Electrons on even-x sites only (quarter filling).
    if (Lx % 2 != 0)
      throw std::runtime_error("stripe_pi2pi2 requires even Lx (got " + std::to_string(Lx) + ")");
    if (rank == 0) cout << "InitState: stripe_pi2pi2\n";
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        bool y_even = (y % 2 == 0);
        bool x_occupied = (x % 2 == 0);
        for (size_t layer = 0; layer < 2; ++layer) {
          bool spin_up_here = y_even ? (layer == 1) : (layer == 0);
          size_t e_lab = hubbard_site.empty;
          if (x_occupied) {
            e_lab = spin_up_here ? hubbard_site.spin_up : hubbard_site.spin_down;
          }
          size_t l_lab = spin_up_here ? 0 : 1;  // 0=up, 1=down
          stat_labs[elec_site(x, y, layer)] = e_lab;
          stat_labs[loc_site(x, y, layer)]  = l_lab;
        }
      }
    }
  } else if (params.InitState == "stripe_pi0") {
    // (pi, 0) stripe: period-2 along x, uniform across chains.
    if (Lx % 2 != 0)
      throw std::runtime_error("stripe_pi0 requires even Lx (got " + std::to_string(Lx) + ")");
    if (rank == 0) cout << "InitState: stripe_pi0\n";
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        for (size_t layer = 0; layer < 2; ++layer) {
          bool x_even = (x % 2 == 0);
          bool spin_up_here = x_even ? (layer == 0) : (layer == 1);
          bool x_occupied = (x % 2 == 0);
          size_t e_lab = hubbard_site.empty;
          if (x_occupied) {
            e_lab = spin_up_here ? hubbard_site.spin_up : hubbard_site.spin_down;
          }
          size_t l_lab = spin_up_here ? 0 : 1;
          stat_labs[elec_site(x, y, layer)] = e_lab;
          stat_labs[loc_site(x, y, layer)]  = l_lab;
        }
      }
    }
  } else {
    // Default: random initial state
    if (params.InitState != "random" && rank == 0)
      cerr << "WARNING: unrecognized InitState '" << params.InitState
           << "', falling back to random\n";
    if (rank == 0) cout << "InitState: random\n";

    std::vector<size_t> elec_labs(n_itinerant);
    const size_t num_electrons = Lx * Ly;  // quarter filling total
    const size_t num_up = num_electrons / 2 + (num_electrons % 2);
    const size_t num_down = num_electrons - num_up;
    std::fill(elec_labs.begin(),
              elec_labs.begin() + static_cast<ptrdiff_t>(num_up),
              hubbard_site.spin_up);
    std::fill(elec_labs.begin() + static_cast<ptrdiff_t>(num_up),
              elec_labs.begin() + static_cast<ptrdiff_t>(num_up + num_down),
              hubbard_site.spin_down);
    std::fill(elec_labs.begin() + static_cast<ptrdiff_t>(num_up + num_down),
              elec_labs.end(),
              hubbard_site.empty);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(elec_labs.begin(), elec_labs.end(), g);

    const size_t n_loc = 2 * Ly * Lx;
    std::vector<size_t> local_spin_labs(n_loc);
    for (size_t i = 0; i < n_loc; ++i) local_spin_labs[i] = i % 2;
    std::shuffle(local_spin_labs.begin(), local_spin_labs.end(), g);

    for (size_t i = 0; i < N; i += 2) {
      stat_labs[i]     = elec_labs[i / 2];
      stat_labs[i + 1] = local_spin_labs[i / 2];
    }
  }

  // MPS I/O: resume if existing MPS found with correct size
  if (IsPathExist(kMpsPath)) {
    if (N == GetNumofMps()) {
      if (rank == 0) {
        cout << "Existing MPS found; resuming.\n";
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

  // -----------------------------------------------------------------------
  // Reference sites for measurements
  // -----------------------------------------------------------------------
  // Quarter-point reference: maximizes rightward correlation range for OBC
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

  // Per-layer localized site lists (for one-site measurements)
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
  auto targets_after_ref = [](const std::vector<size_t> &sites, size_t ref_site) {
    std::vector<size_t> res;
    res.reserve(sites.size());
    for (size_t s : sites) {
      if (s > ref_site) res.push_back(s);
    }
    return res;
  };
  const auto loc0_targets = targets_after_ref(all_loc0_sites, ref_loc0);
  const auto loc1_targets = targets_after_ref(all_loc1_sites, ref_loc1);
  const auto interlayer_loc_targets = targets_after_ref(all_loc1_sites, ref_loc0);

  // SC target interlayer bonds (two electron sites across layers at the same geom site)
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
  // Per-D measurement runner
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

  auto run_measurements = [&](size_t bond_dim) {
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

    size_t job_idx = 0;

    // --- Itinerant electron two-point correlations (all/intra/inter-layer) ---
    for (const auto &[base_label, op1, op2] : elec_two_site_ops) {
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_all);
        DumpMeasuRes(res, base_label + std::string("_elec_all_") + file_postfix);
      }
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_intralayer);
        DumpMeasuRes(res, base_label + std::string("_elec_intra_") + file_postfix);
      }
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets_interlayer);
        DumpMeasuRes(res, base_label + std::string("_elec_inter_") + file_postfix);
      }
    }

    // --- Localized spin two-point correlations: all/intra/inter-layer (from tilted_zigzag) ---
    for (const auto &[base_label, op1, op2] : loc_two_site_ops) {
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_all);
        DumpMeasuRes(res, base_label + std::string("_loc_all_") + file_postfix);
      }
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_intralayer);
        DumpMeasuRes(res, base_label + std::string("_loc_intra_") + file_postfix);
      }
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets_interlayer);
        DumpMeasuRes(res, base_label + std::string("_loc_inter_") + file_postfix);
      }
    }

    // --- Per-layer localized spin correlations (l0, l1, l01 from zigzag) ---
    for (const auto &[label, op1, op2] : loc0_corr_ops) {
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc0, loc0_targets);
        DumpMeasuRes(res, label + std::string("_") + file_postfix);
        cout << "Measured " << label << " at D=" << bond_dim << endl;
      }
    }
    for (const auto &[label, op1, op2] : loc1_corr_ops) {
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc1, loc1_targets);
        DumpMeasuRes(res, label + std::string("_") + file_postfix);
        cout << "Measured " << label << " at D=" << bond_dim << endl;
      }
    }
    for (const auto &[label, op1, op2] : interlayer_corr_ops) {
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc0, interlayer_loc_targets);
        DumpMeasuRes(res, label + std::string("_") + file_postfix);
        cout << "Measured " << label << " at D=" << bond_dim << endl;
      }
    }

    // --- One-site observables ---
    // Itinerant: sz, nf on all even sites
    if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
      const std::vector<QLTensor<TenElemT, QNT>> elec_one_site_ops = {ops.sz, ops.nf};
      const std::vector<std::string> elec_one_site_labels = {
          std::string("sz_elec_") + file_postfix,
          std::string("nf_elec_") + file_postfix,
      };
      MeasureOneSiteOp(mps, mps_path, elec_one_site_ops, even_sites, elec_one_site_labels);
    }

    // Localized: sz on all odd sites
    if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
      const std::vector<QLTensor<TenElemT, QNT>> loc_one_site_ops = {local_spin_ops.sz};
      const std::vector<std::string> loc_one_site_labels = {std::string("sz_loc_") + file_postfix};
      MeasureOneSiteOp(mps, mps_path, loc_one_site_ops, odd_sites, loc_one_site_labels);
    }

    // Per-layer localized: sz_loc0, sz_loc1
    if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
      const std::vector<QLTensor<TenElemT, QNT>> loc_ops = {local_spin_ops.sz};
      const std::vector<std::string> labels = {std::string("sz_loc0_") + file_postfix};
      MeasureOneSiteOp(mps, mps_path, loc_ops, all_loc0_sites, labels);
    }
    if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
      const std::vector<QLTensor<TenElemT, QNT>> loc_ops = {local_spin_ops.sz};
      const std::vector<std::string> labels = {std::string("sz_loc1_") + file_postfix};
      MeasureOneSiteOp(mps, mps_path, loc_ops, all_loc1_sites, labels);
    }

    // --- Interlayer SC correlations (4-site operators) ---
    DeallocAllSites(mps);
    for (size_t i = 0; i < sizeof(sc_tasks) / sizeof(sc_tasks[0]); ++i) {
      if ((job_idx++) % mpi_size == static_cast<size_t>(rank)) {
        auto res = MeasureFourSiteOpGroupInKondoLattice(
            mps,
            mps_path,
            sc_tasks[i].phys_ops,
            ref_sites_sc,
            target_sites_interlayer_bond_set,
            ops.f);
        DumpMeasuRes(res, std::string(sc_tasks[i].label) + "_" + file_postfix);
      }
    }
  };

  // -----------------------------------------------------------------------
  // Optional lattice visualization (rank 0)
  // -----------------------------------------------------------------------
  if (rank == 0) {
    std::ostringstream svg_name;
    svg_name << "figures/tilted_lattice_bilayer_Ly" << Ly << "_Lx" << Lx << ".svg";
    lattice.DumpSVG(svg_name.str());
  }

  // -----------------------------------------------------------------------
  // VMPS sweeps + measurements
  // -----------------------------------------------------------------------
  for (size_t i = 0; i < params.Dmax.size(); i++) {
    const size_t bond_dim = params.Dmax[i];
    // Clamp Dmin to not exceed current bond dimension (important for small-D warmup)
    const size_t effective_dmin = std::min(params.Dmin, bond_dim);
    if (rank == 0) {
      std::cout << "D_max = " << bond_dim << std::endl;
    }
    qlmps::FiniteVMPSSweepParams sweep_params(
        params.Sweeps,
        effective_dmin,
        bond_dim,
        params.CutOff,
        qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter),
        params.noise);
    auto e0 = qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);

    if (rank == 0 && i + 1 == params.Dmax.size()) {
      auto ee_list = mps.GetEntanglementEntropy(1);
      std::copy(ee_list.begin(), ee_list.end(),
                std::ostream_iterator<double>(std::cout, " "));
      cout << "\nmiddle EE = " << ee_list[N / 2] << endl;
    }

    MPI_Barrier(comm);
    run_measurements(bond_dim);
    MPI_Barrier(comm);
  }

  if (rank == 0) {
    const clock_t endTime = clock();
    cout << "CPU Time : " << static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC << "s\n";
  }

  MPI_Finalize();
  return 0;
}
