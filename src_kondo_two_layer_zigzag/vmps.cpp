/*
 * Two-layer Kondo lattice on a tilted zigzag geometry with interlayer J_perp.
 *
 * Purpose: study the effect of a finite interlayer coupling J_perp on the
 * (pi/2, pi/2) spin-stripe order (Referee A, point 1).
 *
 * Geometry
 * --------
 * Two identical single-layer tilted-zigzag Kondo ladders stacked on top of
 * each other.  Each layer has Ly zigzag chains running along x with:
 *   - intra-chain NN hopping   t   : (y, x) -> (y, x+1)
 *   - inter-chain diag hopping t2  : (y, x) -> (y +/- 1, x+1)  [zigzag]
 *   - on-site Kondo coupling   JK  : itinerant s . localized S
 *   - Hubbard repulsion        U   : itinerant sites
 *
 * The two layers are coupled by:
 *   - interlayer exchange       Jperp : S_z^{l0} . S_z^{l1}  (localized spins only)
 *
 * Site ordering (MPS chain index)
 * --------------------------------
 * For geometric site (x, y), 4 consecutive MPS sites:
 *   4*(y + Ly*x) + 0  : layer-0 itinerant electron   (even -> pb_outE)
 *   4*(y + Ly*x) + 1  : layer-0 localized spin        (odd  -> pb_outL)
 *   4*(y + Ly*x) + 2  : layer-1 itinerant electron   (even -> pb_outE)
 *   4*(y + Ly*x) + 3  : layer-1 localized spin        (odd  -> pb_outL)
 *
 * Total MPS sites N = 4 * Ly * Lx.
 *
 * Jordan-Wigner string
 * ---------------------
 * All even-indexed MPS sites are fermionic.  For a hopping term between
 * electron sites p1 < p2, the JW insertion operators sit on every even k
 * with p1 < k < p2 (step +2 skips localised spins automatically).
 *
 * Filling
 * -------
 * Quarter filling of the itinerant band: Lx*Ly electrons across both layers
 * (= 0.5 electrons per itinerant site per layer, matching La3Ni2O7 nominal
 * filling of n_{x^2-y^2} = 0.5 per site).
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

using namespace qlmps;
using namespace qlten;
using namespace std;

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

  if ((Lx * Ly) % 2 != 0) {
    if (rank == 0)
      std::cerr << "Lx*Ly must be even for quarter filling; got "
                << Lx << "*" << Ly << "=" << Lx * Ly << std::endl;
    MPI_Finalize();
    return 1;
  }

  // N = 4 * Ly * Lx: 2 layers x 2 orbitals (itinerant + localized)
  const size_t N = 4 * Ly * Lx;

  if (rank == 0) {
    cout << "Two-layer zigzag Kondo lattice\n";
    cout << "Lx = " << Lx << "\n";
    cout << "Ly = " << Ly << "\n";
    cout << "N  = " << N << "\n";
    cout << "t  = " << t << "\n";
    cout << "t2 = " << t2 << "\n";
    cout << "Jk = " << Jk << "\n";
    cout << "U  = " << U << "\n";
    cout << "Jperp = " << Jperp << "\n";
    cout << "Geometry = " << params.Geometry << "\n";
  }

  // -----------------------------------------------------------------------
  // Site index helpers
  // -----------------------------------------------------------------------
  // electron_site(x, y, layer) and localized_site(x, y, layer)
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
  auto mpo_gen = MPOGenerator<TenElemT, QNT>(sites);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  SpinOneHalfOperatorsU1U1 local_spin_ops;
  auto f = hubbard_ops.f;

  // -----------------------------------------------------------------------
  // Helper: add hopping between two electron sites (JW string inserted)
  // -----------------------------------------------------------------------
  auto add_hop = [&](size_t s1, size_t s2, double coeff) {
    if (s1 > s2) std::swap(s1, s2);
    std::vector<size_t> ins;
    for (size_t k = s1 + 2; k < s2; k += 2) ins.push_back(k);
    mpo_gen.AddTerm(-coeff, hubbard_ops.bupcF, s1, hubbard_ops.bupa,  s2, f, ins);
    mpo_gen.AddTerm( coeff, hubbard_ops.bupaF, s1, hubbard_ops.bupc,  s2, f, ins);
    mpo_gen.AddTerm(-coeff, hubbard_ops.bdnc,  s1, hubbard_ops.Fbdna, s2, f, ins);
    mpo_gen.AddTerm( coeff, hubbard_ops.bdna,  s1, hubbard_ops.Fbdnc, s2, f, ins);
  };

  // -----------------------------------------------------------------------
  // Build Hamiltonian
  // -----------------------------------------------------------------------

  // 1. Intra-chain hopping t along x (both layers)
  for (size_t x = 0; x + 1 < Lx; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      for (size_t layer = 0; layer < 2; ++layer) {
        add_hop(elec_site(x, y, layer), elec_site(x + 1, y, layer), t);
      }
    }
  }

  // 2. Zigzag inter-chain hopping t2 (OBC along y, both layers)
  //    Convention: for even x, couple (y, x) -> (y+1, x+1)
  //                for odd  x, couple (y, x) -> (y-1, x+1)
  for (size_t x = 0; x + 1 < Lx; ++x) {
    const int delta = (x % 2 == 0) ? 1 : -1;
    for (size_t y = 0; y < Ly; ++y) {
      const int target = static_cast<int>(y) + delta;
      if (target >= 0 && target < static_cast<int>(Ly)) {
        const size_t ty = static_cast<size_t>(target);
        for (size_t layer = 0; layer < 2; ++layer) {
          add_hop(elec_site(x, y, layer), elec_site(x + 1, ty, layer), t2);
        }
      }
    }
  }

  // 3. PBC wrap for zigzag inter-chain hopping
  if (params.Geometry == "PBC") {
    for (size_t x = 0; x + 1 < Lx; ++x) {
      const int delta = (x % 2 == 0) ? 1 : -1;
      for (size_t y = 0; y < Ly; ++y) {
        const int raw_target = static_cast<int>(y) + delta;
        if (raw_target < 0 || raw_target >= static_cast<int>(Ly)) {
          int wrapped = raw_target;
          if (wrapped < 0) wrapped += static_cast<int>(Ly);
          else wrapped -= static_cast<int>(Ly);
          const size_t ty = static_cast<size_t>(wrapped);
          for (size_t layer = 0; layer < 2; ++layer) {
            add_hop(elec_site(x, y, layer), elec_site(x + 1, ty, layer), t2);
          }
        }
      }
    }
  }

  // 4. Hubbard U on itinerant sites
  for (size_t x = 0; x < Lx; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      for (size_t layer = 0; layer < 2; ++layer) {
        mpo_gen.AddTerm(U, hubbard_ops.nupndn, elec_site(x, y, layer));
      }
    }
  }

  // 5. Kondo coupling JK between itinerant and localized spins (both layers)
  for (size_t x = 0; x < Lx; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      for (size_t layer = 0; layer < 2; ++layer) {
        const size_t e = elec_site(x, y, layer);
        const size_t l = loc_site(x, y, layer);
        mpo_gen.AddTerm(Jk,     hubbard_ops.sz, e, local_spin_ops.sz, l);
        mpo_gen.AddTerm(Jk / 2, hubbard_ops.sp, e, local_spin_ops.sm, l);
        mpo_gen.AddTerm(Jk / 2, hubbard_ops.sm, e, local_spin_ops.sp, l);
      }
    }
  }

  // 6. Interlayer AFM exchange Jperp between localized spins
  for (size_t x = 0; x < Lx; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      const size_t s0 = loc_site(x, y, 0);
      const size_t s1 = loc_site(x, y, 1);
      mpo_gen.AddTerm(Jperp,     local_spin_ops.sz, s0, local_spin_ops.sz, s1);
      mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sp, s0, local_spin_ops.sm, s1);
      mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sm, s0, local_spin_ops.sp, s1);
    }
  }

  qlmps::MPO<Tensor> mpo = mpo_gen.Gen();

  // -----------------------------------------------------------------------
  // MPS initialisation
  // -----------------------------------------------------------------------
  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
#endif

  // Quarter filling: Lx*Ly electrons distributed across 2*Lx*Ly itinerant sites
  const size_t n_itinerant = 2 * Ly * Lx;

  std::vector<size_t> elec_labs(n_itinerant);
  std::fill(elec_labs.begin(),
            elec_labs.begin() + (ptrdiff_t)(Lx * Ly / 2),
            hubbard_site.spin_up);
  std::fill(elec_labs.begin() + (ptrdiff_t)(Lx * Ly / 2),
            elec_labs.begin() + (ptrdiff_t)(Lx * Ly),
            hubbard_site.spin_down);
  std::fill(elec_labs.begin() + (ptrdiff_t)(Lx * Ly),
            elec_labs.end(),
            hubbard_site.empty);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(elec_labs.begin(), elec_labs.end(), g);

  // Localized spins: alternating up/down (one per orbital-site pair)
  const size_t n_loc = 2 * Ly * Lx;
  std::vector<size_t> loc_labs(n_loc);
  for (size_t i = 0; i < n_loc; ++i) loc_labs[i] = i % 2;
  std::shuffle(loc_labs.begin(), loc_labs.end(), g);

  // Build stat_labs: N entries, even -> electron, odd -> localized
  std::vector<size_t> stat_labs(N);
  for (size_t i = 0; i < N; i += 2) {
    stat_labs[i]     = elec_labs[i / 2];
    stat_labs[i + 1] = loc_labs[i / 2];
  }

  if (IsPathExist(kMpsPath)) {
    if (N == GetNumofMps()) {
      cout << "Existing MPS found; resuming." << endl;
    } else {
      qlmps::DirectStateInitMps(mps, stat_labs);
      if (rank == 0) mps.Dump(kMpsPath, true);
    }
  } else {
    qlmps::DirectStateInitMps(mps, stat_labs);
    if (rank == 0) mps.Dump(kMpsPath, true);
  }

  // -----------------------------------------------------------------------
  // Measurements: run after every D in params.Dmax
  // -----------------------------------------------------------------------
  const std::string mps_path = kMpsPath;

  // Reference sites: choose layer-0 localized spin near the chain centre
  // ref_x is at 1/4 of the chain length to maximize correlation range
  const size_t ref_x = Lx / 4;
  const size_t ref_y = 0;

  // Itinerant reference (layer 0)
  const size_t ref_elec0 = elec_site(ref_x, ref_y, 0);
  // Localized reference (layer 0)
  const size_t ref_loc0  = loc_site(ref_x, ref_y, 0);
  // Localized reference (layer 1)
  const size_t ref_loc1  = loc_site(ref_x, ref_y, 1);

  // Collect all itinerant / localized sites in both layers (for one-site measurements)
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

  // Target site lists for two-site correlations (strictly same-layer for l0/l1)
  auto targets_after_ref = [&](const std::vector<size_t> &sites, size_t ref_site) {
    std::vector<size_t> res;
    res.reserve(sites.size());
    for (size_t s : sites) {
      if (s > ref_site) res.push_back(s);
    }
    return res;
  };
  const auto elec_targets0 = targets_after_ref(all_elec_sites, ref_elec0);
  const auto loc0_targets = targets_after_ref(all_loc0_sites, ref_loc0);
  const auto loc1_targets = targets_after_ref(all_loc1_sites, ref_loc1);

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

  auto run_measurements = [&](size_t bond_dim) {
    std::ostringstream oss;
    oss << "Jperp" << Jperp << "Jk" << Jk << "t2" << t2 << "U" << U
        << "Ly" << Ly << "Lx" << Lx << "D" << bond_dim;
    const std::string file_postfix = oss.str();

    size_t job_idx = 0;
    auto do_job = [&](auto &&job) {
      if ((job_idx % static_cast<size_t>(mpi_size)) == static_cast<size_t>(rank)) {
        job();
      }
      ++job_idx;
    };

    for (const auto &item : elec_corr_ops) {
      const std::string &label = std::get<0>(item);
      const OpT &op1 = std::get<1>(item);
      const OpT &op2 = std::get<2>(item);
      do_job([&]() {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec0, elec_targets0);
        DumpMeasuRes(res, label + file_postfix);
        cout << "Measured " << label << " at D=" << bond_dim << endl;
      });
    }

    for (const auto &item : loc0_corr_ops) {
      const std::string &label = std::get<0>(item);
      const OpT &op1 = std::get<1>(item);
      const OpT &op2 = std::get<2>(item);
      do_job([&]() {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc0, loc0_targets);
        DumpMeasuRes(res, label + file_postfix);
        cout << "Measured " << label << " at D=" << bond_dim << endl;
      });
    }

    for (const auto &item : loc1_corr_ops) {
      const std::string &label = std::get<0>(item);
      const OpT &op1 = std::get<1>(item);
      const OpT &op2 = std::get<2>(item);
      do_job([&]() {
        auto res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc1, loc1_targets);
        DumpMeasuRes(res, label + file_postfix);
        cout << "Measured " << label << " at D=" << bond_dim << endl;
      });
    }

    do_job([&]() {
      std::vector<QLTensor<TenElemT, QNT>> ops = {hubbard_ops.sz, hubbard_ops.nf};
      std::vector<std::string> labels = {"sz_elec" + file_postfix, "n_elec" + file_postfix};
      MeasureOneSiteOp(mps, mps_path, ops, all_elec_sites, labels);
      cout << "Measured one-site itinerant observables at D=" << bond_dim << endl;
    });

    do_job([&]() {
      std::vector<QLTensor<TenElemT, QNT>> ops = {local_spin_ops.sz};
      std::vector<std::string> labels = {"sz_loc0" + file_postfix};
      MeasureOneSiteOp(mps, mps_path, ops, all_loc0_sites, labels);
      cout << "Measured one-site layer-0 localized spin at D=" << bond_dim << endl;
    });

    do_job([&]() {
      std::vector<QLTensor<TenElemT, QNT>> ops = {local_spin_ops.sz};
      std::vector<std::string> labels = {"sz_loc1" + file_postfix};
      MeasureOneSiteOp(mps, mps_path, ops, all_loc1_sites, labels);
      cout << "Measured one-site layer-1 localized spin at D=" << bond_dim << endl;
    });
  };

  // -----------------------------------------------------------------------
  // VMPS sweeps + measurements
  // -----------------------------------------------------------------------
  clock_t startTime = clock();

  for (size_t di = 0; di < params.Dmax.size(); ++di) {
    const size_t bond_dim = params.Dmax[di];
    const size_t effective_dmin = std::min(params.Dmin, bond_dim);
    if (rank == 0) cout << "D_max = " << bond_dim << endl;

    qlmps::FiniteVMPSSweepParams sweep_params(
        params.Sweeps,
        effective_dmin, bond_dim, params.CutOff,
        qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter),
        params.noise
    );
    qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);

    if (rank == 0 && di + 1 == params.Dmax.size()) {
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
    clock_t endTime = clock();
    cout << "CPU Time: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }

  MPI_Finalize();
  return 0;
}
