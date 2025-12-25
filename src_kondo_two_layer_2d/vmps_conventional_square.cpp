//
// Created by Haoxin Wang on 3/7/2025.
//
/*
 * Two-layer, multi-leg Kondo lattice model on a conventional square lattice.
 *
 * OBC on both x and y directions.
 * 
 * DMRG lattice mapping (two layers, two on-site orbitals):
 *   - Total sites N = 4 * Ly * Lx. The factor 4 counts two layers and the
 *     itinerant vs localized orbital on every rung. Even indices are itinerant
 *     (extended) electrons; odd indices are localized spins.
 *   - Indices advance along x. For fixed x the ordering walks through all
 *     transverse legs y = 0 ... Ly - 1 and both layers, switching orbital
 *     character on every step. Hence each x-rung contributes 4 * Ly physical
 *     sites to the 1D chain.
 *   - Hopping along x connects indices separated by 4 * Ly. Hopping along y at
 *     fixed layer connects neighbors whose indices differ by 4. Interlayer
 *     exchange couples localized spins whose indices differ by 2.
 *   - Downstream analysis scripts can reconstruct the spatial separation using
 *     the above mapping: Δx = |i - i_ref| / (4 * Ly) for bonds within the same
 *     leg.
 */

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "params_case.h"
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"
#include "../src_tj_double_layer_single_orbital_2d/my_measure.h"
#include "finite_mps_extended.h"

using namespace qlmps;
using namespace qlten;
using namespace std;

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);

  CaseParams params(argv[1]);
  size_t Lx = params.Lx; // Lx should be even number, for N/4 should on electron site for measure
  size_t Ly = params.Ly;
  double t = params.t, Jk = params.JK, U = params.U;
  double Jperp = params.Jperp;
  double t2 = params.t2;
  size_t N = 4 * Ly * Lx; // 4 for double layer times two orbital (localized & itinerate)
  // order of sites for fixed Lx :
  // (layer0, ly0) ---> (layer1, ly0)
  // ---> (layer0, ly1) ----> (layer1, ly1)

  /*** Print the model parameter Info ***/
  if (rank == 0) {
    cout << "Lx = " << Lx << endl;
    cout << "Ly = " << Ly << endl;
    cout << "N = " << N << endl;
    cout << "t = " << t << endl;
    cout << "t2 = " << t2 << endl;
    cout << "Jk = " << Jk << endl;
    cout << "U = " << U << endl;
    cout << "Geometry = " << params.Geometry << endl;
  }

  clock_t startTime, endTime;
  startTime = clock();

  std::vector<IndexT> pb_set = std::vector<IndexT>(N);
  for (size_t i = 0; i < N; ++i) {
    if (i % 2 == 0) pb_set[i] = pb_outE; // even site is extended electron
    if (i % 2 == 1) pb_set[i] = pb_outL; // odd site is localized electron
  }
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(pb_set);
  auto mpo_gen = MPOGenerator<TenElemT, QNT>(sites);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  auto &ops = hubbard_ops;
  SpinOneHalfOperatorsU1U1 local_spin_ops;
  auto f = hubbard_ops.f;

  auto add_intralayer_hopping = [&](size_t site1, size_t site2) {
    std::vector<size_t> inst_op_idxs;
    for (size_t j = site1 + 2; j < site2; j += 2) {
      inst_op_idxs.push_back(j);
    }
    mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f, inst_op_idxs);
    mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f, inst_op_idxs);
    mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f, inst_op_idxs);
    mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f, inst_op_idxs);
  };

  auto electron_site_index = [&](size_t x, size_t y, size_t layer) {
    const size_t block = 4 * Ly;
    return x * block + y * 4 + layer * 2;
  };

  auto localized_site_index = [&](size_t x, size_t y, size_t layer) {
    return electron_site_index(x, y, layer) + 1;
  };

  // hopping along x direction
  for (size_t i = 0; i < N - 4 * Ly; i += 2) {
    add_intralayer_hopping(i, i + 4 * Ly);
  }

  // hopping along y direction within each layer
  for (size_t x = 0; x < Lx; ++x) {
    for (size_t y = 0; y + 1 < Ly; ++y) {
      const size_t site_layer0_curr = electron_site_index(x, y, 0);
      const size_t site_layer0_next = electron_site_index(x, y + 1, 0);
      add_intralayer_hopping(site_layer0_curr, site_layer0_next);

      const size_t site_layer1_curr = electron_site_index(x, y, 1);
      const size_t site_layer1_next = electron_site_index(x, y + 1, 1);
      add_intralayer_hopping(site_layer1_curr, site_layer1_next);
    }
  }

  for (size_t i = 0; i < N; i += 2) {
    mpo_gen.AddTerm(U, hubbard_ops.nupndn, i);
  }

  for (size_t i = 0; i < N - 1; i = i + 2) {
    mpo_gen.AddTerm(Jk, hubbard_ops.sz, i, local_spin_ops.sz, i + 1);
    mpo_gen.AddTerm(Jk / 2, hubbard_ops.sp, i, local_spin_ops.sm, i + 1);
    mpo_gen.AddTerm(Jk / 2, hubbard_ops.sm, i, local_spin_ops.sp, i + 1);
  }

  // inter-layer AFM coupling within each rung
  for (size_t x = 0; x < Lx; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      const size_t site_layer0 = localized_site_index(x, y, 0);
      const size_t site_layer1 = localized_site_index(x, y, 1);
      mpo_gen.AddTerm(Jperp, local_spin_ops.sz, site_layer0, local_spin_ops.sz, site_layer1);
      mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sp, site_layer0, local_spin_ops.sm, site_layer1);
      mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sm, site_layer0, local_spin_ops.sp, site_layer1);
    }
  }

  qlmps::MPO<Tensor> mpo = mpo_gen.Gen();

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
#endif

  std::vector<size_t> elec_labs(2 * Ly * Lx);
  //electron quarter filling
  std::fill(elec_labs.begin(), elec_labs.begin() + Lx * Ly / 2, hubbard_site.spin_up);
  std::fill(elec_labs.begin() + Lx * Ly / 2, elec_labs.begin() + Lx * Ly, hubbard_site.spin_down);
  std::fill(elec_labs.begin() + Lx * Ly, elec_labs.end(), hubbard_site.empty);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(elec_labs.begin(), elec_labs.end(), g);

  std::vector<size_t> local_spin_labs(2 * Ly * Lx);
  for (size_t i = 0; i < local_spin_labs.size(); i++) {
    local_spin_labs[i] = i % 2;
  }
  std::shuffle(local_spin_labs.begin(), local_spin_labs.end(), g);

  std::vector<size_t> stat_labs(N);
  for (size_t i = 0; i < N; i = i + 2) {
    stat_labs[i] = elec_labs[i / 2];
    stat_labs[i + 1] = local_spin_labs[i / 2];
  }

  if (IsPathExist(kMpsPath)) {
    if (N == GetNumofMps()) {
      cout << "The number of mps files is consistent with mps size." << endl;
      cout << "Directly use mps from files." << endl;
    } else {
      qlmps::DirectStateInitMps(mps, stat_labs);
      cout << "Initial mps as direct product state." << endl;
      if (rank == 0)
        mps.Dump(kMpsPath, true);
    }
  } else {
    qlmps::DirectStateInitMps(mps, stat_labs);
    cout << "Initial mps as direct product state." << endl;
    if (rank == 0)
      mps.Dump(kMpsPath, true);
  }

  std::string mps_path = kMpsPath;

  size_t ref_site = N / 4;
  if (ref_site % 2 == 1) {
    ref_site += 1;
  }
  std::vector<size_t> target_sites;
  for (size_t i = ref_site + 2; i < N; i += 2) {
    target_sites.push_back(i);
  }

  std::vector<size_t> even_sites;
  for (size_t i = 0; i < N; i += 2) {
    even_sites.push_back(i);
  }

  using OpT = Tensor;
  const std::vector<std::tuple<std::string, const OpT &, const OpT &> > two_site_meas_ops = {
    {"szsz", hubbard_ops.sz, hubbard_ops.sz},
    {"spsm", hubbard_ops.sp, hubbard_ops.sm},
    {"smsp", hubbard_ops.sm, hubbard_ops.sp},
    {"nfnf", hubbard_ops.nf, hubbard_ops.nf}
  };

  const std::vector<QLTensor<TenElemT, QNT> > one_site_ops = {hubbard_ops.sz, hubbard_ops.nf};
  const std::vector<std::string> one_site_base_labels = {"sz_local", "nf_local"};

  std::vector<std::array<size_t, 2> > target_sites_interlayer_bond_set;
  target_sites_interlayer_bond_set.reserve(Lx * Ly);

  size_t begin_x = Lx / 4;
  size_t end_x = Lx - 1;
  auto bond_sites = [&](size_t x, size_t y) {
    return std::array<size_t, 2>{electron_site_index(x, y, 0), electron_site_index(x, y, 1)};
  };

  std::array<size_t, 2> ref_sites = bond_sites(begin_x, 0);
  for (size_t x = begin_x + 2; x < end_x; ++x) {
    for (size_t y = 0; y < Ly; ++y) {
      target_sites_interlayer_bond_set.push_back(bond_sites(x, y));
    }
  }

  std::array<Tensor, 4> sc_phys_ops_a = {ops.bupcF, ops.Fbdnc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_b = {ops.bdnc, ops.bupc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_c = {ops.bupcF, ops.Fbdnc, ops.bdna, ops.bupa};
  std::array<Tensor, 4> sc_phys_ops_d = {ops.bdnc, ops.bupc, ops.bdna, ops.bupa};
  std::array<Tensor, 4> sc_phys_ops_e = {ops.bupcF, ops.bupc, ops.bupaF, ops.bupa};
  std::array<Tensor, 4> sc_phys_ops_f = {ops.bdnc, ops.Fbdnc, ops.bdna, ops.Fbdna};

  struct SCTask {
    const std::array<Tensor, 4> &phys_ops;
    const std::array<size_t, 2> &ref_sites;
    const std::vector<std::array<size_t, 2> > &target_sites_set;
    const char *label;
  };

  const SCTask sc_tasks[] = {
    {sc_phys_ops_a, ref_sites, target_sites_interlayer_bond_set, "scs_a"},
    {sc_phys_ops_b, ref_sites, target_sites_interlayer_bond_set, "scs_b"},
    {sc_phys_ops_c, ref_sites, target_sites_interlayer_bond_set, "scs_c"},
    {sc_phys_ops_d, ref_sites, target_sites_interlayer_bond_set, "scs_d"},
    {sc_phys_ops_e, ref_sites, target_sites_interlayer_bond_set, "sct_e"},
    {sc_phys_ops_f, ref_sites, target_sites_interlayer_bond_set, "sct_f"}
  };
  const int total_sc_tasks = sizeof(sc_tasks) / sizeof(sc_tasks[0]);

  auto run_measurements = [&](size_t bond_dim) {
    std::ostringstream oss;
    oss << "conventional_square"
        << "Jk" << Jk
        << "Jperp" << Jperp
        << "U" << U
        << "Lx" << Lx
        << "Ly" << Ly
        << "D" << bond_dim;
    const std::string file_postfix = oss.str();

    for (size_t idx = 0; idx < two_site_meas_ops.size(); ++idx) {
      if (idx % mpi_size == rank) {
        const auto &[label, op1, op2] = two_site_meas_ops[idx];
        std::cout << "[rank " << rank << "] D=" << bond_dim
            << " start measuring two-site correlation " << label << std::endl;
        auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_site, target_sites);
        DumpMeasuRes(measu_res, label + file_postfix);
        std::cout << "[rank " << rank << "] D=" << bond_dim
            << " complete measuring two-site correlation " << label << std::endl;
      }
    }

    std::vector<std::string> one_site_labels;
    one_site_labels.reserve(one_site_base_labels.size());
    for (const auto &base_label : one_site_base_labels) {
      one_site_labels.emplace_back(base_label + file_postfix);
    }

    if ((two_site_meas_ops.size()) % mpi_size == rank) {
      std::cout << "[rank " << rank << "] D=" << bond_dim
          << " start measuring one-site observables" << std::endl;
      MeasureOneSiteOp(mps, mps_path, one_site_ops, even_sites, one_site_labels);
      std::cout << "[rank " << rank << "] D=" << bond_dim
          << " complete measuring one-site observables" << std::endl;
    }

    if ((two_site_meas_ops.size() + 1) % mpi_size == rank) {
      std::cout << "[rank " << rank << "] D=" << bond_dim
          << " start measuring single-particle correlations" << std::endl;
      auto sp_up_a = MeasureTwoSiteOpGroupInKondoLattice(mps,
                                                         mps_path,
                                                         ops.bupcF,
                                                         ops.bupa,
                                                         ref_site,
                                                         ops.f);
      DumpMeasuRes(sp_up_a, std::string("cup_dag_cup") + file_postfix);

      auto sp_up_b = MeasureTwoSiteOpGroupInKondoLattice(mps,
                                                         mps_path,
                                                         TenElemT(-1.0) * ops.bupaF,
                                                         ops.bupc,
                                                         ref_site,
                                                         ops.f);
      DumpMeasuRes(sp_up_b, std::string("cup_cup_dag") + file_postfix);

      auto sp_dn_a = MeasureTwoSiteOpGroupInKondoLattice(mps,
                                                         mps_path,
                                                         ops.bdnc,
                                                         ops.Fbdna,
                                                         ref_site,
                                                         ops.f);
      DumpMeasuRes(sp_dn_a, std::string("cdown_dag_cdown") + file_postfix);

      auto sp_dn_b = MeasureTwoSiteOpGroupInKondoLattice(mps,
                                                         mps_path,
                                                         TenElemT(-1.0) * ops.bdna,
                                                         ops.Fbdnc,
                                                         ref_site,
                                                         ops.f);
      DumpMeasuRes(sp_dn_b, std::string("cdown_cdown_dag") + file_postfix);

      std::cout << "[rank " << rank << "] D=" << bond_dim
          << " complete measuring single-particle correlations" << std::endl;
    }

    for (int idx = (rank + mpi_size * 5 - static_cast<int>(two_site_meas_ops.size()) - 1) % mpi_size;
         idx < total_sc_tasks;
         idx += mpi_size) {
      auto measu_res =
          MeasureFourSiteOpGroupInKondoLattice(mps,
                                               mps_path,
                                               sc_tasks[idx].phys_ops,
                                               sc_tasks[idx].ref_sites,
                                               sc_tasks[idx].target_sites_set,
                                               ops.f);
      std::cout << "[rank " << rank << "] D=" << bond_dim
          << " start measuring superconducting correlation " << sc_tasks[idx].label << std::endl;
      DumpMeasuRes(measu_res, std::string(sc_tasks[idx].label) + file_postfix);
      std::cout << "[rank " << rank << "] D=" << bond_dim
          << " complete measuring superconducting correlation " << sc_tasks[idx].label << std::endl;
    }
    std::cout << "[rank " << rank << "] D=" << bond_dim
        << " complete measuring superconducting correlations" << std::endl;
  };

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
      params.noise
    );
    auto e0 = qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);
    MPI_Barrier(comm);
    run_measurements(params.Dmax[i]);
  }
  if (rank == 0) {
    endTime = clock();
    cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }

  MPI_Finalize();
  return 0;
}
