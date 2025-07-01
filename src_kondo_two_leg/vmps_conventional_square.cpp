//
// Created by 王昊昕 on 19/4/2025.
//
/*
 * 2-leg Kondo lattice model, in convential square lattice.
 */


#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo/kondo_hilbert_space.h"
#include "./params_case.h"
#include "../src_single_orbital/myutil.h"
#include "../src_single_orbital/my_measure.h"

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
  double t = params.t, Jk = params.JK, U = params.U;
  double t2 = params.t2;
  size_t N = 4 * Lx;
  /*** Print the model parameter Info ***/
  if (rank == 0) {
    cout << "Lx = " << Lx << endl;
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
    if (i % 2 == 0) pb_set[i] = pb_outE;   // even site is extended electron
    if (i % 2 == 1) pb_set[i] = pb_outL;   // odd site is localized electron
  }
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(pb_set);
  auto mpo_gen = MPOGenerator<TenElemT, QNT>(sites);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  auto &ops = hubbard_ops;
  SpinOneHalfOperatorsU1U1 local_spin_ops;
  auto f = hubbard_ops.f;
  //hopping along x direction
  for (size_t i = 0; i < N - 4; i = i + 2) {
    size_t site1 = i, site2 = i + 4;
    mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f, {site1 + 2});
    mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f, {site1 + 2});
    mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f, {site1 + 2});
    mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f, {site1 + 2});
  }

  for (size_t i = 0; i < N - 2; i += 4) {
    size_t site1 = i, site2 = i + 2;
    mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2);
    mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2);
    mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2);
    mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2);
  }

//  for (size_t i = second_leg_start_site; i < N - 2; i = i + di_for_t2) {
//    size_t site1 = i, site2 = i + 2;
//    mpo_gen.AddTerm(-t2, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2);
//    mpo_gen.AddTerm(t2, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2);
//    mpo_gen.AddTerm(-t2, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2);
//    mpo_gen.AddTerm(t2, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2);
//  }

  for (size_t i = 0; i < N; i += 2) {
    mpo_gen.AddTerm(U, hubbard_ops.nupndn, i);
  }

  for (size_t i = 0; i < N; i = i + 2) {
    mpo_gen.AddTerm(Jk, hubbard_ops.sz, i, local_spin_ops.sz, i + 1);
    mpo_gen.AddTerm(Jk / 2, hubbard_ops.sp, i, local_spin_ops.sm, i + 1);
    mpo_gen.AddTerm(Jk / 2, hubbard_ops.sm, i, local_spin_ops.sp, i + 1);
  }

  qlmps::MPO<Tensor> mpo = mpo_gen.Gen();

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  std::vector<size_t> elec_labs(2 * Lx);
  //electron quarter filling
  std::fill(elec_labs.begin(), elec_labs.begin() + Lx / 2, hubbard_site.spin_up);
  std::fill(elec_labs.begin() + Lx / 2, elec_labs.begin() + Lx, hubbard_site.spin_down);
  std::fill(elec_labs.begin() + Lx, elec_labs.end(), hubbard_site.empty);
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(elec_labs.begin(), elec_labs.end(), g);

  std::vector<size_t> stat_labs(N);
  for (size_t i = 0; i < N; i = i + 2) {
    stat_labs[i] = elec_labs[i / 2];
  }
  int sz_lab = 0;
  for (size_t i = 1; i < N; i = i + 2) {
    stat_labs[i] = sz_lab % 2;
    sz_lab++;
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

  for (size_t i = 0; i < params.Dmax.size(); i++) {
    if (rank == 0) {
      std::cout << "D_max = " << params.Dmax[i] << std::endl;
    }
    qlmps::FiniteVMPSSweepParams sweep_params(
        params.Sweeps,
        params.Dmin, params.Dmax[i], params.CutOff,
        qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter),
        params.noise
    );
    auto e0 = qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);
  }
  if (rank == 0) {
    endTime = clock();
    cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }

  if (rank == kMPIMasterRank) {
    mps.Load(kMpsPath);
    auto ee_list = mps.GetEntanglementEntropy(1);
    std::copy(ee_list.begin(), ee_list.end(), std::ostream_iterator<double>(std::cout, " "));

    std::cout << "\n";
    std::cout << "middle " << ee_list[2 * Lx] << std::endl;
  }
  size_t ref_site = N / 4;
  std::vector<size_t> target_sites;
  for (size_t i = ref_site + 2; i < N; i += 2) {
    target_sites.push_back(i);
  }
  std::string mps_path = kMpsPath;
  if (rank == 0) {
    mps.Load(mps_path);
    std::cout << "Success load mps into memory." << std::endl;
    mps.Centralize(0);
    std::cout << "Centralize mps to 0 site." << std::endl;
    mps.Dump(mps_path, true);
    std::cout << "Dump mps into disk" << std::endl;
  }
  MPI_Barrier(comm);

  std::ostringstream oss;
  oss << "conventional_square" << "Jk" << Jk << "U" << U << "Lx" << Lx << "D" << params.Dmax.back();
  std::string file_postfix = oss.str();

  // Two-site correlation measurements
  using OpT = Tensor;
  std::vector<std::tuple<std::string, const OpT &, const OpT &>> two_site_meas_ops = {
      {"szsz", hubbard_ops.sz, hubbard_ops.sz},
      {"spsm", hubbard_ops.sp, hubbard_ops.sm},
      {"smsp", hubbard_ops.sm, hubbard_ops.sp},
      {"nfnf", hubbard_ops.nf, hubbard_ops.nf}
  };
  for (size_t i = 0; i < two_site_meas_ops.size(); ++i) {
    if (i % mpi_size == rank) {
      const auto &[label, op1, op2] = two_site_meas_ops[i];
      auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_site, target_sites);
      DumpMeasuRes(measu_res, label + file_postfix);
      std::cout << "Measured two-site correlation" + label << std::endl;
    }
  }

  // One-site local measurements on all even sites (extended electrons)
  std::vector<size_t> even_sites;
  for (size_t i = 0; i < N; i += 2) even_sites.push_back(i);

  std::vector<QLTensor<TenElemT, QNT>> one_site_ops = {hubbard_ops.sz, hubbard_ops.nf};
  std::vector<std::string> one_site_labels = {"sz_local" + file_postfix, "nf_local" + file_postfix};

  if ((two_site_meas_ops.size()) % mpi_size == rank) {
    MeasureOneSiteOp(mps, mps_path, one_site_ops, even_sites, one_site_labels);
    std::cout << "Measured one-site correlation" << std::endl;
  }


  // SC single-pair correlation measurements
  std::vector<std::array<size_t, 2>>
      target_sites_diagonal_set;// a special case that do not need include the insertion operator
  std::vector<std::array<size_t, 2>>
      target_sites_horizontal_set;
  std::vector<std::array<size_t, 2>>
      target_sites_vertical_set;
  target_sites_diagonal_set.reserve(Lx);
  target_sites_horizontal_set.reserve(Lx);
  target_sites_vertical_set.reserve(Lx);

  size_t begin_x = Lx / 4;
  size_t end_x = Lx - 1;
  size_t Ly = 4;
  size_t site1_a = begin_x * Ly;
  size_t site1_b = begin_x * Ly + 2; //a-b: vertical bond
  size_t site1_c = begin_x * Ly + 4; //a-c: horizontal bond
  //b-c: diagonal bond
  std::array<size_t, 2> ref_diag_sites = {site1_b, site1_c};
  std::array<size_t, 2> ref_hori_sites = {site1_a, site1_c};
  std::array<size_t, 2> ref_vert_sites = {site1_a, site1_b};
  for (size_t x = begin_x + 2; x < end_x; x++) {
    size_t site2_a = x * Ly;
    size_t site2_b = x * Ly + 2;
    size_t site2_c = x * Ly + 4;
    target_sites_diagonal_set.push_back({site2_b, site2_c});
    target_sites_vertical_set.push_back({site2_a, site2_b});
    target_sites_horizontal_set.push_back({site2_a, site2_c});
  }
  std::array<Tensor, 4> sc_phys_ops_a = {ops.bupcF, ops.Fbdnc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_b = {ops.bdnc, ops.bupc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_c = {ops.bupcF, ops.Fbdnc, ops.bdna, ops.bupa};
  std::array<Tensor, 4> sc_phys_ops_d = {ops.bdnc, ops.bupc, ops.bdna, ops.bupa};
  std::array<Tensor, 4>
      sc_phys_ops_e = {ops.bupcF, ops.bupc, ops.bupaF, ops.bupa}; // Triplet < up^dag(i) up^dag(j) up(k) up(l) >
  std::array<Tensor, 4>
      sc_phys_ops_f = {ops.bdnc, ops.Fbdnc, ops.bdna, ops.Fbdna}; // Triplet < down^dag(i) down^dag(j) down(k) down(l) >

  struct Task {
    const array<Tensor, 4> &phys_ops;
    const std::array<size_t, 2> &ref_sites;
    const std::vector<std::array<size_t, 2>> &target_sites_set;
    string label;
  };

  Task tasks[] = {
      {sc_phys_ops_a, ref_diag_sites, target_sites_diagonal_set, "scs_diag_a"},
      {sc_phys_ops_b, ref_diag_sites, target_sites_diagonal_set, "scs_diag_b"},
      {sc_phys_ops_c, ref_diag_sites, target_sites_diagonal_set, "scs_diag_c"},
      {sc_phys_ops_d, ref_diag_sites, target_sites_diagonal_set, "scs_diag_d"},
      {sc_phys_ops_e, ref_diag_sites, target_sites_diagonal_set, "sct_diag_e"},
      {sc_phys_ops_f, ref_diag_sites, target_sites_diagonal_set, "sct_diag_f"},
      {sc_phys_ops_a, ref_vert_sites, target_sites_vertical_set, "scs_vert_a"},
      {sc_phys_ops_b, ref_vert_sites, target_sites_vertical_set, "scs_vert_b"},
      {sc_phys_ops_c, ref_vert_sites, target_sites_vertical_set, "scs_vert_c"},
      {sc_phys_ops_d, ref_vert_sites, target_sites_vertical_set, "scs_vert_d"},
      {sc_phys_ops_e, ref_vert_sites, target_sites_vertical_set, "sct_vert_e"},
      {sc_phys_ops_f, ref_vert_sites, target_sites_vertical_set, "sct_vert_f"},
      {sc_phys_ops_a, ref_hori_sites, target_sites_horizontal_set, "scs_hori_a"},
      {sc_phys_ops_b, ref_hori_sites, target_sites_horizontal_set, "scs_hori_b"},
      {sc_phys_ops_c, ref_hori_sites, target_sites_horizontal_set, "scs_hori_c"},
      {sc_phys_ops_d, ref_hori_sites, target_sites_horizontal_set, "scs_hori_d"},
      {sc_phys_ops_e, ref_hori_sites, target_sites_horizontal_set, "sct_hori_e"},
      {sc_phys_ops_f, ref_hori_sites, target_sites_horizontal_set, "sct_hori_f"}
  };

  int total_tasks = sizeof(tasks) / sizeof(tasks[0]);

  for (int i = (rank + mpi_size * 5 - two_site_meas_ops.size() - 1) % mpi_size; i < total_tasks; i += mpi_size) {
    // Each rank processes its assigned tasks
    auto measu_res =
        MeasureFourSiteOpGroupInKondoLattice(mps,
                                             mps_path,
                                             tasks[i].phys_ops,
                                             tasks[i].ref_sites,
                                             tasks[i].target_sites_set,
                                             ops.f);
    DumpMeasuRes(measu_res, tasks[i].label + file_postfix);
    std::cout << "Measured SC correlation" << std::endl;
  }

  MPI_Finalize();
  return 0;
}