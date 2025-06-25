//
// Created by 王昊昕 on 25/6/2025.
//


#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo/kondo_hilbert_space.h"
#include "./params_case.h"
#include "../src_single_orbital/myutil.h"

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
  size_t Lx = params.Lx; // L should be even number, for N/4 should on electron site for measure
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

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  auto &ops = hubbard_ops;
  SpinOneHalfOperatorsU1U1 local_spin_ops;
  auto f = hubbard_ops.f;

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  size_t ref_site = N / 4;
  std::vector<size_t> target_sites;
  for (size_t i = ref_site + 2; i < N; i += 2) {
    target_sites.push_back(i);
  }

  mps.Load();
  mps.Centralize(0);
  mps.Dump();

  std::ostringstream oss;
  oss << "t2" << t2 << "Jk" << Jk << "U" << U << "Lx" << Lx << "D" << params.Dmax.back();
  std::string file_postfix = oss.str();

  // Two-site correlation measurements
  using OpT = Tensor;
  std::vector<std::tuple<std::string, const OpT &, const OpT &>> meas_ops = {
      {"szsz", hubbard_ops.sz, hubbard_ops.sz},
      {"spsm", hubbard_ops.sp, hubbard_ops.sm},
      {"smsp", hubbard_ops.sm, hubbard_ops.sp},
      {"nfnf", hubbard_ops.nf, hubbard_ops.nf}
  };
  for (size_t i = 0; i < meas_ops.size(); ++i) {
    if (i % mpi_size == rank) {
      const auto &[label, op1, op2] = meas_ops[i];
      auto measu_res = MeasureTwoSiteOpGroup(mps, kMpsPath, op1, op2, ref_site, target_sites);
      DumpMeasuRes(measu_res, label + file_postfix);
    }
  }

  // One-site local measurements on all even sites (extended electrons)
  std::vector<size_t> even_sites;
  for (size_t i = 0; i < N; i += 2) even_sites.push_back(i);

  std::vector<QLTensor<TenElemT, QNT>> one_site_ops = {hubbard_ops.sz, hubbard_ops.nf};
  std::vector<std::string> one_site_labels = {"sz_local", "nf_local"};

  if ((meas_ops.size()) % mpi_size == rank) {
    auto one_site_measu = MeasureOneSiteOp(mps, kMpsPath, one_site_ops, even_sites, one_site_labels);
    for (size_t i = 0; i < one_site_labels.size(); ++i) {
      DumpMeasuRes(one_site_measu[i], one_site_labels[i] + file_postfix);
    }
  }


  // SC single-pair correlation measurements
  vector<vector<size_t>>
      four_point_diagonal_bond_sites_setF;//actually no used for new measure API. a special case that do not need include the insertion operator

  std::vector<std::array<size_t, 2>>
      target_sites_diagonal_set;// a special case that do not need include the insertion operator

  four_point_diagonal_bond_sites_setF.reserve(Lx);
  target_sites_diagonal_set.reserve(Lx);

  size_t begin_x = Lx / 4;
  size_t end_x = Lx - 1;
  size_t Ly = 2;
  size_t site1_a = begin_x * Ly;
  size_t site1_b = begin_x * Ly + 1;
  if (site1_b < site1_a) {
    std::swap(site1_a, site1_b);
  }
  for (size_t x = begin_x + 2; x < end_x; x++) {
    size_t site2_a = x * Ly;
    size_t site2_b = x * Ly + 1;
    four_point_diagonal_bond_sites_setF.push_back({site1_a, site1_b, site2_a, site2_b});
    target_sites_diagonal_set.push_back({site2_a, site2_b});
  }
  std::array<Tensor, 4> sc_phys_ops_a = {ops.bupcF, ops.Fbdnc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_b = {ops.bdnc, ops.bupc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_c = {ops.bupcF, ops.Fbdnc, ops.bdna, ops.bupa};
  std::array<Tensor, 4> sc_phys_ops_d = {ops.bdnc, ops.bupc, ops.bdna, ops.bupa};
  std::array<Tensor, 4>
      sc_phys_ops_e = {ops.bupcF, ops.bupc, ops.bupaF, ops.bupa}; // Triplet < up^dag(i) up^dag(j) up(k) up(l) >
  std::array<Tensor, 4>
      sc_phys_ops_f = {ops.bdnc, ops.Fbdnc, ops.bdna, ops.Fbdna}; // Triplet < down^dag(i) down^dag(j) down(k) down(l) >
  std::array<Tensor, 4> sc_inst_ops = {ops.f, ops.id, ops.f};

  size_t sc_corr_set_size = four_point_diagonal_bond_sites_setF.size();
  std::vector<std::array<Tensor, 4>> sc_phys_ops_set_a(sc_corr_set_size, sc_phys_ops_a);
  std::vector<std::array<Tensor, 4>> sc_phys_ops_set_b(sc_corr_set_size, sc_phys_ops_b);
  std::vector<std::array<Tensor, 4>> sc_phys_ops_set_c(sc_corr_set_size, sc_phys_ops_c);
  std::vector<std::array<Tensor, 4>> sc_phys_ops_set_d(sc_corr_set_size, sc_phys_ops_d);
  std::vector<std::array<Tensor, 4>> sc_phys_ops_set_e(sc_corr_set_size, sc_phys_ops_e);
  std::vector<std::array<Tensor, 4>> sc_phys_ops_set_f(sc_corr_set_size, sc_phys_ops_f);

  struct Task {
    const vector<array<Tensor, 4>> &phys_ops_set;
    const vector<vector<size_t>> &bond_sites_set;
    string label;
  };
  Task tasks[] = {
      {sc_phys_ops_set_a, four_point_diagonal_bond_sites_setF, "scsyya"},
      {sc_phys_ops_set_b, four_point_diagonal_bond_sites_setF, "scsyyb"},
      {sc_phys_ops_set_c, four_point_diagonal_bond_sites_setF, "scsyyc"},
      {sc_phys_ops_set_d, four_point_diagonal_bond_sites_setF, "scsyyd"},
      {sc_phys_ops_set_e, four_point_diagonal_bond_sites_setF, "sctyye"},
      {sc_phys_ops_set_f, four_point_diagonal_bond_sites_setF, "sctyyf"},
  };

  int total_tasks = sizeof(tasks) / sizeof(tasks[0]);

  std::array<size_t, 2> ref_sites = {site1_a, site1_b};
  for (int i = rank; i < total_tasks; i += mpi_size) {
    // Each rank processes its assigned tasks
    auto measu_res =
        MeasureFourSiteOpGroup(mps, kMpsPath, tasks[i].phys_ops_set.front(), ref_sites, target_sites_diagonal_set);
    DumpMeasuRes(measu_res, tasks[i].label + file_postfix);

  }

  MPI_Finalize();
  return 0;
}