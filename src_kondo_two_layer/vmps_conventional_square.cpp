//
// Created by Haoxin Wang on 3/7/2025.
//
/*
 * 2-layer 2-leg Kondo lattice model, in conventional square lattice.
 */


#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo_1D/kondo_hilbert_space.h"
#include "params_case.h"
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
  size_t Ly = 2;
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
    cout << "Lx = " << Ly << endl;
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
  for (size_t i = 0; i < N - 4 * Ly; i = i + 2) {
    size_t site1 = i, site2 = i + 4 * Ly;//4 for double layer times two orbital (localized & itinerate)
    std::vector<size_t> inst_op_idxs; //even sites between site1 and site2
    for (size_t j = site1 + 2; j < site2; j += 2) {
      inst_op_idxs.push_back(j);
    }
    mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f, inst_op_idxs);
    mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f, inst_op_idxs);
    mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f, inst_op_idxs);
    mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f, inst_op_idxs);
  }
  //hopping along y direction, assume Ly = 2
  for (size_t i = 0; i < N - 6; i += 4 * Ly) {
    size_t site1 = i, site2 = i + 4; // 0-th layer
    mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f, {site1 + 2});
    mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f, {site1 + 2});
    mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f, {site1 + 2});
    mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f, {site1 + 2});

    site1 = i + 2, site2 = i + 6;   // 1-th layer
    mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f, {site1 + 2});
    mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f, {site1 + 2});
    mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f, {site1 + 2});
    mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f, {site1 + 2});
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

  for (size_t i = 0; i < N - 1; i = i + 2) {
    mpo_gen.AddTerm(Jk, hubbard_ops.sz, i, local_spin_ops.sz, i + 1);
    mpo_gen.AddTerm(Jk / 2, hubbard_ops.sp, i, local_spin_ops.sm, i + 1);
    mpo_gen.AddTerm(Jk / 2, hubbard_ops.sm, i, local_spin_ops.sp, i + 1);
  }

  //inter layer AFM coupling
  for (size_t i = 1; i < N - 2; i = i + 4) { // for each unit cell
    mpo_gen.AddTerm(Jperp, local_spin_ops.sz, i, local_spin_ops.sz, i + 2);
    mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sp, i, local_spin_ops.sm, i + 2);
    mpo_gen.AddTerm(Jperp / 2, local_spin_ops.sm, i, local_spin_ops.sp, i + 2);
  }

  qlmps::MPO<Tensor> mpo = mpo_gen.Gen();

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  std::vector<size_t> elec_labs(2 * Ly * Lx);
  //electron quarter filling
  std::fill(elec_labs.begin(), elec_labs.begin() + Lx, hubbard_site.spin_up);
  std::fill(elec_labs.begin() + Lx, elec_labs.begin() + 2 * Lx, hubbard_site.spin_down);
  std::fill(elec_labs.begin() + 2 * Lx, elec_labs.end(), hubbard_site.empty);
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

//  if (rank == kMPIMasterRank) {
//    mps.Load(kMpsPath);
//    auto ee_list = mps.GetEntanglementEntropy(1);
//    std::copy(ee_list.begin(), ee_list.end(), std::ostream_iterator<double>(std::cout, " "));
//
//    std::cout << "\n";
//    std::cout << "middle " << ee_list[2 * Lx] << std::endl;
//mps.clear();
//  }

  // ******* Measurement ****** //
  size_t ref_site = N / 4;
  if (ref_site % 2 == 1) {
    ref_site += 1;
  }
  std::vector<size_t> target_sites;
  for (size_t i = ref_site + 2; i < N; i += 2) {
    target_sites.push_back(i);
  }
  std::string mps_path = kMpsPath;

  std::ostringstream oss;
  oss << "conventional_square" << "Jk" << Jk << "Jperp" << Jperp << "U" << U << "Lx" << Lx << "D" << params.Dmax.back();
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
      target_sites_interlayer_bond_set;// a special case that do not need include the insertion operator
//  std::vector<std::array<size_t, 2>>
//      target_sites_horizontal_set;
//  std::vector<std::array<size_t, 2>>
//      target_sites_vertical_set;
  target_sites_interlayer_bond_set.reserve(Lx);
//  target_sites_horizontal_set.reserve(Lx);
//  target_sites_vertical_set.reserve(Lx);

  size_t begin_x = Lx / 4;
  size_t end_x = Lx - 1;
  size_t effective_ly = 4 * Ly;
  size_t site1_a = begin_x * effective_ly; //0-th layer, ly = 0
  size_t site1_b = begin_x * effective_ly + 2; //1-th layer, ly = 1; a-b: interlayer bond
  std::array<size_t, 2> ref_sites = {site1_a, site1_b}; // interlayer pairing
//  std::array<size_t, 2> ref_hori_sites = {site1_a, site1_c};
//  std::array<size_t, 2> ref_vert_sites = {site1_a, site1_b};
  for (size_t x = begin_x + 2; x < end_x; x++) {
    // ly = 0
    size_t site2_a = x * effective_ly;
    size_t site2_b = x * effective_ly + 2;
    target_sites_interlayer_bond_set.push_back({site2_a, site2_b});

    // ly = 1
    site2_a = x * effective_ly + 4;
    site2_b = x * effective_ly + 6;
    target_sites_interlayer_bond_set.push_back({site2_a, site2_b});
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
      {sc_phys_ops_a, ref_sites, target_sites_interlayer_bond_set, "scs_a"},
      {sc_phys_ops_b, ref_sites, target_sites_interlayer_bond_set, "scs_b"},
      {sc_phys_ops_c, ref_sites, target_sites_interlayer_bond_set, "scs_c"},
      {sc_phys_ops_d, ref_sites, target_sites_interlayer_bond_set, "scs_d"},
      {sc_phys_ops_e, ref_sites, target_sites_interlayer_bond_set, "sct_e"},
      {sc_phys_ops_f, ref_sites, target_sites_interlayer_bond_set, "sct_f"}
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