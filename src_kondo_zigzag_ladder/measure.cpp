//
// Created by 王昊昕 on 25/6/2025.
//


#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "./params_case.h"
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"
#include "../src_tj_double_layer_single_orbital_2d/my_measure.h"

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
  //additional optional argument for set the MPS path
  std::string mps_path = kMpsPath;
  if (argc > 2) {
    mps_path = argv[2];
    std::cout << "Set MPS path as " << mps_path << std::endl;
  }

  size_t Lx = params.Lx; // Lx should be even number, for N/4 should on electron site for measure
  size_t Ly = params.Ly;
  double t = params.t, Jk = params.JK, U = params.U;
  double t2 = params.t2;
  // Must match the VMPS program (`src_kondo_zigzag_ladder/vmps.cpp`):
  // - each geometric electron site expands to two MPS sites: even = itinerant, odd = localized
  const size_t N = 2 * Ly * Lx;
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

  #ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
  #endif

  // Reference sites for measurements:
  // - itinerant electrons live on even sites
  // - localized spins live on odd sites
  size_t ref_elec = N / 4;
  if (ref_elec % 2 == 1) { ref_elec = ref_elec > 0 ? ref_elec - 1 : 0; } // force even
  const size_t ref_loc = std::min(ref_elec + 1, N - 1);                  // paired odd site

  std::vector<size_t> elec_targets;
  for (size_t i = ref_elec + 2; i < N; i += 2) elec_targets.push_back(i);
  std::vector<size_t> loc_targets;
  for (size_t i = ref_loc + 2; i < N; i += 2) loc_targets.push_back(i);

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
  // Keep the same filename postfix convention as `vmps.cpp`.
  oss << "t2" << t2 << "Jk" << Jk << "U" << U << "Ly" << Ly << "Lx" << Lx << "D" << params.Dmax.back();
  std::string file_postfix = oss.str();

  // Simple MPI scheduling assumption:
  // - each "measurement job" (a full MeasureTwoSiteOpGroup / MeasureOneSiteOp call) costs roughly the same
  // Then a global round-robin over jobs gives decent load balance.
  size_t job_idx = 0;
  auto do_job = [&](auto &&job) {
    if ((job_idx % static_cast<size_t>(mpi_size)) == static_cast<size_t>(rank)) {
      job();
    }
    ++job_idx;
  };

  // Two-site correlation measurements
  using OpT = Tensor;
  std::vector<std::tuple<std::string, const OpT &, const OpT &>> two_site_meas_ops = {
      {"szsz", hubbard_ops.sz, hubbard_ops.sz},
      {"spsm", hubbard_ops.sp, hubbard_ops.sm},
      {"smsp", hubbard_ops.sm, hubbard_ops.sp},
      {"nfnf", hubbard_ops.nf, hubbard_ops.nf}
  };
  for (const auto &[label, op1, op2] : two_site_meas_ops) {
    do_job([&]() {
      auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets);
      DumpMeasuRes(measu_res, label + file_postfix);
      std::cout << "Measured two-site correlation" + label << std::endl;
    });
  }

  // Localized-spin two-site correlations (odd sites)
  std::vector<std::tuple<std::string, const OpT &, const OpT &>> loc_two_site_meas_ops = {
      {"lszsz", local_spin_ops.sz, local_spin_ops.sz},
      {"lspsm", local_spin_ops.sp, local_spin_ops.sm},
      {"lsmsp", local_spin_ops.sm, local_spin_ops.sp}
  };
  for (const auto &[label, op1, op2] : loc_two_site_meas_ops) {
    do_job([&]() {
      auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets);
      DumpMeasuRes(measu_res, label + file_postfix);
      std::cout << "Measured localized-spin two-site correlation " + label << std::endl;
    });
  }

  // One-site local measurements on all even sites (extended electrons)
  std::vector<size_t> even_sites;
  for (size_t i = 0; i < N; i += 2) even_sites.push_back(i);
  std::vector<size_t> odd_sites;
  for (size_t i = 1; i < N; i += 2) odd_sites.push_back(i);

  std::vector<QLTensor<TenElemT, QNT>> one_site_ops = {hubbard_ops.sz, hubbard_ops.nf};
  std::vector<std::string> one_site_labels = {"sz_local" + file_postfix, "nf_local" + file_postfix};

  do_job([&]() {
    MeasureOneSiteOp(mps, mps_path, one_site_ops, even_sites, one_site_labels);
    std::cout << "Measured one-site correlation" << std::endl;
  });

  // One-site localized-spin measurement on all odd sites
  const std::vector<QLTensor<TenElemT, QNT>> loc_one_site_ops = {local_spin_ops.sz};
  const std::vector<std::string> loc_one_site_labels = {"lsz_local" + file_postfix};
  do_job([&]() {
    MeasureOneSiteOp(mps, mps_path, loc_one_site_ops, odd_sites, loc_one_site_labels);
    std::cout << "Measured localized-spin one-site observable" << std::endl;
  });


  // SC single-pair correlation measurements
  // NOTE: The following block measures interlayer SC, and may not work for general Ly.
#if 0
  std::vector<std::array<size_t, 2>>
      target_sites_diagonal_set;// a special case that do not need include the insertion operator
  std::vector<std::array<size_t, 2>>
      target_sites_horizontal_set;
  target_sites_diagonal_set.reserve(Lx);
  target_sites_horizontal_set.reserve(Lx);

  size_t begin_x = Lx / 4;
  if (begin_x % 2 == 0) {
    begin_x += 1;
  }
  size_t end_x = Lx - 1;
  size_t Ly = 4;
  size_t site1_a = begin_x * Ly;
  size_t site1_b = begin_x * Ly + 2; //a-b: diagonal bond
  size_t site1_c = begin_x * Ly + 4; //a-c: horizontal bond
  for (size_t x = begin_x + 2; x < end_x; x++) {
    size_t site2_a = x * Ly;
    size_t site2_b = x * Ly + 2;
    size_t site2_c = x * Ly + 4; // a-c: horizontal or vertical bond
    target_sites_diagonal_set.push_back({site2_a, site2_b});
    target_sites_horizontal_set.push_back({site2_a, site2_c});
  }
  std::array<Tensor, 4> sc_phys_ops_a = {ops.bupcF, ops.Fbdnc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_b = {ops.bdnc, ops.bupc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_c = {ops.bupcF, ops.Fbdnc, ops.bdna, ops.bupa};
  std::array<Tensor, 4> sc_phys_ops_d = {ops.bdnc, ops.bupc, ops.bdna, ops.bupa}; // a==d if spin inversion symmetry
  std::array<Tensor, 4>
      sc_phys_ops_e = {ops.bupcF, ops.bupc, ops.bupaF, ops.bupa}; // Triplet < up^dag(i) up^dag(j) up(k) up(l) >
  std::array<Tensor, 4>
      sc_phys_ops_f = {ops.bdnc, ops.Fbdnc, ops.bdna, ops.Fbdna}; // Triplet < down^dag(i) down^dag(j) down(k) down(l) >
  std::array<Tensor, 4> sc_inst_ops = {ops.f, ops.id, ops.f};

  struct Task {
    const array<Tensor, 4> &phys_ops;
    const std::array<size_t, 2> &ref_sites;
    const std::vector<std::array<size_t, 2>> &target_sites_set;
    string label;
  };
  std::array<size_t, 2> ref_diag_sites = {site1_a, site1_b};
  std::array<size_t, 2> ref_hori_sites = {site1_a, site1_c};

  Task tasks[] = {
      {sc_phys_ops_a, ref_diag_sites, target_sites_diagonal_set, "scs_diag_a"},
      {sc_phys_ops_b, ref_diag_sites, target_sites_diagonal_set, "scs_diag_b"},
      {sc_phys_ops_c, ref_diag_sites, target_sites_diagonal_set, "scs_diag_c"},
      {sc_phys_ops_d, ref_diag_sites, target_sites_diagonal_set, "scs_diag_d"},
      {sc_phys_ops_e, ref_diag_sites, target_sites_diagonal_set, "sct_diag_e"},
      {sc_phys_ops_f, ref_diag_sites, target_sites_diagonal_set, "sct_diag_f"},
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
#endif

  MPI_Finalize();
  return 0;
}