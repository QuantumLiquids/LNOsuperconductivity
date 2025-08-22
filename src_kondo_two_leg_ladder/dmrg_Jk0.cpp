//
// Created by 王昊昕 on 19/4/2025.
//


#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "./params_case.h"
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"

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
  double t = params.t, U = params.U;
  double t2 = params.t2;
  size_t N = 2 * Lx;

  clock_t startTime, endTime;
  startTime = clock();

  std::vector<IndexT> pb_set = std::vector<IndexT>(N);
  for (size_t i = 0; i < N; ++i) {
    pb_set[i] = pb_outE;   // even site is extended electron
  }
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(pb_set);
  auto mpo_gen = MPOGenerator<TenElemT, QNT>(sites);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  auto f = hubbard_ops.f;
  for (size_t i = 0; i < N - 2; i = i + 1) {
    size_t site1 = i, site2 = i + 2;
    mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f);
    mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f);
    mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f);
    mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f);
  }
  size_t di_for_t2(0);
  size_t second_leg_start_site(0);
  if (params.Geometry == "OBC") {
    di_for_t2 = 4;
    second_leg_start_site = 3;
  } else {
    di_for_t2 = 2;
    second_leg_start_site = 1;
  }

  for (size_t i = 0; i < N - 3; i += di_for_t2) {
    size_t site1 = i, site2 = i + 3;
    mpo_gen.AddTerm(-t2, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2, f);
    mpo_gen.AddTerm(t2, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2, f);
    mpo_gen.AddTerm(-t2, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2, f);
    mpo_gen.AddTerm(t2, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2, f);
  }

  for (size_t i = second_leg_start_site; i < N - 1; i = i + di_for_t2) {
    size_t site1 = i, site2 = i + 1;
    mpo_gen.AddTerm(-t2, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2);
    mpo_gen.AddTerm(t2, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2);
    mpo_gen.AddTerm(-t2, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2);
    mpo_gen.AddTerm(t2, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2);
  }

  for (size_t i = 0; i < N; i += 1) {
    mpo_gen.AddTerm(U, hubbard_ops.nupndn, i);
  }

  auto mpo = mpo_gen.GenMatReprMPO();

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

  if (IsPathExist(kMpsPath)) {
    if (N == GetNumofMps()) {
      cout << "The number of mps files is consistent with mps size." << endl;
      cout << "Directly use mps from files." << endl;
    } else {
      qlmps::DirectStateInitMps(mps, elec_labs);
      cout << "Initial mps as direct product state." << endl;
      if (rank == 0)
        mps.Dump(kMpsPath, true);
    }
  } else {
    qlmps::DirectStateInitMps(mps, elec_labs);
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
        qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter)
    );
    auto e0 = qlmps::FiniteDMRG(mps, mpo, sweep_params, comm);
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
    std::cout << "middle " << ee_list[Lx] << std::endl;
  }
  size_t ref_site = N / 4;
  std::vector<size_t> target_sites;
  for (size_t i = ref_site + 1; i < N; i += 1) {
    target_sites.push_back(i);
  }

  std::ostringstream oss;
  oss << "t2" << t2 << "Jk" << 0 << "U" << U;
  std::string file_postfix = oss.str();

  if (0 % mpi_size == rank) {
    MeasuRes<TenElemT> measu_res = MeasureTwoSiteOpGroup(mps,
                                                         kMpsPath,
                                                         hubbard_ops.sz, hubbard_ops.sz,
                                                         ref_site, target_sites);
    DumpMeasuRes(measu_res, "szsz" + file_postfix);
  }
  if (1 % mpi_size == rank) {
    MeasuRes<TenElemT> measu_res = MeasureTwoSiteOpGroup(mps,
                                                         kMpsPath,
                                                         hubbard_ops.sp, hubbard_ops.sm,
                                                         ref_site, target_sites);
    DumpMeasuRes(measu_res, "spsm" + file_postfix);
  }
  if (2 % mpi_size == rank) {
    MeasuRes<TenElemT> measu_res = MeasureTwoSiteOpGroup(mps,
                                                         kMpsPath,
                                                         hubbard_ops.sm, hubbard_ops.sp,
                                                         ref_site, target_sites);
    DumpMeasuRes(measu_res, "smsp" + file_postfix);
  }
  if (3 % mpi_size == rank) {
    MeasuRes<TenElemT> measu_res = MeasureTwoSiteOpGroup(mps,
                                                         kMpsPath,
                                                         hubbard_ops.nf, hubbard_ops.nf,
                                                         ref_site, target_sites);
    DumpMeasuRes(measu_res, "nn" + file_postfix);
  }
  if (4 % mpi_size == rank) {
    MeasureOneSiteOp(mps, kMpsPath, hubbard_ops.nf, "nf" + file_postfix);
  }

  MPI_Finalize();
  return 0;
}