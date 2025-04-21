//
// Created by 王昊昕 on 19/4/2025.
//


#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "kondo_hilbert_space.h"
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
  size_t L = params.L; // L should be even number, for N/4 should on electron site for measure
  double t = params.t, Jk = params.JK, U = params.U;

  size_t N = 2 * L;

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
  SpinOneHalfOperatorsU1U1 local_spin_ops;
  for (size_t i = 0; i < N - 2; i = i + 2) {
    size_t site1 = i, site2 = i + 2;
    mpo_gen.AddTerm(-t, hubbard_ops.bupcF, site1, hubbard_ops.bupa, site2);
    mpo_gen.AddTerm(t, hubbard_ops.bupaF, site1, hubbard_ops.bupc, site2);
    mpo_gen.AddTerm(-t, hubbard_ops.bdnc, site1, hubbard_ops.Fbdna, site2);
    mpo_gen.AddTerm(t, hubbard_ops.bdna, site1, hubbard_ops.Fbdnc, site2);
//    mpo_gen.AddTerm(Jz, Sz, i + 1, Sz, i + 3);
  }

  for (size_t i = 0; i < N; i++) {
    mpo_gen.AddTerm(U, hubbard_ops.nf, i);
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

  std::vector<size_t> elec_labs(L);
  //electron quarter filling
  std::fill(elec_labs.begin(), elec_labs.begin() + L / 4, hubbard_site.spin_up);
  std::fill(elec_labs.begin() + L / 4, elec_labs.begin() + L / 2, hubbard_site.spin_down);
  std::fill(elec_labs.begin() + L / 2, elec_labs.end(), hubbard_site.empty);
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
    std::cout << "middle " << ee_list[L] << std::endl;
  }
  size_t ref_site = N / 4;
  std::vector<size_t> target_sites;
  for (size_t i = ref_site; i < N; i += 2) {
    target_sites.push_back(i);
  }

  std::string file_postfix = "Jk" + std::to_string(Jk) + "U" + std::to_string(U);
  if (rank % mpi_size == 0) {
    MeasuRes<TenElemT> measu_res = MeasureTwoSiteOpGroup(mps,
                                                         kMpsPath,
                                                         hubbard_ops.sz, hubbard_ops.sz,
                                                         ref_site, target_sites);
    DumpMeasuRes(measu_res, "szsz"  + file_postfix);
  }
  if (rank % mpi_size == 1) {
    MeasuRes<TenElemT> measu_res = MeasureTwoSiteOpGroup(mps,
                                                         kMpsPath,
                                                         hubbard_ops.sp, hubbard_ops.sm,
                                                         ref_site, target_sites);
    DumpMeasuRes(measu_res, "spsm"  + file_postfix);
  }
  if (rank % mpi_size == 2) {
    MeasuRes<TenElemT> measu_res = MeasureTwoSiteOpGroup(mps,
                                                         kMpsPath,
                                                         hubbard_ops.sm, hubbard_ops.sp,
                                                         ref_site, target_sites);
    DumpMeasuRes(measu_res, "smsp" + file_postfix);
  }

  MPI_Finalize();
  return 0;
}