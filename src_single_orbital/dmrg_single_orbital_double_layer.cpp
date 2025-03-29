// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-04-1
*
* Description: Single processor DMRG for t-J model with t_2 and J_2. DMRG style code
*/


#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include "tJ_type_hilbert_space.h"
#include "tJ_operators.h"
#include "params_case.h"
#include "myutil.h"
#include "double_layer_squarelattice.h"
#include "double_layer_tJmodel.h"
#include <random>

using namespace std;

int main(int argc, char *argv[]) {
  using namespace qlmps;
  using namespace qlten;
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);

  CaseParams params(argv[1]);
  size_t Lx = params.Lx, Ly = params.Ly;
  size_t N = 2 * Lx * Ly;
  DoubleLayertJModelParamters model_params(params);
  if (rank == 0) {
    model_params.Print();
  }
  clock_t startTime, endTime;
  startTime = clock();
  OperatorInitial();
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);
  std::vector<size_t> input_D_set;
  bool has_bond_dimension_parameter = ParserBondDimension(
      argc, argv,
      input_D_set);

  size_t DMRG_time = input_D_set.size();
  std::vector<size_t> MaxLanczIterSet(DMRG_time);
  if (has_bond_dimension_parameter) {
    MaxLanczIterSet.back() = params.MaxLanczIter;
    if (DMRG_time > 1) {
      size_t MaxLanczIterSetSpace;
      MaxLanczIterSet[0] = 3;
      MaxLanczIterSetSpace = (params.MaxLanczIter - 3) / (DMRG_time - 1);
      std::cout << "Setting MaxLanczIter as : [" << MaxLanczIterSet[0] << ", ";
      for (size_t i = 1; i < DMRG_time - 1; i++) {
        MaxLanczIterSet[i] = MaxLanczIterSet[i - 1] + MaxLanczIterSetSpace;
        std::cout << MaxLanczIterSet[i] << ", ";
      }
      std::cout << MaxLanczIterSet.back() << "]" << std::endl;
    } else {
      std::cout << "Setting MaxLanczIter as : [" << MaxLanczIterSet[0] << "]" << std::endl;
    }
  }

  qlmps::MPOGenerator<TenElemT, QNT> mpo_gen(sites, qn0);

  if (params.Geometry == "Cylinder") {
    if (Ly < 3) {
      std::cout << "Cylinder is not well defined for Ly = " << Ly << std::endl;
      exit(1);
    }
    DoubleLayerSquareCylinder lattice(Ly, Lx);
    cout << "lattice construct" << std::endl;
    ConstructDoubleLayertJMPOGenerator(mpo_gen, lattice, model_params);
  } else if (params.Geometry == "OBC") {
    DoubleLayerSquareOBC lattice(Ly, Lx);
    cout << "lattice construct" << std::endl;
    ConstructDoubleLayertJMPOGenerator(mpo_gen, lattice, model_params);
  } else if (params.Geometry == "Torus") {
    DoubleLayerSquareTorus lattice(Ly, Lx);
    cout << "lattice construct" << std::endl;
    ConstructDoubleLayertJMPOGenerator(mpo_gen, lattice, model_params);
  }

  auto mat_repr_mpo = mpo_gen.GenMatReprMPO();
  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

  std::vector<long unsigned int> stat_labs(N);
  for (size_t i = 0; i < N / 2 + params.Numhole; i++) {
    stat_labs[i] = 2;
  }
  for (size_t i = N / 2 + params.Numhole; i < N; i++) {
    stat_labs[i] = i % 2;
  }

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(stat_labs.begin(), stat_labs.end(), g);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
#endif

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
  double e0;
  if (!has_bond_dimension_parameter) {
    qlmps::FiniteVMPSSweepParams sweep_params(
        params.Sweeps,
        params.Dmin, params.Dmax, params.CutOff,
        qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter)
    );
    e0 = qlmps::FiniteDMRG(mps, mat_repr_mpo, sweep_params, comm);
  } else {
    for (size_t i = 0; i < DMRG_time; i++) {
      size_t D = input_D_set[i];
      if (rank == 0) {
        std::cout << "D_max = " << D << std::endl;
      }
      qlmps::FiniteVMPSSweepParams sweep_params(
          params.Sweeps,
          D, D, params.CutOff,
          qlmps::LanczosParams(params.LanczErr, MaxLanczIterSet[i])
      );
      e0 = qlmps::FiniteDMRG(mps, mat_repr_mpo, sweep_params, comm);
    }
  }
  if (rank == kMPIMasterRank) {
    std::cout << "E0/site: " << e0 / N << std::endl;
    endTime = clock();
    cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }
  MPI_Barrier(comm);
#if SYM_LEVEL == 0
  if (rank == kMPIMasterRank) {
    Timer one_site_timer("measure one site operators");
    MeasureOneSiteOp(mps, kMpsPath, {nf, sz}, {"nf", "sz"});
    cout << "measured one point function.<====" << endl;
    one_site_timer.PrintElapsed();

    size_t ref_site = 0;
    auto szsz_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, sz, sz, ref_site);
    DumpMeasuRes(szsz_corr, "sz" + std::to_string(ref_site) + "sz");
    auto spsm_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, sp, sm, ref_site);
    DumpMeasuRes(spsm_corr, "sp" + std::to_string(ref_site) + "sm");
    auto smsp_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, sm, sp, ref_site);
    DumpMeasuRes(smsp_corr, "sm" + std::to_string(ref_site) + "sp");
    auto nn_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, nf, nf, ref_site);
    DumpMeasuRes(nn_corr, "nf" + std::to_string(ref_site) + "nf");
  }

#endif
  MPI_Finalize();
  return 0;
}


