// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-04-1
*
* Description: DMRG for t-J model with t_2 and J_2. Renormalized operator realization.
*/


#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include <ctime>
#include "../src_single_orbital/tJ_type_hilbert_space.h"
#include "tJ_operators.h"
#include "../src_single_orbital/params_case.h"
#include "../src_single_orbital/myutil.h"
#include "squarelattice.h"
#include "single_layer_tJmodel.h"

using namespace qlmps;
using namespace qlten;
using namespace std;

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  CaseParams params(argv[1]);
  size_t Lx = params.Lx, Ly = params.Ly;
  size_t N = Lx * Ly;
  tJModelParamters model_params(params);
  if (rank == 0) {
    model_params.Print();
  }
  clock_t startTime, endTime;
  startTime = clock();
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

  SquareRotatedCylinder lattice(Ly, Lx);
  cout << "lattice construct" << std::endl;
  ConstructAnitJMPOGenerator(mpo_gen, lattice, model_params);

//  if (Ly == 2) {
//    SquareLadder lattice(Lx);
//    cout << "lattice construct" << std::endl;
//    ConstructAnitJMPOGenerator(mpo_gen, lattice, model_params);
//  } else if (params.Geometry == "Cylinder") {
//    SquareCylinder lattice(Ly, Lx);
//    cout << "lattice construct" << std::endl;
//    ConstructAnitJMPOGenerator(mpo_gen, lattice, model_params);
//  } else if (params.Geometry == "OBC") {
//    SquareOBC lattice(Ly, Lx);
//    cout << "lattice construct" << std::endl;
//    ConstructAnitJMPOGenerator(mpo_gen, lattice, model_params);
//  } else if (params.Geometry == "Rotated") {
//
//  } else if (params.Geometry == "Torus") {
//    SquareTorus lattice(Ly, Lx);
//    cout << "lattice construct" << std::endl;
//    ConstructAnitJMPOGenerator(mpo_gen, lattice, model_params);
//  }

  auto mat_repr_mpo = mpo_gen.GenMatReprMPO();
  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

  std::vector<long unsigned int> stat_labs(N);
  size_t site_number_per_hole;
  if (params.Numhole > 0) {
    site_number_per_hole = N / params.Numhole;
  } else {
    site_number_per_hole = N + 99;
  }

  size_t sz_label = 0;
  size_t hole_num = 0;
  for (size_t i = 0; i < N; ++i) {
    if (i % site_number_per_hole == site_number_per_hole - 1 && hole_num < params.Numhole) {
      stat_labs[i] = 2;
      hole_num++;
    } else {
      stat_labs[i] = sz_label % 2;
      sz_label++;
    }
  }

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
      if (rank == 1) {
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
  std::cout << "E0/site: " << e0 / N << std::endl;
  endTime = clock();
  cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  MPI_Finalize();
  return 0;
}


