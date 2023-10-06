// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-04-1
*
* Description: Single processor DMRG for t-J model with t_2 and J_2. DMRG style code
*/


#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"
#include <time.h>
#include <stdlib.h>
#include "gqdouble.h"
#include "operators.h"
#include "params_case.h"
#include "myutil.h"
#include "squarelattice.h"
#include "tJmodel.h"

using namespace gqmps2;
using namespace gqten;
using namespace std;

int main(int argc, char *argv[]) {
  namespace mpi = boost::mpi;
  mpi::environment env(mpi::threading::multiple);
  if (env.thread_level() < mpi::threading::multiple) {
    std::cout << "thread level of env is not right." << std::endl;
    env.abort(-1);
  }
  mpi::communicator world;
  CaseParams params(argv[1]);
  size_t Lx = params.Lx, Ly = params.Ly;
  size_t N = Lx * Ly;
  DoubleLayertJModelParamters model_params(params);
  if (world.rank() == 0) {
    model_params.Print();
  }
  clock_t startTime, endTime;
  startTime = clock();
  OperatorInitial();
  const SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(N, pb_out);
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


  gqmps2::MPOGenerator<TenElemT, U1U1QN> mpo_gen(sites, qn0);

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
  using FiniteMPST = gqmps2::FiniteMPS<TenElemT, U1U1QN>;
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

  gqten::hp_numeric::SetTensorTransposeNumThreads(params.Threads);
  gqten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  if (IsPathExist(kMpsPath)) {
    if (N == GetNumofMps()) {
      cout << "The number of mps files is consistent with mps size." << endl;
      cout << "Directly use mps from files." << endl;
    } else {
      gqmps2::DirectStateInitMps(mps, stat_labs);
      cout << "Initial mps as direct product state." << endl;
      if (world.rank() == 0)
        mps.Dump(kMpsPath, true);
    }
  } else {
    gqmps2::DirectStateInitMps(mps, stat_labs);
    cout << "Initial mps as direct product state." << endl;
    if (world.rank() == 0)
      mps.Dump(kMpsPath, true);
  }
  double e0;
  if (!has_bond_dimension_parameter) {
    gqmps2::SweepParams sweep_params(
        params.Sweeps,
        params.Dmin, params.Dmax, params.CutOff,
        gqmps2::LanczosParams(params.LanczErr, params.MaxLanczIter)
    );
    e0 = gqmps2::FiniteDMRG(mps, mat_repr_mpo, sweep_params, world);
  } else {
    for (size_t i = 0; i < DMRG_time; i++) {
      size_t D = input_D_set[i];
      if (world.rank() == 1) {
        std::cout << "D_max = " << D << std::endl;
      }
      gqmps2::SweepParams sweep_params(
          params.Sweeps,
          D, D, params.CutOff,
          gqmps2::LanczosParams(params.LanczErr, MaxLanczIterSet[i])
      );
      e0 = gqmps2::FiniteDMRG(mps, mat_repr_mpo, sweep_params, world);
    }
  }
  if (world.rank() == 0) {
    std::cout << "E0/site: " << e0 / N << std::endl;
    endTime = clock();
    cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }
  return 0;
}


