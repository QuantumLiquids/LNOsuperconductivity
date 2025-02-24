/**
 * DMRG for Hubbard Altermagnetism
 *
 */
#include "qlmps/qlmps.h"
#include "./hilbert_space.h"
#include "./hubbard_operators.h"
#include "./myutil.h"
#include "./params_case.h"

/**
 *  Ly = 2
 *  O : d_x^-y^2, o: d_z^2
 *           ------------> x
 *  Layer1   O--o--O--o--O--o--O--o  (y = 0)
 *           |  |  |  |  |  |  |  |
 *  Layer1   O--o--O--o--O--o--O--o  (y = 1)
 *           |  |  |  |  |  |  |  |
 *  Layer2   O--o--O--o--O--o--O--o  (y = 0)
 *           |  |  |  |  |  |  |  |
 *  Layer2   O--o--O--o--O--o--O--o  (y = 1)
 */

using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;

int main(int argc, char *argv[]) {
  using namespace qlmps;
  using namespace qlten;
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);

  if (argc == 1) {
    if (rank == 0) {
      std::cout
          << "Usage: \n mpirun -np <num_proc> ./dmrg <params file> --D=<list of bond dimension, connected by comma>\n";
    }
    return 0;
  } else if (argc == 2) {
    if (rank == 0)
      std::cout
          << "The complete usage can be: Usage: \n mpirun -np <num_proc> ./dmrg <params file> --D=<list of bond dimension, connected by comma>"
          << std::endl;
  }

  CaseParams params(argv[1]);

#ifndef USE_GPU
  if (rank == 0 && mpi_size > 1 && params.TotalThreads > 2) {
    qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads - 2);
  } else {
    qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads);
  }
#endif

  /******** Model parameter ********/
  size_t Lx = params.Lx, Ly = params.Ly;
  if (Ly != 2) {
    std::cout << "Do not support Ly : " << Ly << std::endl;
    exit(1);
  }
  size_t N = 4 * Lx * Ly;//two orbital
  double t1 = params.t1, t2 = params.t2, J_H = params.Jh;
  double U = params.U;
  if (rank == 0) {
    std::cout << "System size = (" << Lx << "," << Ly << ")" << std::endl;
    std::cout << "The number of electron sites =" << N << std::endl;
    std::cout << "Model parameter: t1 :" << t1 << ", t2 :" << t2
              << ", U : " << U
              << ", J_H : " << J_H
              << std::endl;
  }
  /****** DMRG parameter *******/
  qlmps::FiniteVMPSSweepParams sweep_params(
      params.Sweeps,
      params.Dmin, params.Dmax, params.CutOff,
      qlmps::LanczosParams(params.LanczErr, params.MaxLanczIter),
      params.noise
  );

  clock_t startTime, endTime;
  startTime = clock();

  HubbardOperators ops;
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
      if (rank == 0)
        std::cout << "Setting MaxLanczIter as : [" << MaxLanczIterSet[0] << ", ";
      for (size_t i = 1; i < DMRG_time - 1; i++) {
        MaxLanczIterSet[i] = MaxLanczIterSet[i - 1] + MaxLanczIterSetSpace;
        if (rank == 0)
          std::cout << MaxLanczIterSet[i] << ", ";
      }
      if (rank == 0)
        std::cout << MaxLanczIterSet.back() << "]" << std::endl;
    } else {
      if (rank == 0)
        std::cout << "Setting MaxLanczIter as : [" << MaxLanczIterSet[0] << "]" << std::endl;
    }
  }

  /****** Initialize MPS ******/
  FiniteMPST mps(sites);
  if (rank == 0) {
    if (!IsPathExist(kMpsPath) || !(N == GetNumofMps())) {
      std::cout << "Initial mps as direct product state." << std::endl;
      //0: double occupancy; 1: spin up; 2: spin down; 3: empty
      std::vector<size_t> stat_labs1(N / 2, 3), stat_labs2(N / 2, 3); // d_x^2-y^2, d_z^2

      if (params.NumEle1 <= N / 2 && params.NumEle2 <= N / 2 && params.NumEle1 % 2 == 0 && params.NumEle2 % 2 == 0) {
        std::fill(stat_labs1.begin(), stat_labs1.begin() + params.NumEle1 / 2, 1); // orbital 1, spin up
        std::fill(stat_labs1.begin() + params.NumEle1 / 2,
                  stat_labs1.begin() + params.NumEle1,
                  2); //orbital 2, spin down

        std::fill(stat_labs2.begin(), stat_labs2.begin() + params.NumEle2 / 2, 1); // orbital 2, spin up
        std::fill(stat_labs2.begin() + params.NumEle2 / 2,
                  stat_labs2.begin() + params.NumEle2,
                  2); //orbital 2, spin down

      } else {
        std::cout << "Do not support num electrons!" << std::endl;
        exit(1);
      }
      std::shuffle(stat_labs1.begin(), stat_labs1.end(), std::random_device());
      std::shuffle(stat_labs2.begin(), stat_labs2.end(), std::random_device());
      std::vector<size_t> stat_labs(N);
      size_t chunk_size = 2 * Ly;
      size_t index = 0;
      for (size_t i = 0; i < stat_labs1.size(); i += chunk_size) {
        // Copy chunk from stat_labs1
        std::copy(stat_labs1.begin() + i,
                  stat_labs1.begin() + std::min(i + chunk_size, stat_labs1.size()),
                  stat_labs.begin() + index);
        index += chunk_size;

        // Copy chunk from stat_labs2
        std::copy(stat_labs2.begin() + i,
                  stat_labs2.begin() + std::min(i + chunk_size, stat_labs2.size()),
                  stat_labs.begin() + index);
        index += chunk_size;
      }

      qlmps::DirectStateInitMps(mps, stat_labs);
      mps.Dump(sweep_params.mps_path, true);
    }
  }


  /*******  Creation MPO/MRO *******/
  qlmps::MPOGenerator<TenElemT, QNT> mpo_gen(sites, qn0);

  //Hund's coupling
  for (size_t x = 0; x < 2 * Lx; x += 2) {
    for (size_t y = 0; y < 2 * Ly; y++) {
      size_t site1 = x * (2 * Ly) + y;
      size_t site2 = (x + 1) * (2 * Ly) + y;
      mpo_gen.AddTerm(-J_H, ops.sz, site1, ops.sz, site2);
      mpo_gen.AddTerm(-J_H / 2.0, ops.sp, site1, ops.sm, site2);
      mpo_gen.AddTerm(-J_H / 2.0, ops.sm, site1, ops.sp, site2);
    }
  }
  //t_perp hopping
  for (size_t x = 1; x < 2 * Lx; x += 2) {
    for (size_t y = 0; y < Ly; y++) {
      size_t site1 = x * (2 * Ly) + y;
      size_t site2 = site1 + Ly;
      mpo_gen.AddTerm(-t2, ops.bupcF, site1, ops.bupa, site2, ops.f);
      mpo_gen.AddTerm(t2, ops.bupaF, site1, ops.bupc, site2, ops.f);
      mpo_gen.AddTerm(-t2, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
      mpo_gen.AddTerm(t2, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
    }
  }

  //t_para Horizontal hopping
  for (size_t x = 0; x < 2 * Lx - 2; x += 2) {
    for (size_t y = 0; y < (2 * Ly); y++) {
      size_t y_phy = y % Ly; // physical value of y
      size_t x_phy = x / 2;// physical value of 2
      double t_eff;
      if ((x_phy + y_phy) % 2 == 0) {
        t_eff = (1 + params.delta);
      } else {
        t_eff = (1 - params.delta);
      }
      size_t site1 = x * (2 * Ly) + y;
      size_t site2 = site1 + (4 * Ly);
      mpo_gen.AddTerm(-t_eff, ops.bupcF, site1, ops.bupa, site2, ops.f);
      mpo_gen.AddTerm(t_eff, ops.bupaF, site1, ops.bupc, site2, ops.f);
      mpo_gen.AddTerm(-t_eff, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
      mpo_gen.AddTerm(t_eff, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
    }
  }

  //t_para Vertical hopping
  for (size_t x = 0; x < 2 * Lx; x += 2) {
    for (size_t y = 0; y < (2 * Ly); y++) {
      if (y % Ly < Ly - 1) { // OBC
        size_t y_phy = y % Ly; // physical value of y
        size_t x_phy = x / 2;// physical value of 2
        double t_eff;
        if ((x_phy + y_phy) % 2 == 0) {
          t_eff = (1 + params.delta);
        } else {
          t_eff = (1 - params.delta);
        }
        size_t site1 = x * (2 * Ly) + y;
        size_t site2 = site1 + 1;
        mpo_gen.AddTerm(-t_eff, ops.bupcF, site1, ops.bupa, site2, ops.f);
        mpo_gen.AddTerm(t_eff, ops.bupaF, site1, ops.bupc, site2, ops.f);
        mpo_gen.AddTerm(-t_eff, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
        mpo_gen.AddTerm(t_eff, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
      } else if (Ly > 2) { // y% Ly == Ly-1
        //PBC winding code here

      }
    }
  }
  // perturbation hopping
  for (size_t x = 0; x < 2 * Lx - 1; x++) {
    for (size_t y = 0; y < 2 * Ly; y++) {
      size_t site1 = x * (2 * Ly) + y;
      size_t site2 = (x + 1) * (2 * Ly) + y;
      TenElemT t_noise = params.PA;
      mpo_gen.AddTerm(-t_noise, ops.bupcF, site1, ops.bupa, site2, ops.f);
      mpo_gen.AddTerm(t_noise, ops.bupaF, site1, ops.bupc, site2, ops.f);
      mpo_gen.AddTerm(-t_noise, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
      mpo_gen.AddTerm(t_noise, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
    }
  }

  for (size_t i = 0; i < N; i++) {
    mpo_gen.AddTerm(U, ops.Uterm, i);
    if ((i / (2 * Ly)) % 2 == 0) {
      mpo_gen.AddTerm(params.mu1, ops.nf, i); // d_x^2-y^2 orbital
    } else {
      mpo_gen.AddTerm(params.mu2, ops.nf, i); // d_z^2 orbital
    }
  }

  if (params.PinningField) {
    mpo_gen.AddTerm(1.0, ops.sz, 0); // d_x^2-y^2 orbital
  }

  auto mro = mpo_gen.GenMatReprMPO(true);
  if (rank == 0)
    std::cout << "MRO generated." << std::endl;

  // dmrg
  double e0;
  if (!has_bond_dimension_parameter) {
    e0 = qlmps::FiniteDMRG(mps, mro, sweep_params, comm);
  } else {
    for (size_t i = 0; i < DMRG_time; i++) {
      size_t D = input_D_set[i];
      if (rank == 0) {
        std::cout << "D_max = " << D << std::endl;
      }
      qlmps::FiniteVMPSSweepParams sweep_params(
          params.Sweeps,
          D, D, params.CutOff,
          qlmps::LanczosParams(params.LanczErr, MaxLanczIterSet[i]),
          params.noise
      );
      e0 = qlmps::FiniteDMRG(mps, mro, sweep_params, comm);
    }
  }
  endTime = clock();
  std::cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
  MPI_Finalize();
  return 0;
}
