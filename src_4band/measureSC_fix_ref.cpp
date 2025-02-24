/*
    measureSC.cpp
    for measure pair correlation function. memory optimized and parallel version.
    usage:
        mpirun -n 8 ./measureSC --start=start_site
*/
#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include <ctime>
#include "hilbert_space.h"
#include "hubbard_operators.h"
#include "params_case.h"
#include "myutil.h"
#include "qlten/utility/timer.h"

using FiniteMPST = qlmps::FiniteMPS<TenElemT, U1U1QN>;
using qlmps::SiteVec;
using qlmps::MeasureFourSiteOpGroup;
using qlmps::kMpsPath;
using qlmps::DumpMeasuRes;
using qlten::Timer;

// When used to measure, note if should not set start too small to exceed canonical center.
bool ParserSC(const int argc, char *argv[],
            size_t &start,
            size_t &end) {
  int nOptionIndex = 1;

  std::string arguement1 = "--start=";
  std::string arguement2 = "--end=";
  bool start_argument_has(false), end_argument_has(false);
  while (nOptionIndex < argc) {
    if (strncmp(argv[nOptionIndex], arguement1.c_str(), arguement1.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement1.size()];
      start = atoi(para_string.c_str());
      start_argument_has = true;
    } else if (strncmp(argv[nOptionIndex], arguement2.c_str(), arguement2.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement2.size()];
      end = atoi(para_string.c_str());
      end_argument_has = true;
    }
    nOptionIndex++;
  }

  if (start_argument_has != end_argument_has) {
    std::cout << "Only setting one start/end argument, exit(1)." << std::endl;
    exit(1);
  }

  if (!start_argument_has) {
    std::cout << "Note: no start/end argument, set it by default (L/4, 3*L/4+2)." << std::endl;
  }

  return start_argument_has;
}

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &mpi_size);
  if (mpi_size > 8) {
    std::cout << "too many MPI processes. Set -np 8" << std::endl;
  }
  clock_t startTime, endTime;
  startTime = clock();

  size_t start;
  size_t endx;
  bool start_argument_has = ParserSC(argc, argv, start, endx);

  CaseParams params(argv[1]);

  size_t Lx = params.Lx, Ly = params.Ly;
  size_t N = 4 * Lx * Ly;
  if (GetNumofMps() != N) {
    std::cout << "The number of mps files are inconsistent with mps size!" << std::endl;
    exit(1);
  }

  if (!start_argument_has) {
    start = 0;
    endx = start + Lx / 2 + 2;
  }

  HubbardOperators ops;

  const SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(N, pb_out);
  FiniteMPST mps(sites);
#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads);
#endif
  Timer foursite_timer("measure four site operators");
  std::array<size_t, 2> ref_site{start, start + Ly}; // interlayer pairing

  std::vector<std::array<size_t, 2>> target_bond;

  target_bond.reserve(Ly * Lx);
  for (size_t x = 0; x < 2 * Lx; x += 2) {
    for (size_t y = 0; y < Ly; y++) {
      size_t target_site0 = x * (2 * Ly) + y;
      size_t target_site1 = target_site0 + Ly; // interlayer pairing
      if (target_site0 > start + Ly && target_site1 > start + Ly) {
        if (params.Geometry == "OBC" && y < Ly - 1) {
          target_bond.push_back({target_site0, target_site1});
        } else if (params.Geometry == "Cylinder") {
          std::cout << "Please update the cylinder code here !" << std::endl;
          exit(1);
        }
      }
    }
  }

  std::array<Tensor, 4> sc_phys_ops_a = {ops.bupcF, ops.Fbdnc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_b = {ops.bdnc, ops.bupc, ops.bupaF, ops.Fbdna};
  std::array<Tensor, 4> sc_phys_ops_c = {ops.bupcF, ops.Fbdnc, ops.bdna, ops.bupa};
  std::array<Tensor, 4> sc_phys_ops_d = {ops.bdnc, ops.bupc, ops.bdna, ops.bupa};

  std::string file_name_postfix = "_fix_refer" + std::to_string(ref_site[0]);
  Tensor f = ops.f;
  if (rank == 0) {
    auto measure_res = MeasureFourSiteOpGroup(mps, kMpsPath, sc_phys_ops_a, ref_site, target_bond, f);
    DumpMeasuRes(measure_res, "scs_interlayera" + file_name_postfix);
  }
  if (rank == 1 % mpi_size) {
    auto measure_res = MeasureFourSiteOpGroup(mps, kMpsPath, sc_phys_ops_b, ref_site, target_bond, f);
    DumpMeasuRes(measure_res, "scs_interlayerb" + file_name_postfix);
  }
  if (rank == 2 % mpi_size) {
    auto measure_res = MeasureFourSiteOpGroup(mps, kMpsPath, sc_phys_ops_c, ref_site, target_bond, f);
    DumpMeasuRes(measure_res, "scs_interlayerc" + file_name_postfix);
  }
  if (rank == 3 % mpi_size) {
    auto measure_res = MeasureFourSiteOpGroup(mps, kMpsPath, sc_phys_ops_d, ref_site, target_bond, f);
    DumpMeasuRes(measure_res, "scs_interlayerd" + file_name_postfix);
  }
//  if (rank == 4 % mpi_size) {
//    auto measure_res = MeasureFourSiteOpGroup(mps, kMpsPath, sc_phys_ops_a, ref_site, target_x_bond, f);
//    DumpMeasuRes(measure_res, "scsyxa" + file_name_postfix);
//  }
//  if (rank == 5 % mpi_size) {
//    auto measure_res = MeasureFourSiteOpGroup(mps, kMpsPath, sc_phys_ops_b, ref_site, target_x_bond, f);
//    DumpMeasuRes(measure_res, "scsyxb" + file_name_postfix);
//  }
//  if (rank == 6 % mpi_size) {
//    auto measure_res = MeasureFourSiteOpGroup(mps, kMpsPath, sc_phys_ops_c, ref_site, target_x_bond, f);
//    DumpMeasuRes(measure_res, "scsyxc" + file_name_postfix);
//  }
//  if (rank == 7 % mpi_size) {
//    auto measure_res = MeasureFourSiteOpGroup(mps, kMpsPath, sc_phys_ops_d, ref_site, target_x_bond, f);
//    DumpMeasuRes(measure_res, "scsyxd" + file_name_postfix);
//  }
  std::cout << "measured SC correlation function.<====" << std::endl;
  foursite_timer.PrintElapsed();

  endTime = clock();
  std::cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
  MPI_Finalize();
  return 0;
}
