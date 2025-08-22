//
// Created by haoxinwang on 10/10/2023.
//

/*
    measure2.cpp
    for measure spin, charge, on-site pair, single-particle correlation function.
    memory optimized and parallel version.
    usage:
        mpirun -n 2*Ly ./measure2
    Optional arguments:
      --start=
      --end=
    Which are set as start=Lx/4, end = 3*Lx/4+2 by default
*/

#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include <ctime>
#include "tJ_type_hilbert_space.h"

#include "params_case.h"
#include "myutil.h"
#include "my_measure.h"
#include "qlten/utility/timer.h"

using std::cout;
using std::endl;
using std::vector;
using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
using qlmps::SiteVec;
using qlmps::MeasureTwoSiteOp;
using qlten::Timer;
using qlmps::MeasureGroupTask;
using qlmps::kMpsPath;

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);
  clock_t startTime, endTime;
  startTime = clock();

  size_t beginx;
  size_t endx;
  bool start_argument_has = ParserMeasureSite(argc, argv, beginx, endx);

  CaseParams params(argv[1]);

  size_t Lx = params.Lx, Ly = params.Ly;
  size_t N = 2 * Lx * Ly;
  if (GetNumofMps() != N) {
    std::cout << "The number of mps files are inconsistent with mps size!" << std::endl;
    exit(1);
  }

  if (!start_argument_has) {
    beginx = Lx / 4;
    endx = beginx + Lx / 2 + 2;
  }

  qlmps::tJOperators<TenElemT, QNT>  ops;

  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);
  FiniteMPST mps(sites);

  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  Timer two_site_timer("measure two site operators");

  std::vector<MeasureGroupTask> measure_tasks;
  measure_tasks.reserve(N);
  //for structure factor.
//  for (size_t site1 = beginx * Ly; site1 < endx * Ly; site1++) {
//    std::vector<size_t> site2;
//    site2.reserve(N - site1);
//    for (size_t j = site1 + 1; j < endx * Ly; j++) {
//      site2.push_back(j);
//    }
//    measure_tasks.push_back(MeasureGroupTask(site1, site2));
//  }

  for (size_t y = 0; y < (2 * Ly); ++y) {
    auto site1 = beginx * (2 * Ly) + y;
    std::vector<size_t> site2;
    site2.reserve((2 * Ly) * (endx - beginx));
    for (size_t j = (beginx + 2) * (2 * Ly); j < endx * (2 * Ly); j++) {
      site2.push_back(j);
    }
    measure_tasks.push_back(MeasureGroupTask(site1, site2));
  }

  Timer two_site_measure_timer("measure spin_ structure factors");
  MeasureTwoSiteOp(mps, kMpsPath, sz, sz, measure_tasks, "zzsf", comm);
  if (rank == 0) {
    std::cout << "measured sz sz correlation." << std::endl;
  }
  MPI_Barrier(comm);
  MeasureTwoSiteOp(mps, kMpsPath, sp, sm, measure_tasks, "pmsf", comm);
  MeasureTwoSiteOp(mps, kMpsPath, sm, sp, measure_tasks, "mpsf", comm);
  MeasureTwoSiteOp(mps, kMpsPath, nf, nf, measure_tasks, "nfnf", comm);
  MeasureTwoSiteOp(mps, kMpsPath, bupc, bupa, measure_tasks, "bupcbupa",
                   comm, f);  //directly equal to the single-particle correlatin function
  MeasureTwoSiteOp(mps, kMpsPath, bdnc, bdna, measure_tasks, "bupcbupa",
                   comm, f);  //directly equal to the single-particle correlatin function
  two_site_measure_timer.PrintElapsed();

  endTime = clock();
  cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

  return 0;
}


