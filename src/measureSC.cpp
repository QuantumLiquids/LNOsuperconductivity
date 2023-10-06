/*
    measureSC.cpp
    for measure pair correlation function. memory optimized and parallel version.
    usage:
        mpirun -n 8 ./measureSC
    note: processor number must be 4.
    Optional arguments:
      --start=
      --end=
    Which are defaultly set as start=Lx/4, end = 3*Lx/4+2
*/
#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"
#include <ctime>
#include "gqdouble.h"
#include "operators.h"
#include "params_case.h"
#include "myutil.h"
#include "my_measure.h"
#include "gqten/utility/timer.h"

#include "boost/mpi.hpp"

using std::cout;
using std::endl;
using std::vector;
using FiniteMPST = gqmps2::FiniteMPS<TenElemT, U1U1QN>;
using gqmps2::SiteVec;
using gqmps2::MeasureOneSiteOp;
using gqten::Timer;
using gqmps2::MeasureElectronPhonon4PointFunction;

// When used to measure, note if should not set start too small to exceed canonical center.
bool Parser(const int argc, char *argv[],
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
  namespace mpi = boost::mpi;
  mpi::environment env;
  mpi::communicator world;
  clock_t startTime, endTime;
  startTime = clock();

  size_t beginx;
  size_t endx;
  bool start_argument_has = Parser(argc, argv, beginx, endx);

  CaseParams params(argv[1]);

  size_t Lx = params.Lx, Ly = params.Ly;
  size_t N = Lx * Ly;
  if (GetNumofMps() != N) {
    std::cout << "The number of mps files are inconsistent with mps size!" << std::endl;
    exit(1);
  }

  if (!start_argument_has) {
    beginx = Lx / 4;
    endx = beginx + Lx / 2 + 2;
  }

  OperatorInitial();

  const SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(N, pb_out);
  FiniteMPST mps(sites);
  gqten::hp_numeric::SetTensorTransposeNumThreads(params.Threads);
  gqten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  Timer foursite_timer("measure four site operators");
  vector<vector<size_t>> xx_fourpoint_sitessetF;
  vector<vector<size_t>> yy_fourpoint_sitessetF;
  vector<vector<size_t>> yx_fourpoint_sitessetF;
  std::vector<size_t> Tx(Lx * Ly), Ty(Lx * Ly);
  for (size_t i = 0; i < Lx * Ly; ++i) {
    size_t y = i % Ly, x = i / Ly;
    Tx[i] = y + Ly * ((x + 1) % Lx);
    Ty[i] = (y + 1) % Ly + Ly * x;
  }

  xx_fourpoint_sitessetF.reserve(Ly * (endx - beginx));
  yx_fourpoint_sitessetF.reserve(Ly * (endx - beginx));
  yy_fourpoint_sitessetF.reserve(Ly * (endx - beginx));
  for (size_t y = 0; y < Ly; ++y) {
    auto site1F = beginx * Ly + y;
    for (size_t x = beginx + 2; x < endx; x = x + 1) {
      auto site2F = x * Ly + y;
      vector<size_t> xxsites = {site1F, Tx[site1F], site2F, Tx[site2F]};
      xx_fourpoint_sitessetF.push_back(xxsites);

      vector<size_t> yysites = {site1F, Ty[site1F], site2F, Ty[site2F]};
      sort(yysites.begin(), yysites.end());
      yy_fourpoint_sitessetF.push_back(yysites);

      vector<size_t> yxsites = {site1F, Ty[site1F], site2F, Tx[site2F]};
      sort(yxsites.begin(), yxsites.end());
      yx_fourpoint_sitessetF.push_back(yxsites);
    }
  }

  std::vector<Tensor> sc_phys_ops_a = {bupc, bdnc, bupa, bdna};
  std::vector<Tensor> sc_phys_ops_b = {bdnc, bupc, bupa, bdna};
  std::vector<Tensor> sc_phys_ops_c = {bupc, bdnc, bdna, bupa}; //<B>=<C>
  std::vector<Tensor> sc_phys_ops_d = {bdnc, bupc, bdna, bupa}; //<A>=<D>

  std::string file_name_postfix;
  if (start_argument_has) {
    file_name_postfix = "begin" + std::to_string(beginx) + "end" + std::to_string(endx);
  } else {
    file_name_postfix = "";
  }

  if (world.rank() == 0) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_a, yy_fourpoint_sitessetF, Ly, "scsyya" + file_name_postfix);
  } else if (world.rank() == 1) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_b, yy_fourpoint_sitessetF, Ly, "scsyyb" + file_name_postfix);
  } else if (world.rank() == 2) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_c, yy_fourpoint_sitessetF, Ly, "scsyyc" + file_name_postfix);
  } else if (world.rank() == 3) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_d, yy_fourpoint_sitessetF, Ly, "scsyyd" + file_name_postfix);
  } else if (world.rank() == 4) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_a, yx_fourpoint_sitessetF, Ly, "scsyxa" + file_name_postfix);
  } else if (world.rank() == 5) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_b, yx_fourpoint_sitessetF, Ly, "scsyxb" + file_name_postfix);
  } else if (world.rank() == 6) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_c, yx_fourpoint_sitessetF, Ly, "scsyxc" + file_name_postfix);
  } else if (world.rank() == 7) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_d, yx_fourpoint_sitessetF, Ly, "scsyxd" + file_name_postfix);
  }
  cout << "measured SC correlation function.<====" << endl;
  foursite_timer.PrintElapsed();

  endTime = clock();
  cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

  return 0;

}
