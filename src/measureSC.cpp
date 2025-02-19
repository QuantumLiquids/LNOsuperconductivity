/*
    measureSC.cpp
    for measure pair correlation function. memory optimized and parallel version.
    usage:
        mpirun -n 16 ./measureSC
    note: processor number must be 16.
    Optional arguments:
      --start=
      --end=
    Which are set as start=Lx/4, end = 3*Lx/4+2 by default
*/
#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include <ctime>
#include "gqdouble.h"
#include "operators.h"
#include "params_case.h"
#include "myutil.h"
#include "my_measure.h"
#include "qlten/utility/timer.h"

#include "boost/mpi.hpp"

using std::cout;
using std::endl;
using std::vector;
using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
using qlmps::SiteVec;
using qlmps::MeasureOneSiteOp;
using qlten::Timer;
using qlmps::MeasureElectronPhonon4PointFunction;

int main(int argc, char *argv[]) {
  namespace mpi = boost::mpi;
  mpi::environment env;
  mpi::communicator world;
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

  OperatorInitial();

  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);
  FiniteMPST mps(sites);

  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  Timer foursite_timer("measure four site operators");
  vector<vector<size_t>> xx_fourpoint_sitessetF, yy_fourpoint_sitessetF, yx_fourpoint_sitessetF, zz_fourpoint_sitessetF;
  std::vector<size_t> Tx(2 * Lx * Ly), Ty(2 * Lx * Ly);
  for (size_t i = 0; i < N; ++i) {
    size_t y = i % (2 * Ly), x = i / (2 * Ly);
    Tx[i] = y + (2 * Ly) * ((x + 1) % Lx);
    Ty[i] = (y + 2) % (2 * Ly) + (2 * Ly) * x;
  }

  xx_fourpoint_sitessetF.reserve((2 * Ly) * (endx - beginx));
  yx_fourpoint_sitessetF.reserve((2 * Ly) * (endx - beginx));
  yy_fourpoint_sitessetF.reserve((2 * Ly) * (endx - beginx));
  zz_fourpoint_sitessetF.reserve((Ly) * (endx - beginx));
  for (size_t y = 0; y < (2 * Ly); ++y) {
    auto site1F = beginx * (2 * Ly) + y;
    for (size_t x = beginx + 2; x < endx; x = x + 1) {
      auto site2F = x * (2 * Ly) + y;
      vector<size_t> xxsites = {site1F, Tx[site1F], site2F, Tx[site2F]};
      xx_fourpoint_sitessetF.push_back(xxsites);

      vector<size_t> yysites = {site1F, Ty[site1F], site2F, Ty[site2F]};
      sort(yysites.begin(), yysites.end());
      yy_fourpoint_sitessetF.push_back(yysites);

      vector<size_t> yxsites = {site1F, Ty[site1F], site2F, Tx[site2F]};
      sort(yxsites.begin(), yxsites.end());
      yx_fourpoint_sitessetF.push_back(yxsites);

      if (y % 2 == 0) {
        vector<size_t> zzsites = {site1F, site1F + 1, site2F, site2F + 1};
        zz_fourpoint_sitessetF.push_back(zzsites);
      }
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

  if (rank == 0) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_a, yy_fourpoint_sitessetF, (2*Ly), "scsyya" + file_name_postfix);
  } else if (rank == 1) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_b, yy_fourpoint_sitessetF, (2*Ly), "scsyyb" + file_name_postfix);
  } else if (rank == 2) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_c, yy_fourpoint_sitessetF, (2*Ly), "scsyyc" + file_name_postfix);
  } else if (rank == 3) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_d, yy_fourpoint_sitessetF, (2*Ly), "scsyyd" + file_name_postfix);
  } else if (rank == 4) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_a, yx_fourpoint_sitessetF, (2*Ly), "scsyxa" + file_name_postfix);
  } else if (rank == 5) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_b, yx_fourpoint_sitessetF, (2*Ly), "scsyxb" + file_name_postfix);
  } else if (rank == 6) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_c, yx_fourpoint_sitessetF, (2*Ly), "scsyxc" + file_name_postfix);
  } else if (rank == 7) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_d, yx_fourpoint_sitessetF, (2*Ly), "scsyxd" + file_name_postfix);
  } else if (rank == 8) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_a, xx_fourpoint_sitessetF, (2*Ly), "scsxxa" + file_name_postfix);
  } else if (rank == 9) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_b, xx_fourpoint_sitessetF, (2*Ly), "scsxxb" + file_name_postfix);
  } else if (rank == 10) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_c, xx_fourpoint_sitessetF, (2*Ly), "scsxxc" + file_name_postfix);
  } else if (rank == 11) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_d, xx_fourpoint_sitessetF, (2*Ly), "scsxxd" + file_name_postfix);
  } else if (rank == 12) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_a, zz_fourpoint_sitessetF, Ly, "scszza" + file_name_postfix);
  } else if (rank == 13) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_b, zz_fourpoint_sitessetF, Ly, "scszzb" + file_name_postfix);
  } else if (rank == 14) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_c, zz_fourpoint_sitessetF, Ly, "scszzc" + file_name_postfix);
  } else if (rank == 15) {
    MeasureElectronPhonon4PointFunction(mps, sc_phys_ops_d, zz_fourpoint_sitessetF, Ly, "scszzd" + file_name_postfix);
  }
  cout << "measured SC correlation function.<====" << endl;
  foursite_timer.PrintElapsed();

  endTime = clock();
  cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

  return 0;

}
