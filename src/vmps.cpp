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
#include <random>

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
  size_t N = 2 * Lx * Ly;
  DoubleLayertJModelParamters model_params(params);
  if (world.rank() == 0) {
    model_params.Print();
  }

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

  clock_t startTime, endTime;
  startTime = clock();
  OperatorInitial();
  const SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(N, pb_out);
  gqmps2::MPO<Tensor> mpo(N);
  const std::string kMpoPath = "mpo";
  const std::string kMpoTenBaseName = "mpo_ten";
  if (IsPathExist(kMpoPath)) {
    for (size_t i = 0; i < mpo.size(); i++) {
      std::string filename = kMpoPath + "/" +
                             kMpoTenBaseName + std::to_string(i) + "." + kGQTenFileSuffix;
      mpo.LoadTen(i, filename);
    }

    cout << "MPO loaded." << endl;
  } else {
    cout << "No mpo directory. exiting" << std::endl;
    exit(0);
  }

  using FiniteMPST = gqmps2::FiniteMPS<TenElemT, U1U1QN>;
  FiniteMPST mps(sites);

  if (params.Threads == 0) {
    const size_t max_threads = std::thread::hardware_concurrency();
    params.Threads = max_threads;
  }

  if (world.rank() == 0) {
    if (params.Threads > 2) {
      gqten::hp_numeric::SetTensorTransposeNumThreads(params.Threads - 2);
      gqten::hp_numeric::SetTensorManipulationThreads(params.Threads - 2);
    } else {
      gqten::hp_numeric::SetTensorTransposeNumThreads(params.Threads);
      gqten::hp_numeric::SetTensorManipulationThreads(params.Threads);
    }
    std::cout << "max threads = " << params.Threads << std::endl;
  } else {
    gqten::hp_numeric::SetTensorTransposeNumThreads(params.Threads);
    gqten::hp_numeric::SetTensorManipulationThreads(params.Threads);
  }

  std::vector<long unsigned int> stat_labs(N);
  for (size_t i = 0; i < N / 2 + params.Numhole; i++) {
    stat_labs[i] = 2;
  }
  for (size_t i = N / 2 + params.Numhole; i < N; i++) {
    stat_labs[i] = i % 2;
  }
  std::srand(std::time(nullptr));
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(stat_labs.begin(), stat_labs.end(), g);

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
    cout << "No mps directory." << endl;
    gqmps2::DirectStateInitMps(mps, stat_labs);
    cout << "Initial mps as direct product state." << endl;
    mps.Dump(kMpsPath, true);
    if (world.rank() == 0)
      env.abort(-1);
  }

  double e0;

  if (!has_bond_dimension_parameter) {
    gqmps2::SweepParams sweep_params(
        params.Sweeps,
        params.Dmin, params.Dmax, params.CutOff,
        gqmps2::LanczosParams(params.LanczErr, params.MaxLanczIter)
    );
    if (world.rank() == 0) {
      e0 = gqmps2::TwoSiteFiniteVMPS(mps, mpo, sweep_params);
    } else {
      e0 = gqmps2::TwoSiteFiniteVMPS(mps, mpo, sweep_params, world);
    }
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
      if (world.rank() == 0) {
        e0 = gqmps2::TwoSiteFiniteVMPS(mps, mpo, sweep_params);
      } else {
        e0 = gqmps2::TwoSiteFiniteVMPS(mps, mpo, sweep_params, world);
      }
    }
  }

  if (world.rank() == 0) {
    std::cout << "E0/site: " << e0 / N << std::endl;
    endTime = clock();
    cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }
  return 0;
}


