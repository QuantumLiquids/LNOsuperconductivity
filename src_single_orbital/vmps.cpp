#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include "tJ_type_hilbert_space.h"
#include "tJ_operators.h"
#include "params_case.h"
#include "myutil.h"
#include "double_layer_squarelattice.h"
#include "double_layer_tJmodel.h"
#include <random>

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
  size_t Lx = params.Lx, Ly = params.Ly;
  size_t N = 2 * Lx * Ly;
  DoubleLayertJModelParamters model_params(params);
  if (rank == 0) {
    model_params.Print();
  }

  std::vector<size_t> input_D_set;
  bool has_bond_dimension_parameter = ParserBondDimension(
      argc, argv,
      input_D_set);

  std::vector<size_t> Dmin_set, Dmax_set;
  if (has_bond_dimension_parameter) {
    Dmin_set = input_D_set;
    Dmax_set = input_D_set;
  } else {
    Dmin_set = {params.Dmin};
    Dmax_set = {params.Dmax};
  }
  size_t DMRG_time = Dmax_set.size();
  std::vector<size_t> MaxLanczIterSet(DMRG_time);
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

  clock_t startTime, endTime;
  startTime = clock();
  OperatorInitial();
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);
  qlmps::MPO<Tensor> mpo(N);
  if (IsPathExist(kMpoPath)) {
    for (size_t i = 0; i < mpo.size(); i++) {
      std::string filename = kMpoPath + "/" +
          kMpoTenBaseName + std::to_string(i) + "." + kQLTenFileSuffix;
      mpo.LoadTen(i, filename);
    }

    cout << "MPO loaded." << endl;
  } else {
    cout << "No mpo directory. exiting" << std::endl;
    exit(0);
  }

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

  if (params.Threads == 0) {
    const size_t max_threads = std::thread::hardware_concurrency();
    params.Threads = max_threads;
  }

  if (mpi_size != 0) {
    if (params.Threads > 2 && rank == kMPIMasterRank) {
      qlten::hp_numeric::SetTensorManipulationThreads(params.Threads - 2);
    } else {

      qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
    }
    std::cout << "max threads = " << params.Threads << std::endl;
  } else {

    qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
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
      qlmps::DirectStateInitMps(mps, stat_labs);
      cout << "Initial mps as direct product state." << endl;
      if (rank == 0)
        mps.Dump(kMpsPath, true);
    }
  } else {
    cout << "No mps directory." << endl;
    qlmps::DirectStateInitMps(mps, stat_labs);
    cout << "Initial mps as direct product state." << endl;
    if (rank == 0)
      mps.Dump(kMpsPath, true);
  }

  double e0;

  for (size_t i = 0; i < DMRG_time; i++) {
    if (rank == 0) {
      std::cout << "D_max = " << Dmax_set[i] << std::endl;
    }
//    qlmps::SweepParams sweep_params(
//        params.Sweeps,
//        Dmin_set[i], Dmax_set[i], params.CutOff,
//        qlmps::LanczosParams(params.LanczErr, MaxLanczIterSet[i])
//    );
    qlmps::FiniteVMPSSweepParams sweep_params(
        params.Sweeps,
        Dmin_set[i], Dmax_set[i], params.CutOff,
        qlmps::LanczosParams(params.LanczErr, MaxLanczIterSet[i]),
        params.noise
    );
    if (mpi_size == 1) {
      e0 = qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params);
    } else {
      e0 = qlmps::TwoSiteFiniteVMPS(mps, mpo, sweep_params, comm);
    }
  }

  if (rank == 0) {
    std::cout << "E0/site: " << e0 / N << std::endl;
    endTime = clock();
    cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  }
  MPI_Finalize();
  return 0;
}


