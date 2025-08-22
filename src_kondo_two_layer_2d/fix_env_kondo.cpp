// Kondo-version of environment fixer, mirroring working t-J utility but with alternating site types

#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "qlmps/qlmps.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"

using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;
using namespace qlmps;
using namespace qlten;
using namespace std;

int ParserFixMpsArgs(const int argc, char *argv[],
                     size_t &site,
                     size_t &thread,
                     bool &load_mps);

int main(int argc, char *argv[]) {
  std::cout << "This program fixes environment tensors around a specific site (Kondo model)" << std::endl;
  std::cout << "The input must include the relevant files: renv*.qlten, lenv*.qlten, mpo_ten*.qlten" << std::endl;
  std::cout << "The output is the files renv*.qlten, lenv*.qlten" << std::endl;

  size_t site(0), thread(0);
  bool load_mps;
  ParserFixMpsArgs(argc, argv, site, thread, load_mps);

  std::cout << "Argument read: " << std::endl;
  std::cout << "site = " << site << std::endl;
  std::cout << "thread = " << thread << std::endl;

  qlten::hp_numeric::SetTensorManipulationThreads(thread);
  const size_t N = GetNumofMps();
  const string temp_path = kRuntimeTempPath;
  std::string mps_path = kMpsPath;

  MPO<Tensor> mpo(N);
  const std::string kMpoPath = "mpo";
  const std::string kMpoTenBaseName = "mpo_ten";
  for (size_t i = 0; i < N; i++) {
    std::string filename = kMpoPath + "/" +
        kMpoTenBaseName + std::to_string(i) + "." + kQLTenFileSuffix;
    mpo.LoadTen(i, filename);
  }

  // Kondo: alternating physical bonds for electron and localized spin
  std::vector<IndexT> pb_set(N);
  for (size_t i = 0; i < N; ++i) {
    pb_set[i] = (i % 2 == 0) ? pb_outE : pb_outL;
  }
  const SiteVec<TenElemT, QNT> sites(pb_set);
  FiniteMPS<TenElemT, QNT> mps(sites);

  int left_start_env, right_start_env;
  for (left_start_env = static_cast<int>(site); left_start_env >= 0; left_start_env--) {
    string file = GenEnvTenName("l", left_start_env, temp_path);
    if (access(file.c_str(), 4) != 0) {
      std::cout << "Cannot read file " << file << "." << std::endl;
    } else {
      std::cout << "Found file " << file << std::endl;
      break;
    }
  }

  for (right_start_env = static_cast<int>(N - site - 1); right_start_env >= 0; right_start_env--) {
    string file = GenEnvTenName("r", right_start_env, temp_path);
    if (access(file.c_str(), 4) != 0) {
      std::cout << "Cannot read file " << file << "." << std::endl;
    } else {
      std::cout << "Found file " << file << std::endl;
      break;
    }
  }

  if (left_start_env == -1) {
    mps.LoadTen(0, GenMPSTenName(mps_path, 0));
    auto mps_trivial_index = mps.front().GetIndexes()[0];
    auto mpo_trivial_index_inv = InverseIndex(mpo.front().GetIndexes()[0]);
    auto mps_trivial_index_inv = InverseIndex(mps_trivial_index);
    Tensor lenv({mps_trivial_index, mpo_trivial_index_inv, mps_trivial_index_inv});
    lenv({0, 0, 0}) = 1;
    mps.dealloc(0);
    std::string file = GenEnvTenName("l", 0, temp_path);
    WriteQLTensorTOFile(lenv, file);
    left_start_env = 0;
  }
  size_t from = static_cast<size_t>(left_start_env + 1);

  string file = GenEnvTenName("l", from - 1, temp_path);
  Tensor lenv;
  ifstream lenv_file(file);
  lenv_file >> lenv;

  mps.LoadTen(from - 1, GenMPSTenName(mps_path, from - 1));
  bool new_code;
  if (mps[from - 1].GetIndexes()[0] == lenv.GetIndexes()[0]) {
    new_code = true;
  } else if (mps[from - 1].GetIndexes()[0] == lenv.GetIndexes()[2]) {
    new_code = false;
  } else {
    std::cout << "Unexpected index layout; aborting." << std::endl;
    exit(1);
  }

  for (size_t i = from; i <= site; i++) {
    mps.LoadTen(i - 1, GenMPSTenName(mps_path, i - 1));
    auto dump_file = GenEnvTenName("l", i, temp_path);
    if (!new_code) {
      Tensor temp1;
      Contract(&mps[i - 1], &lenv, {{0}, {0}}, &temp1);
      lenv = Tensor();
      Tensor temp2;
      Contract(&temp1, &mpo[i - 1], {{0, 2}, {1, 0}}, &temp2);
      auto mps_ten_dag = Dag(mps[i - 1]);
      Contract(&temp2, &mps_ten_dag, {{1, 2}, {0, 1}}, &lenv);
    } else {
      lenv = UpdateSiteLenvs(lenv, mps[i - 1], mpo[i - 1]);
    }
    WriteQLTensorTOFile(lenv, dump_file);
    std::cout << "Grown and dumped tensor " << dump_file << std::endl;
    mps.dealloc(i - 1);
  }

  if (right_start_env == -1) {
    mps.LoadTen(N - 1, GenMPSTenName(mps_path, N - 1));
    auto mps_trivial_index = mps.back().GetIndexes()[2];
    auto mpo_trivial_index_inv = InverseIndex(mpo.back().GetIndexes()[3]);
    auto mps_trivial_index_inv = InverseIndex(mps_trivial_index);
    Tensor renv({mps_trivial_index, mpo_trivial_index_inv, mps_trivial_index_inv});
    renv({0, 0, 0}) = 1;
    mps.dealloc(N - 1);
    string file_r0 = GenEnvTenName("r", 0, temp_path);
    WriteQLTensorTOFile(renv, file_r0);
    right_start_env = 0;
  }

  from = static_cast<size_t>(right_start_env + 1);
  file = GenEnvTenName("r", from - 1, temp_path);
  Tensor renv;
  ifstream renv_file(file);
  renv_file >> renv;

  mps.LoadTen(N - from, GenMPSTenName(mps_path, N - from));
  if (mps[N - from].GetIndexes()[2] == renv.GetIndexes()[0]) {
    new_code = true;
  } else if (mps[N - from].GetIndexes()[2] == renv.GetIndexes()[2]) {
    new_code = false;
  } else {
    std::cout << "Unexpected index layout; aborting." << std::endl;
    exit(1);
  }

  for (size_t i = from; i <= N - site - 1; i++) {
    std::string mps_file_name = GenMPSTenName(mps_path, N - i);
    mps.LoadTen(N - i, mps_file_name);
    auto dump_file = GenEnvTenName("r", i, temp_path);
    if (!new_code) {
      Tensor temp1;
      Contract(&mps[N - i], &renv, {{2}, {0}}, &temp1);
      renv = Tensor();
      Tensor temp2;
      Contract(&temp1, &mpo[N - i], {{1, 2}, {1, 3}}, &temp2);
      auto mps_ten_dag = Dag(mps[N - i]);
      Contract(&temp2, &mps_ten_dag, {{3, 1}, {1, 2}}, &renv);
    } else {
      renv = UpdateSiteRenvs(renv, mps[N - i], mpo[N - i]);
    }
    WriteQLTensorTOFile(renv, dump_file);
    std::cout << "Grown and dumped tensor " << dump_file << std::endl;
    mps.dealloc(N - i);
  }

  return 0;
}

int ParserFixMpsArgs(const int argc, char *argv[],
                     size_t &site,
                     size_t &thread,
                     bool &load_mps) {
  int nOptionIndex = 1;

  string arguement1 = "--site=";
  string arguement2 = "--thread=";
  string arguement3 = "--load_mps=";
  bool site_argument_has(false), thread_argument_has(false), load_mps_argument_has(false);
  while (nOptionIndex < argc) {
    if (strncmp(argv[nOptionIndex], arguement1.c_str(), arguement1.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement1.size()];
      site = atoi(para_string.c_str());
      site_argument_has = true;
    } else if (strncmp(argv[nOptionIndex], arguement2.c_str(), arguement2.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement2.size()];
      thread = atoi(para_string.c_str());
      thread_argument_has = true;
    } else if (strncmp(argv[nOptionIndex], arguement3.c_str(), arguement3.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement3.size()];
      load_mps = (bool) atoi(para_string.c_str());
      load_mps_argument_has = true;
    } else {
      cout << "Options '" << argv[nOptionIndex] << "' not valid. Run '" << argv[0] << "' for details." << endl;
    }
    nOptionIndex++;
  }

  if (!site_argument_has) {
    site = 0;
    std::cout << "Note: no site argument, set it as 0 by default." << std::endl;
  }

  if (!thread_argument_has) {
    thread = 24;
    std::cout << "Note: no thread argument, set it as 24 by default." << std::endl;
  }

  if (!load_mps_argument_has) {
    load_mps = false;
    std::cout << "Note: no load_mps argument, set it as false by default." << std::endl;
  }

  return 0;
}