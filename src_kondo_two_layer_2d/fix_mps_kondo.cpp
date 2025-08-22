/**
 * Fix MPS tensor by single-site Lanczos (Kondo model variant)
 */

#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
// #include "qlmps/algorithm/lanczos_solver.h"
// #include "qlmps/algorithm/lanczos_solver_impl.h"
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

/**
 * @example ./fix_mps_kondo --thread=24 --site=10
 */
int main(int argc, char *argv[]) {
  std::cout << "This program patches one MPS tensor by single-site Lanczos (Kondo model)" << std::endl;
  std::cout << "Inputs required: renv*.qlten, lenv*.qlten, mpo_ten*.qlten" << std::endl;
  std::cout << "Output: mps/mps_ten*.qlten" << std::endl;

  size_t site(0), thread(0);
  bool load_mps;
  ParserFixMpsArgs(argc, argv, site, thread, load_mps);

  std::cout << "Argument read:\nsite = " << site << "\nthread = " << thread
            << "\nload original mps: " << load_mps << std::endl;

  qlten::hp_numeric::SetTensorManipulationThreads(thread);
  const size_t N = GetNumofMps();
  const string temp_path = kRuntimeTempPath;

  const size_t target_site = site;

  Tensor renv, lenv, mpo, mps;

  // Load renv
  string file = GenEnvTenName("r", (N - 1) - target_site, temp_path);
  if (access(file.c_str(), 4) != 0) {
    std::cout << "Cannot read file " << file << std::endl;
    exit(1);
  }
  ifstream tensor_file(file, ifstream::binary);
  tensor_file >> renv;
  tensor_file.close();

  // Load lenv
  file = GenEnvTenName("l", target_site, temp_path);
  if (access(file.c_str(), 4) != 0) {
    std::cout << "Cannot read file " << file << std::endl;
    exit(1);
  }
  tensor_file.open(file, ifstream::binary);
  tensor_file >> lenv;
  tensor_file.close();

  // Load mpo at target_site
  file = "mpo/mpo_ten" + std::to_string(target_site) + ".qlten";
  if (access(file.c_str(), 4) != 0) {
    std::cout << "Cannot read file " << file << std::endl;
    exit(1);
  }
  tensor_file.open(file, ifstream::binary);
  tensor_file >> mpo;
  tensor_file.close();
  std::cout << "Loaded renv, lenv, and mpo tensors" << std::endl;

  bool new_code;
  if (lenv.GetIndexes()[0].GetDir() == TenIndexDirType::OUT) {
    new_code = false;
  } else {
    new_code = true;
  }
  IndexT index0, index1, index2;
  if (!new_code) {
    index0 = InverseIndex(lenv.GetIndexes()[0]);
    index1 = InverseIndex(mpo.GetIndexes()[1]);
    index2 = InverseIndex(renv.GetIndexes()[0]);
  } else {
    index0 = lenv.GetIndexes()[0];
    index1 = InverseIndex(mpo.GetIndexes()[1]);
    index2 = renv.GetIndexes()[0];
  }

  vector<IndexT> indexes = {index0, index1, index2};

  // Initial state
  Tensor *initial_state;
  if (!load_mps) {
    initial_state = new Tensor({index0, index1, index2});
    std::cout << "Initialize default tensor as initial state" << std::endl;

    qlten::ShapeT blk_shape = {index0.GetQNSctNum(), index1.GetQNSctNum(), index2.GetQNSctNum()};
    qlten::CoorsT blk_coors;
    bool found_block(false);
    for (size_t i = 0; i < blk_shape[0] && !found_block; i++) {
      for (size_t j = 0; j < blk_shape[1] && !found_block; j++) {
        for (size_t k = 0; k < blk_shape[2]; k++) {
          if (CalcDiv(indexes, {i, j, k}) == qn0) {
            blk_coors = {i, j, k};
            found_block = true;
            std::cout << "Found proper block for qn0." << std::endl;
            break;
          }
        }
      }
    }
    if (!found_block) {
      std::cout << "Cannot find a proper block for qn0." << std::endl;
      exit(1);
    }
    qlten::CoorsT zeros_coor = {0, 0, 0};
    auto &bstd = initial_state->GetBlkSparDataTen();
    bstd.ElemSet(std::make_pair(blk_coors, zeros_coor), 1.0);
    std::cout << "Generated initial tensor." << std::endl;
  } else {
    initial_state = new Tensor();
    file = "mps/mps_ten" + std::to_string(target_site) + ".qlten";
    if (access(file.c_str(), 4) != 0) {
      std::cout << "Cannot read file " << file << std::endl;
      exit(1);
    }
    tensor_file.open(file, ifstream::binary);
    tensor_file >> *initial_state;
    tensor_file.close();
    std::cout << "Loaded MPS tensor from " << file << std::endl;
    if (initial_state->GetIndexes() != indexes) {
      std::cout << "Loaded MPS indexes inconsistent with environment; abort." << std::endl;
      exit(2);
    }
  }

  qlmps::LanczosParams params(1e-9, 200);

  std::vector<Tensor *> eff_ham(3);
  eff_ham[0] = const_cast<Tensor *>(&lenv);
  eff_ham[1] = const_cast<Tensor *>(&mpo);
  eff_ham[2] = const_cast<Tensor *>(&renv);

  LanczosRes<Tensor> res = LanczosSolver<Tensor>(
      eff_ham,
      initial_state,
      &eff_ham_mul_single_site_state,
      params);

  mps = std::move(*res.gs_vec);
  std::cout << "Lanczos Iter = " << res.iters << std::endl;
  std::cout << "Ground state energy = " << res.gs_eng << std::endl;
  delete res.gs_vec;

  file = "mps/mps_ten" + std::to_string(target_site) + ".qlten";
  ofstream dump_file(file, ofstream::binary);
  dump_file << mps;
  std::cout << "Dumped optimized MPS to " << file << std::endl;

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