/**
 * Fix two-site update (Kondo model variant)
 *
 * DMRG lattice mapping (two-layer, two-orbital, two-leg):
 *   - N = 4 * Ly * Lx with Ly = 2; 4 = two layers × two on-site dof
 *     (extended electron, localized spin). Even indices are electrons,
 *     odd indices are localized spins.
 *   - Indices advance along x; within each x there are 8 sites covering legs,
 *     layers, and on-site dof. Horizontal bonds satisfy (i - i_ref) % 8 == 0,
 *     and integer x-distance is |i - i_ref| / 8 in plotting/analysis.
 *
 * Patches two adjacent MPS tensors using a two-site Lanczos optimization
 * with existing left/right environments and MPO tensors at sites lsite and lsite+1.
 */

#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "qlmps/qlmps.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <unistd.h>
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"

using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;
using namespace qlmps;
using namespace qlten;
using namespace std;

int ParserFixMpsArgs(const int argc, char *argv[],
                     size_t &lsite,
                     size_t &thread,
                     bool &load_mps);

/**
 * @example ./kondo_two_layer_fix_mps2 --thread=24 --lsite=10 --load_mps=0
 * Puts the canonical center on the left MPS tensor after SVD.
 */
int main(int argc, char *argv[]) {
  std::cout << "Patch two MPS tensors by two-site Lanczos (Kondo model)" << std::endl;
  std::cout << "Inputs required: renv*.qlten, lenv*.qlten, mpo/mpo_ten*.qlten" << std::endl;

  size_t lsite(0), thread(0);
  bool load_mps(false);
  ParserFixMpsArgs(argc, argv, lsite, thread, load_mps);

  std::cout << "Argument read:\nleft site = " << lsite
            << "\nthread = " << thread
            << "\nload old mps = " << load_mps << std::endl;

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(thread);
#endif
  const size_t N = GetNumofMps();
  const string temp_path = kRuntimeTempPath;

  const size_t target_site = lsite; // left site of the two-site block

  Tensor renv, lenv, lmpo, rmpo, lmps, rmps;

  // Load right environment that starts at site (target_site + 2)
  string file = GenEnvTenName("r", (N - 1) - target_site - 1, temp_path);
  if (access(file.c_str(), 4) != 0) {
    std::cout << "Cannot read file " << file << std::endl;
    return 1;
  }
  ifstream tensor_file(file, ifstream::binary);
  tensor_file >> renv;
  tensor_file.close();

  // Load left environment ending at site (target_site - 1)
  file = GenEnvTenName("l", target_site, temp_path);
  if (access(file.c_str(), 4) != 0) {
    std::cout << "Cannot read file " << file << std::endl;
    return 1;
  }
  tensor_file.open(file, ifstream::binary);
  tensor_file >> lenv;
  tensor_file.close();

  // Load MPO tensors at sites target_site and target_site + 1
  file = "mpo/mpo_ten" + std::to_string(target_site) + ".qlten";
  if (access(file.c_str(), 4) != 0) {
    std::cout << "Cannot read file " << file << std::endl;
    return 1;
  }
  tensor_file.open(file, ifstream::binary);
  tensor_file >> lmpo;
  tensor_file.close();

  file = "mpo/mpo_ten" + std::to_string(target_site + 1) + ".qlten";
  if (access(file.c_str(), 4) != 0) {
    std::cout << "Cannot read file " << file << std::endl;
    return 1;
  }
  tensor_file.open(file, ifstream::binary);
  tensor_file >> rmpo;
  tensor_file.close();

  bool new_code;
  if (lenv.GetIndexes()[0].GetDir() == TenIndexDirType::OUT) {
    new_code = false;
  } else {
    new_code = true;
  }

  IndexT index0, index1, index2, index3;
  if (!new_code) {
    index0 = InverseIndex(lenv.GetIndexes()[0]);
    index1 = InverseIndex(lmpo.GetIndexes()[1]);
    index2 = InverseIndex(rmpo.GetIndexes()[1]);
    index3 = InverseIndex(renv.GetIndexes()[0]);
  } else {
    index0 = lenv.GetIndexes()[0];
    index1 = InverseIndex(lmpo.GetIndexes()[1]);
    index2 = InverseIndex(rmpo.GetIndexes()[1]);
    index3 = renv.GetIndexes()[0];
  }

  vector<IndexT> indexes = {index0, index1, index2, index3};

  // Build initial state: either from existing MPS pair or default qn0 block
  Tensor *initial_state = nullptr;
  if (!load_mps) {
    initial_state = new Tensor({index0, index1, index2, index3});
    qlten::ShapeT blk_shape = {index0.GetQNSctNum(), index1.GetQNSctNum(), index2.GetQNSctNum(), index3.GetQNSctNum()};
    qlten::CoorsT blk_coors;
    bool found(false);
    for (size_t i = 0; i < blk_shape[0] && !found; i++) {
      for (size_t j = 0; j < blk_shape[1] && !found; j++) {
        for (size_t k = 0; k < blk_shape[2] && !found; k++) {
          for (size_t l = 0; l < blk_shape[3]; l++) {
            if (CalcDiv(indexes, {i, j, k, l}) == qn0) {
              blk_coors = {i, j, k, l};
              found = true;
              break;
            }
          }
        }
      }
    }
    if (!found) {
      std::cout << "Cannot find a proper block for qn0." << std::endl;
      return 1;
    }
    qlten::CoorsT zeros_coor = {0, 0, 0, 0};
    auto &bstd = initial_state->GetBlkSparDataTen();
    bstd.ElemSet(std::make_pair(blk_coors, zeros_coor), 1.0);
  } else {
    // Load existing two MPS tensors and contract them to build initial state
    file = "mps/mps_ten" + std::to_string(target_site) + ".qlten";
    if (access(file.c_str(), 4) != 0) {
      std::cout << "Cannot read file " << file << std::endl;
      return 1;
    }
    tensor_file.open(file, ifstream::binary);
    tensor_file >> lmps;
    tensor_file.close();

    file = "mps/mps_ten" + std::to_string(target_site + 1) + ".qlten";
    if (access(file.c_str(), 4) != 0) {
      std::cout << "Cannot read file " << file << std::endl;
      return 1;
    }
    tensor_file.open(file, ifstream::binary);
    tensor_file >> rmps;
    tensor_file.close();

    initial_state = new Tensor();
    Contract(&lmps, &rmps, {{2}, {0}}, initial_state);
    if (initial_state->GetIndexes() != indexes) {
      std::cout << "Loaded MPS pair indexes inconsistent with environment." << std::endl;
      return 2;
    }
  }

  // Lanczos on two-site effective Hamiltonian
  qlmps::LanczosParams lancz_params(1e-9, 60);

  std::vector<Tensor *> eff_ham(4);
  eff_ham[0] = const_cast<Tensor *>(&lenv);
  eff_ham[1] = const_cast<Tensor *>(&lmpo);
  eff_ham[2] = const_cast<Tensor *>(&rmpo);
  eff_ham[3] = const_cast<Tensor *>(&renv);

  LanczosRes<Tensor> lancz_res = LanczosSolver<Tensor>(
      eff_ham,
      initial_state,
      &eff_ham_mul_two_site_state,
      lancz_params);

  // SVD the optimized two-site tensor and dump back to mps_ten<lsite>, mps_ten<lsite+1>
  Tensor u, vt;
  using DTenT = QLTensor<QLTEN_Double, QNT>;
  DTenT s;
  QLTEN_Double actual_trunc_err;
  size_t D;
  size_t svd_ldims = 2;
  size_t Dmax = lenv.GetShape()[0];
  SVD(
      lancz_res.gs_vec,
      svd_ldims, Div(lenv) - Div(lenv),
      1e-12, Dmax, Dmax,
      &u, &s, &vt, &actual_trunc_err, &D
  );

  std::cout << "Ground state energy = " << lancz_res.gs_eng << std::endl;
  std::cout << "SVD D = " << D << ", TruncErr = " << actual_trunc_err << std::endl;
  delete lancz_res.gs_vec;

  lmps = Tensor();
  rmps = std::move(vt);
  Contract(&u, &s, {{2}, {0}}, &lmps);

  file = "mps/mps_ten" + std::to_string(target_site) + ".qlten";
  ofstream dump_file(file, ofstream::binary);
  dump_file << lmps;
  dump_file.close();

  file = "mps/mps_ten" + std::to_string(target_site + 1) + ".qlten";
  dump_file.open(file, ofstream::binary);
  dump_file << rmps;
  dump_file.close();

  return 0;
}

int ParserFixMpsArgs(const int argc, char *argv[],
                     size_t &lsite,
                     size_t &thread,
                     bool &load_mps) {
  int nOptionIndex = 1;

  string arguement1 = "--lsite=";
  string arguement2 = "--thread=";
  string arguement3 = "--load_mps=";
  bool site_argument_has(false), thread_argument_has(false), load_mps_argument_has(false);
  while (nOptionIndex < argc) {
    if (strncmp(argv[nOptionIndex], arguement1.c_str(), arguement1.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement1.size()];
      lsite = atoi(para_string.c_str());
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
    lsite = 0;
    std::cout << "Note: no lsite argument, set it as 0 by default." << std::endl;
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


