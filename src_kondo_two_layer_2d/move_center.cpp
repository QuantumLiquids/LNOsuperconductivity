/**
 * Move the canonical center of an existing MPS stored on disk
 * (Kondo lattice variant: alternating itinerant/local-spin sites)
 *
 * DMRG lattice mapping (two-layer, two-orbital, two-leg):
 *   - N = 4 * Ly * Lx with Ly = 2; 4 = two layers × two on-site dof
 *     (extended electron, localized spin). Even indices are electrons,
 *     odd indices are localized spins.
 *   - Indices advance along x; per physical x there are 8 sites spanning
 *     legs, layers and on-site dof. Same-row (delta y = 0) index differences
 *     are multiples of 8; plotting uses |i - i_ref| / 8 as integer x distance.
 */

#include "qlmps/qlmps.h"
#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include <iostream>
#include <fstream>
#include <vector>
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"

using namespace qlmps;
using namespace qlten;
using namespace std;
using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;

int Parser(const int argc, char *argv[],
           size_t &from,
           size_t &to,
           size_t &thread);

/**
 * @example ./move_center --from=10, --to=2, --thread=24
 * the previous center is 10, and the center will move to 2;
 */
int main(int argc, char *argv[]) {
  size_t from(0), to(0), thread(0);
  Parser(argc, argv, from, to, thread);

  std::cout << "Argument read: " << std::endl;
  std::cout << "from = " << from << std::endl;
  std::cout << "to = " << to << std::endl;
  std::cout << "thread = " << thread << std::endl;

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(thread);
#endif
  const size_t N = GetNumofMps();
  // Build Kondo sites: even -> electron, odd -> localized spin
  std::vector<IndexT> pb_set(N);
  for (size_t i = 0; i < N; ++i) {
    pb_set[i] = (i % 2 == 0) ? pb_outE : pb_outL;
  }
  const SiteVec<TenElemT, QNT> sites(pb_set);

  FiniteMPS<TenElemT, QNT> mps(sites);

  std::string mps_path = kMpsPath;
  mps.LoadTen(from, GenMPSTenName(mps_path, from));
  if (to < from) {
    for (int i = from - 1; i >= int(to); i--) {
      mps.LoadTen(i, GenMPSTenName(mps_path, i));
      mps.RightCanonicalizeTen(i + 1);
      std::cout << "Right canonical tensor " << i + 1 << std::endl;
      mps.DumpTen(
          i + 1,
          GenMPSTenName(mps_path, i + 1),
          true
      );
    }
  } else if (to > from) {
    for (size_t i = from + 1; i <= to; i++) {
      mps.LoadTen(i, GenMPSTenName(mps_path, i));
      mps.LeftCanonicalizeTen(i - 1);
      std::cout << "Left canonical tensor " << i - 1 << std::endl;
      mps.DumpTen(
          i - 1,
          GenMPSTenName(mps_path, i - 1),
          true
      );
    }
  }
  mps.DumpTen(
      to,
      GenMPSTenName(mps_path, to),
      true
  );
  return 0;
}

int Parser(const int argc, char *argv[],
           size_t &from,
           size_t &to,
           size_t &thread) {
  int nOptionIndex = 1;

  string arguement1 = "--from=";
  string arguement2 = "--to=";
  string arguement3 = "--thread=";
  bool from_argument_has(false), to_argument_has(false), thread_argument_has(false);
  while (nOptionIndex < argc) {
    if (strncmp(argv[nOptionIndex], arguement1.c_str(), arguement1.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement1.size()];
      from = atoi(para_string.c_str());
      from_argument_has = true;
    } else if (strncmp(argv[nOptionIndex], arguement2.c_str(), arguement2.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement2.size()];
      to = atoi(para_string.c_str());
      to_argument_has = true;
    } else if (strncmp(argv[nOptionIndex], arguement3.c_str(), arguement3.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement3.size()];
      thread = atoi(para_string.c_str());
      thread_argument_has = true;
    } else {
      cout << "Options '" << argv[nOptionIndex] << "' not valid. Run '" << argv[0] << "' for details." << endl;
      //   return -1;
    }
    nOptionIndex++;
  }

  if (!from_argument_has) {
    from = 0;
    std::cout << "Note: no from argument, set it as 0 by default." << std::endl;
  }

  if (!to_argument_has) {
    to = 10;
    std::cout << "Note: no to argument, set it as 10 by default." << std::endl;
  }

  if (!thread_argument_has) {
    thread = 24;
    std::cout << "Note: no thread argument, set it as 24 by default." << std::endl;
  }

  return 0;

}