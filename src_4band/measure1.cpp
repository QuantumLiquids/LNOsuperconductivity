// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
* Creation Date: 19th, Feb, 2025
*
* Description: Measurement of the one-point function
* Usage: ./measure1 params.json=
*/

#include "qlten/qlten.h"
#include <ctime>
#include "hilbert_space.h"
#include "hubbard_operators.h"
#include "params_case.h"
#include "myutil.h"

using namespace qlmps;
using namespace qlten;

using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;

int main(int argc, char *argv[]) {
  CaseParams params(argv[1]);
  size_t Lx = params.Lx;
  size_t N = 4 * Lx * params.Ly;

  clock_t startTime, endTime;
  startTime = clock();
#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads);
#endif
  HubbardOperators ops;

  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);

  FiniteMPST mps(sites);

  Timer one_site_timer("measure one site operators");
  MeasureOneSiteOp(mps, kMpsPath, {ops.sz, ops.nf}, {"sz", "nf"});
  std::cout << "measured one point function.<====" << std::endl;
  one_site_timer.PrintElapsed();

  endTime = clock();
  std::cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
  return 0;
}
