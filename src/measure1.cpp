// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-04-1
*
* Description: Measurement of the one-point function for the MPS of ground state of t-J model
* Usage: mpirun -n 1 ./measure1 params.json
 * or ./measure1 params.json
*/

///<  TODO: optimize the CPU cost.

#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include "gqdouble.h"
#include "operators.h"
#include "params_case.h"
#include "myutil.h"
#include "my_measure.h"

using namespace qlmps;
using namespace qlten;
using namespace std;

int main(int argc, char *argv[]) {
  CaseParams params(argv[1]);
  size_t Lx = params.Lx;
  size_t N = 2 * Lx * params.Ly;

  clock_t startTime, endTime;
  startTime = clock();

  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  OperatorInitial();
#if SYMMETRY_LEVEL == 0
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);
#elif SYMMETRY_LEVEL == 1
  IndexVec<QNT>  pb_out_vec(N);
  for(size_t i = 0; i < N; i++){
    if(i%2==0){
      pb_out_vec[i]= pb_out_layer1;
    } else{
      pb_out_vec[i] = pb_out_layer2;
    }
  }
   const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(pb_out_vec);
#endif

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

  Timer one_site_timer("measure one site operators");
  MeasureOneSiteOp(mps, kMpsPath, {nf, sz}, {"nf", "sz"});
  cout << "measured one point function.<====" << endl;
  one_site_timer.PrintElapsed();

  endTime = clock();
  cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  return 0;
}
