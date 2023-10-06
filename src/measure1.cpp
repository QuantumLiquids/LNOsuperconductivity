// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Hao-Xin Wang <wanghx18@mails.tsinghua.edu.cn>
* Creation Date: 2023-04-1
*
* Description: Measurement of the one-point function for the MPS of ground state of t-J model
* Usage: mpirun -n 2 ./measure1 params.json
*/

///<  TODO: optimize the CPU cost.

#include "gqmps2/gqmps2.h"
#include "gqten/gqten.h"
#include <ctime>
#include "gqdouble.h"
#include "operators.h"
#include "params_case.h"
#include "myutil.h"
#include "my_measure.h"


using namespace gqmps2;
using namespace gqten;
using namespace std;

int main(int argc, char *argv[]) {
  namespace mpi = boost::mpi;
  mpi::environment env;
  mpi::communicator world;

  CaseParams params(argv[1]);
  size_t Lx = params.Lx;
  size_t N= Lx * params.Ly;

  clock_t startTime,endTime;
  startTime = clock();

  gqten::hp_numeric::SetTensorTransposeNumThreads(params.Threads);
  gqten::hp_numeric::SetTensorManipulationThreads(params.Threads);

  OperatorInitial();
  const SiteVec<TenElemT, U1U1QN> sites=SiteVec<TenElemT, U1U1QN>(N, pb_out);


  using FiniteMPST = gqmps2::FiniteMPS<TenElemT, U1U1QN>;
  FiniteMPST mps(sites);

  Timer one_site_timer("measure  one site operators");
  if(world.rank() == 0){
    MeasureOneSiteOp(mps, kMpsPath, nf, "nf");
  } else{
    MeasureOneSiteOp(mps, kMpsPath, sz, "sz");
  }

  cout << "measured one point function.<====" <<endl;
  one_site_timer.PrintElapsed();


  endTime = clock();
  cout << "CPU Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  return 0;
}
