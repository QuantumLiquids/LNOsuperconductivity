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

using namespace gqmps2;
using namespace gqten;
using namespace std;

int main(int argc, char *argv[]) {
  CaseParams params(argv[1]);
  size_t Lx = params.Lx, Ly = params.Ly;
  size_t N = 2 * Lx * Ly;
  DoubleLayertJModelParamters model_params(params);
  model_params.Print();
  clock_t startTime, endTime;
  startTime = clock();
  OperatorInitial();
  const SiteVec<TenElemT, U1U1QN> sites = SiteVec<TenElemT, U1U1QN>(N, pb_out);
  gqmps2::MPOGenerator<TenElemT, U1U1QN> mpo_gen(sites, qn0);

  if (params.Geometry == "Cylinder") {
    if (Ly < 3) {
      std::cout << "Cylinder is not well defined for Ly = " << Ly << std::endl;
      exit(1);
    }
    DoubleLayerSquareCylinder lattice(Ly, Lx);
    cout << "lattice construct" << std::endl;
    ConstructDoubleLayertJMPOGenerator(mpo_gen, lattice, model_params);
  } else if (params.Geometry == "OBC") {
    DoubleLayerSquareOBC lattice(Ly, Lx);
    cout << "lattice construct" << std::endl;
    ConstructDoubleLayertJMPOGenerator(mpo_gen, lattice, model_params);
  } else if (params.Geometry == "Torus") {
    DoubleLayerSquareTorus lattice(Ly, Lx);
    cout << "lattice construct" << std::endl;
    ConstructDoubleLayertJMPOGenerator(mpo_gen, lattice, model_params);
  }
//  else if (params.Geometry == "Rotated") {
//    SquareRotatedCylinder lattice(Ly, Lx);
//    cout << "lattice construct" << std::endl;
//    ConstructDoubleLayertJMPOGenerator(mpo_gen, lattice, model_params);
//  }

  gqten::hp_numeric::SetTensorTransposeNumThreads(params.Threads);
  gqten::hp_numeric::SetTensorManipulationThreads(params.Threads);
  gqmps2::MPO<Tensor> mpo = mpo_gen.Gen();
  cout << "MPO generated." << endl;

  const std::string kMpoPath = "mpo";
  const std::string kMpoTenBaseName = "mpo_ten";
  if (!IsPathExist(kMpoPath)) {
    CreatPath(kMpoPath);
  }
  for (size_t i = 0; i < mpo.size(); i++) {
    std::string filename = kMpoPath + "/" +
                           kMpoTenBaseName + std::to_string(i)
                           + "." + kGQTenFileSuffix;
    mpo.DumpTen(i, filename);
  }

  endTime = clock();
  cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  return 0;
}


