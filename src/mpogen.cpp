#include "qlmps/qlmps.h"
#include "qlten/qlten.h"
#include "gqdouble.h"
#include "operators.h"
#include "params_case.h"
#include "myutil.h"
#include "squarelattice.h"
#include "tJmodel.h"

using namespace qlmps;
using namespace qlten;
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
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);
  qlmps::MPOGenerator<TenElemT, QNT> mpo_gen(sites, qn0);

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


  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
  qlmps::MPO<Tensor> mpo = mpo_gen.Gen();
  cout << "MPO generated." << endl;

  if (!IsPathExist(kMpoPath)) {
    CreatPath(kMpoPath);
  }
  for (size_t i = 0; i < mpo.size(); i++) {
    std::string filename = kMpoPath + "/" +
        kMpoTenBaseName + std::to_string(i)
        + "." + kQLTenFileSuffix;
    mpo.DumpTen(i, filename);
  }

  endTime = clock();
  cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
  return 0;
}


