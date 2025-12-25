//
// Standalone single-particle correlation measurement for two-layer 2D Kondo lattice
//

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "../src_kondo_1d_chain/kondo_hilbert_space.h"
#include "params_case.h"
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"
#include "../src_tj_double_layer_single_orbital_2d/my_measure.h"
#include "finite_mps_extended.h"

using namespace qlmps;
using namespace qlten;
using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: mpirun -np <np> ./kondo_two_layer_measure_sp params.json [mps_path] [ref_site]" << std::endl;
    return 1;
  }

  CaseParams params(argv[1]);
  std::string mps_path = kMpsPath;
  if (argc > 2) {
    mps_path = argv[2];
  }

  size_t Lx = params.Lx;
  size_t Ly = params.Ly;
  size_t N = 4 * Ly * Lx; // even: itinerant electrons; odd: local spins

  size_t ref_site = N / 4;
  if (argc > 3) {
    ref_site = static_cast<size_t>(std::stoul(argv[3]));
  }
  if (ref_site % 2 == 1) ref_site += 1; // ensure even site

  cout << "[Single-Particle Measure] Lx=" << Lx << " Ly=" << Ly << " N=" << N << endl;
  cout << "ref_site=" << ref_site << " (even itinerant)\n";
  cout << "MPS path: " << mps_path << endl;

  // build physical basis: even->extended electron, odd->localized spin
  std::vector<IndexT> pb_set = std::vector<IndexT>(N);
  for (size_t i = 0; i < N; ++i) {
    if (i % 2 == 0) pb_set[i] = pb_outE;
    else pb_set[i] = pb_outL;
  }
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(pb_set);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  auto &ops = hubbard_ops;

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
#endif

  // file postfix to match existing conventions
  std::ostringstream oss;
  oss << "conventional_square" << "Jk" << params.JK
      << "Jperp" << params.Jperp
      << "U" << params.U
      << "Lx" << Lx
      << "Ly" << Ly
      << "D" << params.Dmax.back();
  std::string file_postfix = oss.str();

  // Up-spin channel
  auto sp_up_a = MeasureTwoSiteOpGroupInKondoLattice(mps, mps_path, ops.bupcF, ops.bupa, ref_site, ops.f);
  DumpMeasuRes(sp_up_a, std::string("cup_dag_cup") + file_postfix);

  auto sp_up_b = MeasureTwoSiteOpGroupInKondoLattice(mps, mps_path, TenElemT(-1) * ops.bupaF, ops.bupc, ref_site, ops.f);
  DumpMeasuRes(sp_up_b, std::string("cup_cup_dag") + file_postfix);

  // Down-spin channel
  auto sp_dn_a = MeasureTwoSiteOpGroupInKondoLattice(mps, mps_path, ops.bdnc, ops.Fbdna, ref_site, ops.f);
  DumpMeasuRes(sp_dn_a, std::string("cdown_dag_cdown") + file_postfix);

  auto sp_dn_b = MeasureTwoSiteOpGroupInKondoLattice(mps, mps_path, TenElemT(-1) * ops.bdna, ops.Fbdnc, ref_site, ops.f);
  DumpMeasuRes(sp_dn_b, std::string("cdown_cdown_dag") + file_postfix);

  std::cout << "Measured single-particle correlations (4 variants)" << std::endl;

  return 0;
}
