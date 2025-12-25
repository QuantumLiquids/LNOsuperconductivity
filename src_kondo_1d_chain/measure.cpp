/*
 * Author: Hao-Xin Wang <wanghaoxin1996@gmail.com>
 * Description: Post-process measurements for 1D Kondo chain MPS on disk.
 *
 * This program is "measure-only": it loads an existing MPS dump and measures
 * both itinerant-electron observables (even sites) and localized-spin
 * observables (odd sites).
 */

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include "kondo_hilbert_space.h"
#include "./params_case.h"
#include "../src_tj_double_layer_single_orbital_2d/myutil.h"
#include "../src_tj_double_layer_single_orbital_2d/my_measure.h"

using namespace qlmps;
using namespace qlten;
using namespace std;

int main(int argc, char *argv[]) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);

  CaseParams params(argv[1]);

  // Optional argument: set the MPS path (default: ./mps).
  std::string mps_path = kMpsPath;
  if (argc > 2) {
    mps_path = argv[2];
    std::cout << "Set MPS path as " << mps_path << std::endl;
  }

  const size_t L = params.L;
  const double Jk = params.JK;
  const double U = params.U;
  const size_t N = 2 * L;

  if (rank == 0) {
    cout << "L = " << L << "\n";
    cout << "N = " << N << "\n";
    cout << "Jk = " << Jk << "\n";
    cout << "U = " << U << "\n";
    cout << "mu(pinning defect) = " << params.mu << "\n";
  }

  std::vector<IndexT> pb_set(N);
  for (size_t i = 0; i < N; ++i) {
    pb_set[i] = (i % 2 == 0) ? pb_outE : pb_outL;
  }
  const SiteVec<TenElemT, QNT> sites(pb_set);

  HubbardOperators<TenElemT, QNT> hubbard_ops;
  SpinOneHalfOperatorsU1U1 local_spin_ops;

  using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;
  FiniteMPST mps(sites);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.Threads);
#endif

  // Make sure the on-disk MPS is centralized at site 0.
  if (rank == 0) {
    mps.Load(mps_path);
    std::cout << "Success load mps into memory.\n";
    mps.Centralize(0);
    std::cout << "Centralize mps to 0 site.\n";
    mps.Dump(mps_path, true);
    std::cout << "Dump mps into disk.\n";
  }
  MPI_Barrier(comm);

  // Reference sites for measurements.
  size_t ref_elec = N / 4;
  if (ref_elec % 2 == 1) { ref_elec = (ref_elec > 0) ? (ref_elec - 1) : 0; } // force even
  const size_t ref_loc = std::min(ref_elec + 1, N - 1);                      // paired odd

  std::vector<size_t> elec_targets;
  for (size_t i = ref_elec + 2; i < N; i += 2) elec_targets.push_back(i);
  std::vector<size_t> loc_targets;
  for (size_t i = ref_loc + 2; i < N; i += 2) loc_targets.push_back(i);

  std::vector<size_t> even_sites;
  for (size_t i = 0; i < N; i += 2) even_sites.push_back(i);
  std::vector<size_t> odd_sites;
  for (size_t i = 1; i < N; i += 2) odd_sites.push_back(i);

  std::ostringstream oss;
  oss << "Jk" << Jk << "U" << U << "L" << L << "D" << params.Dmax.back();
  if (params.mu != 0.0) oss << "mu" << params.mu;
  const std::string file_postfix = oss.str();

  // --- Two-site correlations (itinerant electrons, even sites) ---
  using OpT = Tensor;
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> elec_two_site_ops = {
      {"szsz", hubbard_ops.sz, hubbard_ops.sz},
      {"spsm", hubbard_ops.sp, hubbard_ops.sm},
      {"smsp", hubbard_ops.sm, hubbard_ops.sp},
      {"nfnf", hubbard_ops.nf, hubbard_ops.nf},
  };
  for (size_t i = 0; i < elec_two_site_ops.size(); ++i) {
    if (i % mpi_size == rank) {
      const auto &[label, op1, op2] = elec_two_site_ops[i];
      auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_elec, elec_targets);
      DumpMeasuRes(measu_res, label + file_postfix);
    }
  }

  // --- Two-site correlations (localized spins, odd sites) ---
  const std::vector<std::tuple<std::string, const OpT &, const OpT &>> loc_two_site_ops = {
      {"lszsz", local_spin_ops.sz, local_spin_ops.sz},
      {"lspsm", local_spin_ops.sp, local_spin_ops.sm},
      {"lsmsp", local_spin_ops.sm, local_spin_ops.sp},
  };
  for (size_t i = 0; i < loc_two_site_ops.size(); ++i) {
    if (i % mpi_size == rank) {
      const auto &[label, op1, op2] = loc_two_site_ops[i];
      auto measu_res = MeasureTwoSiteOpGroup(mps, mps_path, op1, op2, ref_loc, loc_targets);
      DumpMeasuRes(measu_res, label + file_postfix);
    }
  }

  // --- One-site observables ---
  // Keep the legacy electron one-site filenames (no postfix) for backward compatibility.
  const std::vector<QLTensor<TenElemT, QNT>> elec_one_site_ops = {hubbard_ops.sz, hubbard_ops.nf};
  const std::vector<std::string> elec_one_site_labels = {"sz_local", "nf_local"};
  if ((elec_two_site_ops.size()) % mpi_size == rank) {
    MeasureOneSiteOp(mps, mps_path, elec_one_site_ops, even_sites, elec_one_site_labels);
  }

  const std::vector<QLTensor<TenElemT, QNT>> loc_one_site_ops = {local_spin_ops.sz};
  const std::vector<std::string> loc_one_site_labels = {"lsz_local" + file_postfix};
  if ((elec_two_site_ops.size() + 1) % mpi_size == rank) {
    MeasureOneSiteOp(mps, mps_path, loc_one_site_ops, odd_sites, loc_one_site_labels);
  }

  MPI_Finalize();
  return 0;
}


