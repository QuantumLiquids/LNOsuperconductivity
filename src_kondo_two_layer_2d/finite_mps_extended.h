//
// Kondo-lattice specific extensions for finite MPS measurements
//
// This header provides two-site correlation measurement helpers that:
//  - enforce reference and target sites to be itinerant-electron sites (even indices)
//  - insert the fermionic string operator only on even (itinerant) intermediate sites
//  - leave odd (localized-spin) intermediate sites as identity
//
// The functions mirror the classic MeasureTwoSiteOpGroup API but are tailored for
// the two-layer Kondo lattice mapping used in this project.

#ifndef FINITE_MPS_EXTENDED
#define FINITE_MPS_EXTENDED

#include "qlten/qlten.h"
#include "qlmps/qlmps.h"
#include <vector>
#include <string>
#include <stdexcept>

// Measurement utilities and types (MeasuRes, DumpMeasuRes, ContractHeadSite, etc.)
#include "../src_tj_double_layer_single_orbital_2d/my_measure.h"

namespace qlmps {

/**
 * Measure a set of two-site correlations <O1(site1) S O2(site2)>, where S is the
 * fermionic string product applied only on even (itinerant) intermediate sites.
 *
 * Constraints:
 *  - site1 must be even
 *  - every site in site2_set must be even
 *  - inst_op is mandatory and will be inserted on even intermediate sites only
 */
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureTwoSiteOpGroupInKondoLattice(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string &mps_path,
    const qlten::QLTensor<TenElemT, QNT> &phys_ops1,
    const qlten::QLTensor<TenElemT, QNT> &phys_ops2,
    const size_t site1,
    const std::vector<size_t> &site2_set,
    const qlten::QLTensor<TenElemT, QNT> &inst_op) {
  if (site1 % 2 != 0) {
    throw std::invalid_argument("MeasureTwoSiteOpGroupInKondoLattice: site1 must be even (itinerant electron site)");
  }

  // Prepare MPS center at site1
  mps.LoadTen(mps_path, 0);
  for (size_t j = 0; j < site1; j++) {
    mps.LoadTen(j + 1, GenMPSTenName(mps_path, j + 1));
    mps.LeftCanonicalizeTen(j);
    mps.dealloc(j);
  }

  // Contract mps[site1] * phys_ops1 * dag(mps[site1])
  auto id_op_set = mps.GetSitesInfo().id_ops;
  auto ptemp_ten = new qlten::QLTensor<TenElemT, QNT>;
  *ptemp_ten = ContractHeadSite(mps[site1], phys_ops1);
  mps.dealloc(site1);

  size_t eated_site = site1; // last site has been contracted into ptemp_ten

  // Filter and collect only even targets (enforce constraint)
  std::vector<size_t> even_targets;
  even_targets.reserve(site2_set.size());
  for (size_t s : site2_set) {
    if (s % 2 == 0) even_targets.push_back(s);
  }

  MeasuRes<TenElemT> measure_res(even_targets.size());
  for (size_t event = 0; event < even_targets.size(); event++) {
    const size_t site2 = even_targets[event];
    while (eated_site < site2 - 1) {
      size_t eating_site = eated_site + 1;
      // Insert inst_op only on even (itinerant) intermediate sites; identity otherwise
      const auto &left_op = (eating_site % 2 == 0) ? inst_op : id_op_set[eating_site];
      CtrctMidTen(mps, eating_site, left_op, id_op_set[eating_site], ptemp_ten, mps_path);
      eated_site = eating_site;
    }

    // now site2-1 has been eaten
    mps.LoadTen(site2, GenMPSTenName(mps_path, site2));
    auto avg = ContractTailSite(mps[site2], phys_ops2, *ptemp_ten);
    measure_res[event] = MeasuResElem<TenElemT>({site1, site2}, avg);
    mps.dealloc(site2);
  }
  delete ptemp_ten;
  return measure_res;
}

/**
 * Measure two-site correlations from a ref_site to all later even sites:
 *   targets = { ref_site + 2, ref_site + 4, ... }
 *
 * Constraints:
 *  - ref_site must be even
 *  - inst_op is mandatory and is inserted on even intermediate sites only
 */
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureTwoSiteOpGroupInKondoLattice(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string &mps_path,
    const qlten::QLTensor<TenElemT, QNT> &phys_ops1,
    const qlten::QLTensor<TenElemT, QNT> &phys_ops2,
    const size_t ref_site,
    const qlten::QLTensor<TenElemT, QNT> &inst_op) {
  if (ref_site % 2 != 0) {
    throw std::invalid_argument("MeasureTwoSiteOpGroupInKondoLattice: ref_site must be even (itinerant electron site)");
  }
  const size_t N = mps.size();
  std::vector<size_t> site2_set;
  site2_set.reserve((N > ref_site) ? (N - ref_site) / 2 : 0);
  for (size_t site = ref_site + 2; site < N; site += 2) {
    site2_set.push_back(site);
  }
  return MeasureTwoSiteOpGroupInKondoLattice(mps, mps_path, phys_ops1, phys_ops2, ref_site, site2_set, inst_op);
}

/**
 * Same as above but automatically dumps the results with filename_base.
 */
template<typename TenElemT, typename QNT>
MeasuRes<TenElemT> MeasureTwoSiteOpGroupInKondoLattice(
    FiniteMPS<TenElemT, QNT> &mps,
    const std::string &mps_path,
    const qlten::QLTensor<TenElemT, QNT> &phys_ops1,
    const qlten::QLTensor<TenElemT, QNT> &phys_ops2,
    const size_t ref_site,
    const qlten::QLTensor<TenElemT, QNT> &inst_op,
    const std::string &filename_base) {
  auto res = MeasureTwoSiteOpGroupInKondoLattice(mps, mps_path, phys_ops1, phys_ops2, ref_site, inst_op);
  DumpMeasuRes(res, filename_base);
  return res;
}

} // namespace qlmps

#endif

