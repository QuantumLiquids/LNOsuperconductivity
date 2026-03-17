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
  qlmps::HubbardOperators<TenElemT, QNT>  ops;

  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);

  FiniteMPST mps(sites);
  const std::vector<size_t> dx2y2_sites = CollectOrbitalSites(N, params.Ly, false);
  const std::vector<size_t> dz2_sites = CollectOrbitalSites(N, params.Ly, true);

  Timer one_site_timer("measure one site operators");
  const std::vector<QLTensor<TenElemT, QNT>> one_site_ops = {ops.sz, ops.nf, ops.nupndn};
  for (bool dz2_orbital : {false, true}) {
    const auto &measurement_sites = dz2_orbital ? dz2_sites : dx2y2_sites;
    const std::string orbital_tag = OrbitalTag(dz2_orbital);
    MeasureOneSiteOp(mps,
                     kMpsPath,
                     one_site_ops,
                     measurement_sites,
                     {"sz_" + orbital_tag, "nf_" + orbital_tag, "nupndn_" + orbital_tag});
  }
  std::cout << "measured one point function.<====" << std::endl;
  one_site_timer.PrintElapsed();
  for (bool dz2_orbital : {false, true}) {
    const auto &orbital_sites = dz2_orbital ? dz2_sites : dx2y2_sites;
    if (orbital_sites.size() < 2) {
      continue;
    }
    const std::string orbital_tag = OrbitalTag(dz2_orbital);
    const size_t ref_site = orbital_sites[orbital_sites.size() / 2];
    std::vector<size_t> target_sites;
    for (size_t site : orbital_sites) {
      if (site > ref_site) {
        target_sites.push_back(site);
      }
    }
    auto szsz_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, ops.sz, ops.sz, ref_site, target_sites);
    DumpMeasuRes(szsz_corr, "szsz_" + orbital_tag + "_ref" + std::to_string(ref_site));
    auto spsm_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, ops.sp, ops.sm, ref_site, target_sites);
    DumpMeasuRes(spsm_corr, "spsm_" + orbital_tag + "_ref" + std::to_string(ref_site));
    auto smsp_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, ops.sm, ops.sp, ref_site, target_sites);
    DumpMeasuRes(smsp_corr, "smsp_" + orbital_tag + "_ref" + std::to_string(ref_site));
    auto nn_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, ops.nf, ops.nf, ref_site, target_sites);
    DumpMeasuRes(nn_corr, "nfnf_" + orbital_tag + "_ref" + std::to_string(ref_site));
    auto doublon_corr = MeasureTwoSiteOpGroup(mps, kMpsPath, ops.nupndn, ops.nupndn, ref_site, target_sites);
    DumpMeasuRes(doublon_corr, "nupndn_nupndn_" + orbital_tag + "_ref" + std::to_string(ref_site));
  }
  endTime = clock();
  std::cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
  return 0;
}
