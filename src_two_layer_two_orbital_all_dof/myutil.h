#ifndef HX_MYUTIL_H //hao-xin's myutil
#define HX_MYUTIL_H
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qlmps/qlmps.h"

bool IsElectron(size_t i, size_t Ly, size_t Np);
bool IsDx2Y2Site(size_t site, size_t Ly);
bool IsDz2Site(size_t site, size_t Ly);
size_t GetNumofMps();
void Show(std::vector<size_t> v);
std::vector<size_t> CollectOrbitalSites(size_t total_sites, size_t Ly, bool dz2_orbital);
std::vector<std::pair<size_t, size_t>> CollectOnsiteInterOrbitalBonds(size_t Lx, size_t Ly);
std::vector<std::pair<size_t, size_t>> CollectInterlayerDx2Y2Bonds(size_t Lx, size_t Ly);
size_t CanonicalizeInterlayerPairReferenceStart(size_t start,
                                                size_t total_sites,
                                                size_t Ly,
                                                bool dz2_orbital);
std::vector<size_t> BuildInitialOrbitalStateLabels(size_t orbital_site_count,
                                                   size_t electron_count);
std::vector<size_t> BuildInitialDz2StateLabels(size_t orbital_site_count,
                                               size_t electron_count,
                                               size_t Ly);
std::vector<size_t> BuildInitialDx2Y2StateLabels(size_t orbital_site_count,
                                                 size_t electron_count,
                                                 const std::vector<size_t> &dz2_state_labels);
std::vector<size_t> InterleaveOrbitalStateLabels(const std::vector<size_t> &orbital1_labels,
                                                 const std::vector<size_t> &orbital2_labels,
                                                 size_t Ly);
std::string OrbitalTag(bool dz2_orbital);
std::string BuildStageBackupPath(const std::string &source_path, const std::string &stage_tag);
void CopyDirectoryRecursively(const std::string &source_path, const std::string &destination_path);

template <typename AvgT, typename CombineFn>
qlmps::MeasuRes<AvgT> DeriveOneSiteObservableMeasuRes(const qlmps::MeasuRes<AvgT> &first_res,
                                                      const qlmps::MeasuRes<AvgT> &second_res,
                                                      CombineFn combine_fn) {
  if (first_res.size() != second_res.size()) {
    throw std::invalid_argument("Derived one-site observable requires equal result sizes.");
  }

  qlmps::MeasuRes<AvgT> derived_res;
  derived_res.reserve(first_res.size());
  for (size_t i = 0; i < first_res.size(); ++i) {
    if (first_res[i].sites != second_res[i].sites) {
      throw std::invalid_argument("Derived one-site observable requires aligned site indices.");
    }
    if (first_res[i].sites.size() != 1) {
      throw std::invalid_argument("Derived one-site observable expects one-site measurements.");
    }
    derived_res.emplace_back(first_res[i].sites,
                             combine_fn(first_res[i].avg, second_res[i].avg));
  }
  return derived_res;
}

template <typename AvgT>
qlmps::MeasuRes<AvgT> DeriveSingleOccupancyMeasuRes(const qlmps::MeasuRes<AvgT> &nf_res,
                                                    const qlmps::MeasuRes<AvgT> &doublon_res) {
  return DeriveOneSiteObservableMeasuRes(
      nf_res,
      doublon_res,
      [](const AvgT &nf, const AvgT &doublon) {
        return nf - static_cast<AvgT>(2.0) * doublon;
      });
}

template <typename AvgT>
qlmps::MeasuRes<AvgT> DeriveChargeVarianceMeasuRes(const qlmps::MeasuRes<AvgT> &nf_res,
                                                   const qlmps::MeasuRes<AvgT> &doublon_res) {
  return DeriveOneSiteObservableMeasuRes(
      nf_res,
      doublon_res,
      [](const AvgT &nf, const AvgT &doublon) {
        return nf + static_cast<AvgT>(2.0) * doublon - nf * nf;
      });
}

bool Parser(const int argc, char *argv[],
            size_t &start,
            size_t &end);
bool ParserBondDimension(int argc, char *argv[],
                         std::vector<size_t> &D_set);

bool ParserMeasureSite(const int argc, char *argv[],
                       size_t &start,
                       size_t &end);

#endif //HX_MYUTIL_H
