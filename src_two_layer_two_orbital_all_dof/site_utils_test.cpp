#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <stdexcept>
#include <vector>

#include "myutil.h"

namespace {

bool CheckCounts(const std::vector<size_t> &state_labels,
                 size_t expected_electrons,
                 size_t expected_double_occ) {
  size_t electron_count = 0;
  size_t double_occ_count = 0;
  for (size_t label : state_labels) {
    if (label == 0) {
      electron_count += 2;
      double_occ_count += 1;
    } else if (label == 1 || label == 2) {
      electron_count += 1;
    }
  }
  return electron_count == expected_electrons &&
         double_occ_count == expected_double_occ;
}

std::vector<size_t> OccupiedSites(const std::vector<size_t> &state_labels) {
  std::vector<size_t> occupied_sites;
  for (size_t site = 0; site < state_labels.size(); ++site) {
    if (state_labels[site] != 3) {
      occupied_sites.push_back(site);
    }
  }
  return occupied_sites;
}

int TotalSzTwice(const std::vector<size_t> &state_labels) {
  int total_sz_twice = 0;
  for (size_t label : state_labels) {
    if (label == 1) {
      total_sz_twice += 1;
    } else if (label == 2) {
      total_sz_twice -= 1;
    }
  }
  return total_sz_twice;
}

bool SingleOccupanciesParallel(const std::vector<size_t> &dx2y2_labels,
                               const std::vector<size_t> &dz2_labels) {
  if (dx2y2_labels.size() != dz2_labels.size()) {
    return false;
  }
  for (size_t site = 0; site < dx2y2_labels.size(); ++site) {
    if ((dx2y2_labels[site] == 1 || dx2y2_labels[site] == 2) &&
        dx2y2_labels[site] != dz2_labels[site]) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr size_t kLy = 2;
  constexpr size_t kTotalSites = 16;

  const auto dx2y2_sites = CollectOrbitalSites(kTotalSites, kLy, false);
  const auto dz2_sites = CollectOrbitalSites(kTotalSites, kLy, true);
  if (dx2y2_sites != std::vector<size_t>({0, 1, 2, 3, 8, 9, 10, 11})) {
    std::cerr << "Unexpected d_x2-y2 site list." << std::endl;
    return EXIT_FAILURE;
  }
  if (dz2_sites != std::vector<size_t>({4, 5, 6, 7, 12, 13, 14, 15})) {
    std::cerr << "Unexpected d_z2 site list." << std::endl;
    return EXIT_FAILURE;
  }

  if (CollectOnsiteInterOrbitalBonds(2, kLy) !=
      std::vector<std::pair<size_t, size_t>>(
          {{0, 4}, {1, 5}, {2, 6}, {3, 7}, {8, 12}, {9, 13}, {10, 14}, {11, 15}})) {
    std::cerr << "Unexpected onsite inter-orbital bond list." << std::endl;
    return EXIT_FAILURE;
  }
  if (CollectInterlayerDx2Y2Bonds(2, kLy) !=
      std::vector<std::pair<size_t, size_t>>({{0, 2}, {1, 3}, {8, 10}, {9, 11}})) {
    std::cerr << "Unexpected d_x2-y^2 interlayer bond list." << std::endl;
    return EXIT_FAILURE;
  }
  if (CanonicalizeInterlayerPairReferenceStart(0, kTotalSites, kLy, false) != 0 ||
      CanonicalizeInterlayerPairReferenceStart(1, kTotalSites, kLy, false) != 1) {
    std::cerr << "d_x2-y^2 reference starts should stay on the d_x2-y^2 sublattice." << std::endl;
    return EXIT_FAILURE;
  }
  if (CanonicalizeInterlayerPairReferenceStart(0, kTotalSites, kLy, true) != 4 ||
      CanonicalizeInterlayerPairReferenceStart(1, kTotalSites, kLy, true) != 5) {
    std::cerr << "d_z2 reference starts should map onto the d_z2 sublattice." << std::endl;
    return EXIT_FAILURE;
  }
  bool caught_invalid_start = false;
  try {
    (void)CanonicalizeInterlayerPairReferenceStart(2, kTotalSites, kLy, false);
  } catch (const std::invalid_argument &) {
    caught_invalid_start = true;
  }
  if (!caught_invalid_start) {
    std::cerr << "Bottom-layer reference starts should be rejected." << std::endl;
    return EXIT_FAILURE;
  }

  const qlmps::MeasuRes<double> nf_res = {
      {{4}, 1.0},
      {{5}, 0.9}
  };
  const qlmps::MeasuRes<double> doublon_res = {
      {{4}, 0.1},
      {{5}, 0.05}
  };
  const auto single_occ_res = DeriveSingleOccupancyMeasuRes(nf_res, doublon_res);
  const auto charge_var_res = DeriveChargeVarianceMeasuRes(nf_res, doublon_res);
  if (single_occ_res.size() != 2 ||
      single_occ_res[0].sites != std::vector<size_t>({4}) ||
      single_occ_res[1].sites != std::vector<size_t>({5}) ||
      std::abs(single_occ_res[0].avg - 0.8) > 1e-12 ||
      std::abs(single_occ_res[1].avg - 0.8) > 1e-12) {
    std::cerr << "Derived single-occupancy observable is incorrect." << std::endl;
    return EXIT_FAILURE;
  }
  if (charge_var_res.size() != 2 ||
      std::abs(charge_var_res[0].avg - 0.2) > 1e-12 ||
      std::abs(charge_var_res[1].avg - 0.19) > 1e-12) {
    std::cerr << "Derived charge-variance observable is incorrect." << std::endl;
    return EXIT_FAILURE;
  }
  bool caught_misaligned_sites = false;
  try {
    (void)DeriveSingleOccupancyMeasuRes(
        nf_res,
        qlmps::MeasuRes<double>{{{6}, 0.1}, {{5}, 0.05}});
  } catch (const std::invalid_argument &) {
    caught_misaligned_sites = true;
  }
  if (!caught_misaligned_sites) {
    std::cerr << "Derived observables should reject misaligned site indices." << std::endl;
    return EXIT_FAILURE;
  }
  if (BuildStageBackupPath("mps", "GeometryOBC_stage2_Dmax40") !=
      "mps_GeometryOBC_stage2_Dmax40") {
    std::cerr << "Backup path should append the stage tag to the MPS path." << std::endl;
    return EXIT_FAILURE;
  }
  namespace fs = std::filesystem;
  const fs::path source_dir = fs::path("/tmp") / "lno_4band_backup_source";
  const fs::path backup_dir = fs::path("/tmp") / "lno_4band_backup_target";
  fs::remove_all(source_dir);
  fs::remove_all(backup_dir);
  fs::create_directories(source_dir / "nested");
  {
    std::ofstream(source_dir / "mps0.txt") << "stage-one";
    std::ofstream(source_dir / "nested" / "mps1.txt") << "stage-one-nested";
  }
  CopyDirectoryRecursively(source_dir.string(), backup_dir.string());
  if (!fs::exists(backup_dir / "mps0.txt") || !fs::exists(backup_dir / "nested" / "mps1.txt")) {
    std::cerr << "Backup should copy the full MPS directory tree." << std::endl;
    return EXIT_FAILURE;
  }
  {
    std::ofstream(source_dir / "mps0.txt") << "stage-two";
  }
  CopyDirectoryRecursively(source_dir.string(), backup_dir.string());
  std::ifstream backup_file(backup_dir / "mps0.txt");
  std::string backup_contents;
  backup_file >> backup_contents;
  if (backup_contents != "stage-two") {
    std::cerr << "Backup should overwrite stale MPS directories." << std::endl;
    return EXIT_FAILURE;
  }
  fs::remove_all(source_dir);
  fs::remove_all(backup_dir);

  const auto quarter_filled = BuildInitialOrbitalStateLabels(8, 4);
  if (!CheckCounts(quarter_filled, 4, 0)) {
    std::cerr << "Quarter-filled state labels are inconsistent." << std::endl;
    return EXIT_FAILURE;
  }
  if (OccupiedSites(quarter_filled) != std::vector<size_t>({0, 2, 4, 6})) {
    std::cerr << "Quarter-filled state should occupy every other site." << std::endl;
    return EXIT_FAILURE;
  }
  if (TotalSzTwice(quarter_filled) != 0) {
    std::cerr << "Quarter-filled state should have total Sz = 0." << std::endl;
    return EXIT_FAILURE;
  }

  const auto half_filled = BuildInitialOrbitalStateLabels(8, 8);
  if (!CheckCounts(half_filled, 8, 0)) {
    std::cerr << "Half-filled state should start without double occupancy." << std::endl;
    return EXIT_FAILURE;
  }
  if (OccupiedSites(half_filled) != std::vector<size_t>({0, 1, 2, 3, 4, 5, 6, 7})) {
    std::cerr << "Half-filled state should singly occupy every site." << std::endl;
    return EXIT_FAILURE;
  }
  if (TotalSzTwice(half_filled) != 0) {
    std::cerr << "Half-filled state should have total Sz = 0." << std::endl;
    return EXIT_FAILURE;
  }

  const auto dz2_default_pattern = BuildInitialDz2StateLabels(8, 8, kLy);
  if (dz2_default_pattern != std::vector<size_t>({1, 2, 2, 1, 1, 2, 2, 1})) {
    std::cerr << "Default d_z2 pattern should alternate by y and invert between layers." << std::endl;
    return EXIT_FAILURE;
  }
  if (!CheckCounts(dz2_default_pattern, 8, 0)) {
    std::cerr << "Default d_z2 pattern should remain singly occupied." << std::endl;
    return EXIT_FAILURE;
  }
  if (TotalSzTwice(dz2_default_pattern) != 0) {
    std::cerr << "Default d_z2 pattern should have total Sz = 0." << std::endl;
    return EXIT_FAILURE;
  }

  const auto dx2y2_hund_quarter = BuildInitialDx2Y2StateLabels(8, 4, dz2_default_pattern);
  if (!CheckCounts(dx2y2_hund_quarter, 4, 0)) {
    std::cerr << "Hund-aligned d_x2-y2 quarter filling has the wrong electron count." << std::endl;
    return EXIT_FAILURE;
  }
  if (!SingleOccupanciesParallel(dx2y2_hund_quarter, dz2_default_pattern)) {
    std::cerr << "Quarter-filled d_x2-y2 singles should align with d_z2." << std::endl;
    return EXIT_FAILURE;
  }
  if (TotalSzTwice(dx2y2_hund_quarter) != 0) {
    std::cerr << "Quarter-filled d_x2-y2 pattern should keep total Sz = 0." << std::endl;
    return EXIT_FAILURE;
  }

  const auto dx2y2_hund_half = BuildInitialDx2Y2StateLabels(8, 8, dz2_default_pattern);
  if (dx2y2_hund_half != dz2_default_pattern) {
    std::cerr << "Half-filled d_x2-y2 pattern should match d_z2 exactly." << std::endl;
    return EXIT_FAILURE;
  }

  const auto dx2y2_hund_two = BuildInitialDx2Y2StateLabels(8, 2, dz2_default_pattern);
  if (!CheckCounts(dx2y2_hund_two, 2, 0)) {
    std::cerr << "Low-filled Hund-aligned d_x2-y2 state has the wrong electron count." << std::endl;
    return EXIT_FAILURE;
  }
  if (!SingleOccupanciesParallel(dx2y2_hund_two, dz2_default_pattern)) {
    std::cerr << "Low-filled d_x2-y2 singles should align with d_z2." << std::endl;
    return EXIT_FAILURE;
  }
  if (TotalSzTwice(dx2y2_hund_two) != 0) {
    std::cerr << "Low-filled d_x2-y2 state should keep total Sz = 0." << std::endl;
    return EXIT_FAILURE;
  }

  const auto dx2y2_hund_overdoped = BuildInitialDx2Y2StateLabels(8, 12, dz2_default_pattern);
  if (!CheckCounts(dx2y2_hund_overdoped, 12, 4)) {
    std::cerr << "Overdoped Hund-aligned d_x2-y2 state has the wrong electron count." << std::endl;
    return EXIT_FAILURE;
  }
  if (!SingleOccupanciesParallel(dx2y2_hund_overdoped, dz2_default_pattern)) {
    std::cerr << "Overdoped d_x2-y2 singles should align with d_z2." << std::endl;
    return EXIT_FAILURE;
  }

  const auto overdoped = BuildInitialOrbitalStateLabels(8, 12);
  if (!CheckCounts(overdoped, 12, 4)) {
    std::cerr << "Overfilled state labels should have the minimal double occupancy." << std::endl;
    return EXIT_FAILURE;
  }

  const auto interleaved = InterleaveOrbitalStateLabels(quarter_filled, half_filled, kLy);
  if (interleaved.size() != kTotalSites) {
    std::cerr << "Interleaved state has the wrong size." << std::endl;
    return EXIT_FAILURE;
  }

  if (!std::equal(interleaved.begin(), interleaved.begin() + 4, quarter_filled.begin()) ||
      !std::equal(interleaved.begin() + 4, interleaved.begin() + 8, half_filled.begin())) {
    std::cerr << "Interleaving does not preserve the per-orbital chunk order." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
