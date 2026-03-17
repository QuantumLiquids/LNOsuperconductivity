#include "qlmps/qlmps.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

using qlmps::kMpsPath;
using qlmps::kMpsTenBaseName;
using qlmps::kQLTenFileSuffix;

//number of mps file in default mps path("./mps")
size_t GetNumofMps() {
  size_t NumberOfMpsFile = 0;
  for (NumberOfMpsFile = 0; NumberOfMpsFile < 1e5; NumberOfMpsFile++) {
    std::string file;
    file = kMpsPath + "/" + kMpsTenBaseName + std::to_string(NumberOfMpsFile) + "." + kQLTenFileSuffix;
    std::ifstream ifs(file, std::ifstream::binary);
    if (ifs.good()) {
      ifs.close();
    } else {
      break;
    }
  }
  return NumberOfMpsFile;
}

//If site `i` is electron site, for SSH-Hubbard model
bool IsElectron(size_t i, size_t Ly, size_t Np) {
  size_t residue = i % ((2 * Np + 1) * Ly);
  if (residue < (Np + 1) * Ly && residue % (Np + 1) == 0) {
    return true;
  } else return false;
}

bool IsDx2Y2Site(size_t site, size_t Ly) {
  return ((site / (2 * Ly)) % 2) == 0;
}

bool IsDz2Site(size_t site, size_t Ly) {
  return !IsDx2Y2Site(site, Ly);
}

void Show(std::vector<size_t> v) {
  for (auto iter = v.begin(); iter < v.end(); iter++) {
    std::cout << *iter << ",";
  }
  std::cout << '\b' << std::endl;
}

std::vector<size_t> CollectOrbitalSites(size_t total_sites, size_t Ly, bool dz2_orbital) {
  std::vector<size_t> sites;
  sites.reserve(total_sites / 2);
  for (size_t site = 0; site < total_sites; ++site) {
    if (IsDz2Site(site, Ly) == dz2_orbital) {
      sites.push_back(site);
    }
  }
  return sites;
}

std::vector<std::pair<size_t, size_t>> CollectOnsiteInterOrbitalBonds(size_t Lx, size_t Ly) {
  std::vector<std::pair<size_t, size_t>> bonds;
  bonds.reserve(2 * Lx * Ly);
  const size_t orbital_block_size = 2 * Ly;
  const size_t unit_cell_size = 2 * orbital_block_size;
  for (size_t x = 0; x < Lx; ++x) {
    const size_t dx2y2_block_start = x * unit_cell_size;
    const size_t dz2_block_start = dx2y2_block_start + orbital_block_size;
    for (size_t y = 0; y < orbital_block_size; ++y) {
      bonds.emplace_back(dx2y2_block_start + y, dz2_block_start + y);
    }
  }
  return bonds;
}

std::vector<std::pair<size_t, size_t>> CollectInterlayerDx2Y2Bonds(size_t Lx, size_t Ly) {
  std::vector<std::pair<size_t, size_t>> bonds;
  bonds.reserve(Lx * Ly);
  const size_t unit_cell_size = 4 * Ly;
  for (size_t x = 0; x < Lx; ++x) {
    const size_t dx2y2_block_start = x * unit_cell_size;
    for (size_t y = 0; y < Ly; ++y) {
      bonds.emplace_back(dx2y2_block_start + y, dx2y2_block_start + y + Ly);
    }
  }
  return bonds;
}

size_t CanonicalizeInterlayerPairReferenceStart(size_t start,
                                                size_t total_sites,
                                                size_t Ly,
                                                bool dz2_orbital) {
  if (Ly == 0) {
    throw std::invalid_argument("Ly must be positive.");
  }
  if (start >= total_sites) {
    throw std::invalid_argument("Reference start is out of bounds.");
  }

  const size_t orbital_block_size = 2 * Ly;
  if ((start % orbital_block_size) >= Ly) {
    throw std::invalid_argument("Reference start must be the top-layer site of an interlayer bond.");
  }

  size_t canonical_start = start;
  if (IsDz2Site(start, Ly) != dz2_orbital) {
    const size_t orbital_shift = 2 * Ly;
    if (dz2_orbital) {
      canonical_start += orbital_shift;
    } else {
      if (start < orbital_shift) {
        throw std::invalid_argument("Cannot shift reference start onto the requested orbital.");
      }
      canonical_start -= orbital_shift;
    }
  }

  if (canonical_start >= total_sites || canonical_start + Ly >= total_sites) {
    throw std::invalid_argument("Canonical reference bond is out of bounds.");
  }
  if ((canonical_start % orbital_block_size) >= Ly) {
    throw std::invalid_argument("Canonical reference start must stay on the top layer.");
  }
  if (IsDz2Site(canonical_start, Ly) != dz2_orbital) {
    throw std::invalid_argument("Canonical reference start does not match the requested orbital.");
  }
  return canonical_start;
}

namespace {

std::vector<size_t> SelectEvenlySpacedSites(const std::vector<size_t> &candidate_sites,
                                            size_t selection_count) {
  if (selection_count > candidate_sites.size()) {
    throw std::invalid_argument("Cannot select more sites than available candidates.");
  }

  std::vector<size_t> selected_sites;
  selected_sites.reserve(selection_count);
  for (size_t idx = 0; idx < selection_count; ++idx) {
    const size_t candidate_index =
        (idx * candidate_sites.size() + selection_count / 2) / selection_count;
    selected_sites.push_back(candidate_sites[candidate_index]);
  }
  return selected_sites;
}

std::pair<size_t, size_t> FindClosestSwapPair(const std::vector<size_t> &selected_sites,
                                              const std::vector<size_t> &unselected_sites) {
  if (selected_sites.empty() || unselected_sites.empty()) {
    throw std::invalid_argument("Cannot swap without both selected and unselected candidates.");
  }

  std::pair<size_t, size_t> best_pair(selected_sites.front(), unselected_sites.front());
  size_t best_distance = std::max(best_pair.first, best_pair.second) - std::min(best_pair.first, best_pair.second);
  for (size_t selected_site : selected_sites) {
    for (size_t unselected_site : unselected_sites) {
      const size_t distance = std::max(selected_site, unselected_site) - std::min(selected_site, unselected_site);
      if (distance < best_distance ||
          (distance == best_distance && std::pair<size_t, size_t>(selected_site, unselected_site) < best_pair)) {
        best_pair = {selected_site, unselected_site};
        best_distance = distance;
      }
    }
  }
  return best_pair;
}

}  // namespace

std::vector<size_t> BuildInitialOrbitalStateLabels(size_t orbital_site_count,
                                                   size_t electron_count) {
  if (electron_count > 2 * orbital_site_count) {
    throw std::invalid_argument("Number of electrons exceeds orbital capacity.");
  }
  if (electron_count % 2 != 0) {
    throw std::invalid_argument("Only even electron counts are supported for spin-balanced initialization.");
  }

  const size_t num_up = electron_count / 2;
  const size_t num_down = electron_count / 2;
  std::vector<size_t> state_labels(orbital_site_count, 3);  // 0: doublon, 1: up, 2: down, 3: empty
  std::vector<size_t> all_sites(orbital_site_count);
  std::iota(all_sites.begin(), all_sites.end(), 0);

  const size_t num_single_occupancies = std::min(electron_count, orbital_site_count);
  const auto singly_occupied_sites = SelectEvenlySpacedSites(all_sites, num_single_occupancies);
  for (size_t idx = 0; idx < singly_occupied_sites.size(); ++idx) {
    state_labels[singly_occupied_sites[idx]] = (idx % 2 == 0) ? 1 : 2;
  }

  size_t num_up_singles = 0;
  size_t num_down_singles = 0;
  std::vector<size_t> up_sites;
  std::vector<size_t> down_sites;
  for (size_t site = 0; site < orbital_site_count; ++site) {
    if (state_labels[site] == 1) {
      ++num_up_singles;
      up_sites.push_back(site);
    } else if (state_labels[site] == 2) {
      ++num_down_singles;
      down_sites.push_back(site);
    }
  }

  size_t remaining_up = num_up - num_up_singles;
  size_t remaining_down = num_down - num_down_singles;

  const auto doublon_on_up_sites = SelectEvenlySpacedSites(up_sites, remaining_down);
  for (size_t site : doublon_on_up_sites) {
    state_labels[site] = 0;
  }
  const auto doublon_on_down_sites = SelectEvenlySpacedSites(down_sites, remaining_up);
  for (size_t site : doublon_on_down_sites) {
    state_labels[site] = 0;
  }

  return state_labels;
}

std::vector<size_t> BuildInitialDz2StateLabels(size_t orbital_site_count,
                                               size_t electron_count,
                                               size_t Ly) {
  if (electron_count != orbital_site_count) {
    return BuildInitialOrbitalStateLabels(orbital_site_count, electron_count);
  }

  const size_t chunk_size = 2 * Ly;
  if (orbital_site_count % chunk_size != 0) {
    throw std::invalid_argument("d_z2 state labels are inconsistent with Ly.");
  }

  std::vector<size_t> state_labels(orbital_site_count, 3);
  for (size_t offset = 0; offset < orbital_site_count; offset += chunk_size) {
    for (size_t y = 0; y < Ly; ++y) {
      const bool top_layer_spin_up = (y % 2) == 0;
      state_labels[offset + y] = top_layer_spin_up ? 1 : 2;
      state_labels[offset + Ly + y] = top_layer_spin_up ? 2 : 1;
    }
  }
  return state_labels;
}

std::vector<size_t> BuildInitialDx2Y2StateLabels(size_t orbital_site_count,
                                                 size_t electron_count,
                                                 const std::vector<size_t> &dz2_state_labels) {
  if (dz2_state_labels.size() != orbital_site_count) {
    throw std::invalid_argument("d_x2-y2 and d_z2 state labels must have the same size.");
  }

  for (size_t label : dz2_state_labels) {
    if (label != 1 && label != 2) {
      return BuildInitialOrbitalStateLabels(orbital_site_count, electron_count);
    }
  }

  if (electron_count > 2 * orbital_site_count) {
    throw std::invalid_argument("Number of electrons exceeds orbital capacity.");
  }
  if (electron_count % 2 != 0) {
    throw std::invalid_argument("Only even electron counts are supported for spin-balanced initialization.");
  }

  const size_t total_up = electron_count / 2;
  const size_t total_down = electron_count / 2;
  const size_t num_single_occupancies = std::min(electron_count, orbital_site_count);

  std::vector<size_t> all_sites(orbital_site_count);
  std::iota(all_sites.begin(), all_sites.end(), 0);
  const auto initially_selected_sites = SelectEvenlySpacedSites(all_sites, num_single_occupancies);
  std::vector<bool> is_selected(orbital_site_count, false);
  for (size_t site : initially_selected_sites) {
    is_selected[site] = true;
  }

  auto collect_selected_sites_with_spin = [&](size_t spin_label) {
    std::vector<size_t> sites;
    for (size_t site = 0; site < orbital_site_count; ++site) {
      if (is_selected[site] && dz2_state_labels[site] == spin_label) {
        sites.push_back(site);
      }
    }
    return sites;
  };
  auto collect_unselected_sites_with_spin = [&](size_t spin_label) {
    std::vector<size_t> sites;
    for (size_t site = 0; site < orbital_site_count; ++site) {
      if (!is_selected[site] && dz2_state_labels[site] == spin_label) {
        sites.push_back(site);
      }
    }
    return sites;
  };

  while (collect_selected_sites_with_spin(1).size() > total_up) {
    const auto selected_up_sites = collect_selected_sites_with_spin(1);
    const auto unselected_down_sites = collect_unselected_sites_with_spin(2);
    if (unselected_down_sites.empty()) {
      return BuildInitialOrbitalStateLabels(orbital_site_count, electron_count);
    }
    const auto [selected_site, unselected_site] = FindClosestSwapPair(selected_up_sites, unselected_down_sites);
    is_selected[selected_site] = false;
    is_selected[unselected_site] = true;
  }

  while (collect_selected_sites_with_spin(2).size() > total_down) {
    const auto selected_down_sites = collect_selected_sites_with_spin(2);
    const auto unselected_up_sites = collect_unselected_sites_with_spin(1);
    if (unselected_up_sites.empty()) {
      return BuildInitialOrbitalStateLabels(orbital_site_count, electron_count);
    }
    const auto [selected_site, unselected_site] = FindClosestSwapPair(selected_down_sites, unselected_up_sites);
    is_selected[selected_site] = false;
    is_selected[unselected_site] = true;
  }

  std::vector<size_t> state_labels(orbital_site_count, 3);
  size_t num_up_singles = 0;
  size_t num_down_singles = 0;
  std::vector<size_t> up_sites;
  std::vector<size_t> down_sites;
  for (size_t site = 0; site < orbital_site_count; ++site) {
    if (is_selected[site]) {
      state_labels[site] = dz2_state_labels[site];
      if (state_labels[site] == 1) {
        ++num_up_singles;
        up_sites.push_back(site);
      } else {
        ++num_down_singles;
        down_sites.push_back(site);
      }
    }
  }

  if (num_up_singles > total_up || num_down_singles > total_down) {
    return BuildInitialOrbitalStateLabels(orbital_site_count, electron_count);
  }

  const size_t remaining_up = total_up - num_up_singles;
  const size_t remaining_down = total_down - num_down_singles;

  const auto doublon_on_up_sites = SelectEvenlySpacedSites(up_sites, remaining_down);
  for (size_t site : doublon_on_up_sites) {
    state_labels[site] = 0;
  }
  const auto doublon_on_down_sites = SelectEvenlySpacedSites(down_sites, remaining_up);
  for (size_t site : doublon_on_down_sites) {
    state_labels[site] = 0;
  }

  return state_labels;
}

std::vector<size_t> InterleaveOrbitalStateLabels(const std::vector<size_t> &orbital1_labels,
                                                 const std::vector<size_t> &orbital2_labels,
                                                 size_t Ly) {
  if (orbital1_labels.size() != orbital2_labels.size()) {
    throw std::invalid_argument("The two orbitals must have the same number of sites.");
  }
  const size_t chunk_size = 2 * Ly;
  if (orbital1_labels.size() % chunk_size != 0) {
    throw std::invalid_argument("Orbital state labels are inconsistent with Ly.");
  }

  std::vector<size_t> state_labels(orbital1_labels.size() + orbital2_labels.size());
  size_t index = 0;
  for (size_t offset = 0; offset < orbital1_labels.size(); offset += chunk_size) {
    std::copy(orbital1_labels.begin() + offset,
              orbital1_labels.begin() + offset + chunk_size,
              state_labels.begin() + index);
    index += chunk_size;
    std::copy(orbital2_labels.begin() + offset,
              orbital2_labels.begin() + offset + chunk_size,
              state_labels.begin() + index);
    index += chunk_size;
  }
  return state_labels;
}

std::string OrbitalTag(bool dz2_orbital) {
  return dz2_orbital ? "dz2" : "dx2y2";
}

// When used to measure, note if should not set start too small to exceed canonical center.
bool Parser(const int argc, char *argv[],
            size_t &start,
            size_t &end) {
  int nOptionIndex = 1;

  std::string arguement1 = "--start=";
  std::string arguement2 = "--end=";
  bool start_argument_has(false), end_argument_has(false);
  while (nOptionIndex < argc) {
    if (strncmp(argv[nOptionIndex], arguement1.c_str(), arguement1.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement1.size()];
      start = atoi(para_string.c_str());
      start_argument_has = true;
    } else if (strncmp(argv[nOptionIndex], arguement2.c_str(), arguement2.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement2.size()];
      end = atoi(para_string.c_str());
      end_argument_has = true;
    }
    nOptionIndex++;
  }

  if (start_argument_has != end_argument_has) {
    std::cout << "Only setting one start/end argument, exit(1)." << std::endl;
    exit(1);
  }

  if (!start_argument_has) {
    std::cout << "Note: no start/end argument, set it by default (L/4, 3*L/4)." << std::endl;
  }

  return start_argument_has;
}

bool ParserBondDimension(int argc, char *argv[],
                         std::vector<size_t> &D_set) {
  int nOptionIndex = 1;
  std::string D_string;
  std::string arguement1 = "--D=";
  bool has_D_parameter(false);
  while (nOptionIndex < argc) {
    if (strncmp(argv[nOptionIndex], arguement1.c_str(), arguement1.size()) == 0) {
      D_string = &argv[nOptionIndex][arguement1.size()];
      has_D_parameter = true;
    }
    nOptionIndex++;
  }

  //split thread num list
  const char *split = ",";
  char *p;
  const size_t MAX_CHAR_LENTH = 1000;
  char D_char[MAX_CHAR_LENTH];
  for (size_t i = 0; i < MAX_CHAR_LENTH; i++) {
    D_char[i] = 0;
  }

  strcpy(D_char, D_string.c_str());

  p = strtok(D_char, split);
  while (p != nullptr) {
    D_set.push_back(atoi(p));
    p = strtok(nullptr, split);
  }

  return has_D_parameter;
}

bool ParserMeasureSite(const int argc, char *argv[],
                       size_t &start,
                       size_t &end) {
  int nOptionIndex = 1;

  std::string arguement1 = "--start=";
  std::string arguement2 = "--end=";
  bool start_argument_has(false), end_argument_has(false);
  while (nOptionIndex < argc) {
    if (strncmp(argv[nOptionIndex], arguement1.c_str(), arguement1.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement1.size()];
      start = atoi(para_string.c_str());
      start_argument_has = true;
    } else if (strncmp(argv[nOptionIndex], arguement2.c_str(), arguement2.size()) == 0) {
      std::string para_string = &argv[nOptionIndex][arguement2.size()];
      end = atoi(para_string.c_str());
      end_argument_has = true;
    }
    nOptionIndex++;
  }

  if (start_argument_has != end_argument_has) {
    std::cout << "Only setting one start/end argument, exit(1)." << std::endl;
    exit(1);
  }

  if (!start_argument_has) {
    std::cout << "Note: no start/end argument, set it by default (L/4, 3*L/4+2)." << std::endl;
  }

  return start_argument_has;
}
