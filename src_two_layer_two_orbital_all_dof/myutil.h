#ifndef HX_MYUTIL_H //hao-xin's myutil
#define HX_MYUTIL_H
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

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
bool Parser(const int argc, char *argv[],
            size_t &start,
            size_t &end);
bool ParserBondDimension(int argc, char *argv[],
                         std::vector<size_t> &D_set);

bool ParserMeasureSite(const int argc, char *argv[],
                       size_t &start,
                       size_t &end);

#endif //HX_MYUTIL_H
