#ifndef KONDO_TWO_LEG_PARAMS_CASE_H
#define KONDO_TWO_LEG_PARAMS_CASE_H

#include <stdexcept>
#include <string>

#include "qlmps/qlmps.h"
using qlmps::CaseParamsParserBasic;

struct CaseParams : public CaseParamsParserBasic {
  CaseParams(const char *pf) : CaseParamsParserBasic(pf) {
    Geometry = ParseStr("Geometry");
    InitState = ParseStrOr("InitState", "random");
    Lx = ParseInt("Lx");
    Ly = ParseIntOr("Ly", 2);
    t = ParseDouble("t");
    t2 = ParseDouble("t2");
    JK = ParseDouble("Jk");
    U = ParseDouble("U");
    MpsPath = ParseStrOr("MpsPath", "");
    const int num_hole_raw = ParseIntOr("NumHole", 0);
    if (num_hole_raw < 0) {
      throw std::invalid_argument("NumHole must be non-negative");
    }
    NumHole = static_cast<size_t>(num_hole_raw);
    noise = ParseDoubleVec("noise");
    Sweeps = ParseInt("Sweeps");
    Dmin = ParseInt("Dmin");
    Dmax = ParseSizeTVec("Dmax");
    CutOff = ParseDouble("CutOff");
    LanczErr = ParseDouble("LanczErr");
    MaxLanczIter = ParseInt("MaxLanczIter");
    Threads = ParseInt("Threads");
    ValidateFilling();
  }

  std::string Geometry; // PBC, OBC
  std::string InitState; // "random", "stripe_pi2pi2", "stripe_pi0"
  size_t Lx;
  size_t Ly; // number of zig-zag chains (tilted cylinder circumference)
  double t;
  double t2;   // Inter-chain hopping
  double JK;   //Hund's coupling
  double U;
  std::string MpsPath;
  size_t NumHole;
  size_t Sweeps;
  size_t Dmin;
  std::vector<size_t> Dmax;
  double CutOff;
  double LanczErr;
  size_t MaxLanczIter;
  size_t Threads;
  std::vector<double> noise;

  size_t NumItinerantSites() const { return Lx * Ly; }

  size_t QuarterFilledElectronCount() const { return NumItinerantSites() / 2; }

  size_t NumElectrons() const { return QuarterFilledElectronCount() - NumHole; }

  double Filling() const {
    return NumItinerantSites() == 0
               ? 0.0
               : static_cast<double>(NumElectrons()) / static_cast<double>(NumItinerantSites());
  }

  std::string HoleTag() const {
    return NumHole == 0 ? "" : "Nh" + std::to_string(NumHole);
  }

  std::string ResolvedMpsPath() const {
    if (!MpsPath.empty()) {
      return MpsPath;
    }
    return qlmps::kMpsPath;
  }

  std::string ResolvedTempPath() const {
    if (MpsPath.empty()) {
      return qlmps::kRuntimeTempPath;
    }
    return ResolvedMpsPath() + "_temp";
  }

 private:
  void ValidateFilling() const {
    const size_t quarter_filled_electron_count = QuarterFilledElectronCount();
    if (NumHole > quarter_filled_electron_count) {
      throw std::invalid_argument(
          "NumHole = " + std::to_string(NumHole) +
          " exceeds the quarter-filled itinerant electron count = " +
          std::to_string(quarter_filled_electron_count));
    }
  }
};

#endif //KONDO_TWO_LEG_PARAMS_CASE_H
