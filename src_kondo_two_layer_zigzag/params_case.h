#ifndef KONDO_TWO_LAYER_ZIGZAG_PARAMS_CASE_H
#define KONDO_TWO_LAYER_ZIGZAG_PARAMS_CASE_H
#include "qlmps/qlmps.h"
#include <iostream>
#include <cstdlib>
using qlmps::CaseParamsParserBasic;

struct CaseParams : public CaseParamsParserBasic {
  CaseParams(const char *pf) : CaseParamsParserBasic(pf) {
    Geometry = ParseStr("Geometry");
    Lx = ParseInt("Lx");
    const int ly_raw = ParseIntOr("Ly", 2);
    if (ly_raw <= 0) {
      std::cerr << "Ly must be positive; got " << ly_raw << std::endl;
      exit(1);
    }
    Ly = static_cast<size_t>(ly_raw);
    t = ParseDouble("t");
    t2 = ParseDouble("t2");
    JK = ParseDouble("Jk");
    Jperp = ParseDouble("Jperp");
    U = ParseDouble("U");
    noise = ParseDoubleVec("noise");
    Sweeps = ParseInt("Sweeps");
    Dmin = ParseInt("Dmin");
    Dmax = ParseSizeTVec("Dmax");
    CutOff = ParseDouble("CutOff");
    LanczErr = ParseDouble("LanczErr");
    MaxLanczIter = ParseInt("MaxLanczIter");
    Threads = ParseInt("Threads");
  }

  std::string Geometry; // PBC or OBC (along y)
  size_t Lx;
  size_t Ly; // number of zigzag chains per layer
  double t;  // intra-chain NN hopping
  double t2; // inter-chain diagonal hopping (zigzag)
  double JK; // Kondo/Hund coupling (JK = -J_H, FM <=> JK < 0)
  double Jperp; // interlayer AFM exchange between localized spins
  double U;
  size_t Sweeps;
  size_t Dmin;
  std::vector<size_t> Dmax;
  double CutOff;
  double LanczErr;
  size_t MaxLanczIter;
  size_t Threads;
  std::vector<double> noise;
};

#endif // KONDO_TWO_LAYER_ZIGZAG_PARAMS_CASE_H
