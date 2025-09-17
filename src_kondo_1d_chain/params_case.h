#ifndef KONDO_PARAMS_CASE_H
#define KONDO_PARAMS_CASE_H
#include "qlmps/qlmps.h"
using qlmps::CaseParamsParserBasic;

struct CaseParams : public CaseParamsParserBasic {
  CaseParams(const char *pf) : CaseParamsParserBasic(pf) {
    L = ParseInt("L");
    t = ParseDouble("t");
    JK = ParseDouble("Jk");
    U = ParseDouble("U");
    // Optional pinning field strength on middle itinerant site; default 0 for backward compatibility
    mu = ParseDoubleOr("mu", 0.0);
    noise = ParseDoubleVec("noise");
//    NumEle = ParseInt("NumEle");
    Sweeps = ParseInt("Sweeps");
    Dmin = ParseInt("Dmin");
    Dmax = ParseSizeTVec("Dmax");
    CutOff = ParseDouble("CutOff");
    LanczErr = ParseDouble("LanczErr");
    MaxLanczIter = ParseInt("MaxLanczIter");
    Threads = ParseInt("Threads");
  }

  size_t L;
  double t;
  double JK;   //Hund's coupling
  double U;
  double mu;   // optional pinning field strength (defaults to 0), simulate defect
  size_t Sweeps;
  size_t Dmin;
  std::vector<size_t> Dmax;
//  size_t NumEle; //quarter filling
  double CutOff;
  double LanczErr;
  size_t MaxLanczIter;
  size_t Threads;
  std::vector<double> noise;
};

#endif