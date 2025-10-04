#ifndef KONDO_TWO_LEG_PARAMS_CASE_H
#define KONDO_TWO_LEG_PARAMS_CASE_H
#include "qlmps/qlmps.h"
using qlmps::CaseParamsParserBasic;

struct CaseParams : public CaseParamsParserBasic {
  CaseParams(const char *pf) : CaseParamsParserBasic(pf) {
    Geometry = ParseStr("Geometry");
    Lx = ParseInt("Lx");
    Ly = ParseIntOr("Ly", 2);
    t = ParseDouble("t");
    t2 = ParseDouble("t2");
    JK = ParseDouble("Jk");
    U = ParseDouble("U");
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

  std::string Geometry; // PBC, OBC
  size_t Lx;
  size_t Ly; // number of zig-zag chains (tilted cylinder circumference)
  double t;
  double t2;   // Inter-chain hopping
  double JK;   //Hund's coupling
  double U;
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

#endif //KONDO_TWO_LEG_PARAMS_CASE_H