#pragma once
#include "qlmps/qlmps.h"
using qlmps::CaseParamsParserBasic;

struct CaseParams : public CaseParamsParserBasic {
  CaseParams(const char *pf) : CaseParamsParserBasic(pf) {
    //symmetry_mode: 1 for only spin U1, 2 for spin cross particle U1
    Geometry = ParseStr("Geometry");
    Lx = ParseInt("Lx");
    Ly = ParseInt("Ly");
    t1 = ParseDouble("t1");
    t2 = ParseDouble("t2");
    Jh = ParseDouble("Jh");
    U = ParseDouble("U");
    delta = ParseDouble("delta");
    noise = ParseDoubleVec("noise");
    if (Has("NumEle1") || Has("NumEle2")) {
      std::cerr << "Use 'NumElectronsDx2Y2' and 'NumElectronsDz2'. "
                << "Legacy keys 'NumEle1'/'NumEle2' are no longer supported." << std::endl;
      exit(1);
    }
    NumElectronsDx2Y2 = ParseIntOr("NumElectronsDx2Y2", static_cast<int>(Lx * Ly));
    NumElectronsDz2 = ParseIntOr("NumElectronsDz2", static_cast<int>(2 * Lx * Ly));
    if (Has("Perturbation") || Has("PerturbationAmplitude") || Has("PerturbationPeriod")) {
      std::cerr << "Use 'InterOrbitalHybridization' and 'Dx2Y2InterlayerHopping'. "
                << "Legacy keys 'Perturbation*' are no longer supported." << std::endl;
      exit(1);
    }
    InterOrbitalHybridization = ParseDoubleOr("InterOrbitalHybridization", 0.0);
    Dx2Y2InterlayerHopping = ParseDoubleOr("Dx2Y2InterlayerHopping", 0.0);
    mu1 = ParseDouble("mu1");
    mu2 = ParseDouble("mu2");
    Sweeps = ParseInt("Sweeps");
    Dmin = ParseInt("Dmin");
    Dmax = ParseInt("Dmax");
    CutOff = ParseDouble("CutOff");
    LanczErr = ParseDouble("LanczErr");
    MaxLanczIter = ParseInt("MaxLanczIter");
    TotalThreads = ParseInt("TotalThreads");
    PinningField = ParseBool("PinningField");
  }

  std::string Geometry; // Cylinder, Torus, OBC, Rotated, Ladder
  size_t Lx;
  size_t Ly;
  double t1;   // t_parallel
  double t2;   // t_perp
  double Jh;   //Hund's coupling
  double U;
  double delta;
  double mu1;  //chemical potential
  double mu2;  //chemical potential
  size_t Sweeps;
  size_t Dmin;
  size_t Dmax;
  size_t NumElectronsDx2Y2;
  size_t NumElectronsDz2;
  double CutOff;
  double LanczErr;
  size_t MaxLanczIter;
  size_t TotalThreads;
  std::vector<double> noise;
  bool PinningField;
  double InterOrbitalHybridization;
  double Dx2Y2InterlayerHopping;
};
