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
    NumEle1 = ParseInt("NumEle1");
    NumEle2 = ParseInt("NumEle2");
    mu1 = ParseDouble("mu1");
    mu2 = ParseDouble("mu2");
    Sweeps = ParseInt("Sweeps");
    Dmin = ParseInt("Dmin");
    Dmax = ParseInt("Dmax");
    CutOff = ParseDouble("CutOff");
    LanczErr = ParseDouble("LanczErr");
    MaxLanczIter = ParseInt("MaxLanczIter");
    TotalThreads = ParseInt("TotalThreads");
    Perturbation = ParseBool("Perturbation");
    if (Perturbation) {
      PA = ParseDouble("PerturbationAmplitude");
      PerturbationPeriod = ParseInt("PerturbationPeriod");
    } else {
      PA = 0.0;
      PerturbationPeriod = 1;
    }
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
  size_t NumEle1;
  size_t NumEle2;
  double CutOff;
  double LanczErr;
  size_t MaxLanczIter;
  size_t TotalThreads;
  std::vector<double> noise;
  bool Perturbation;
  double PA;
  size_t PerturbationPeriod;
};
