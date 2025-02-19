/*
 * File Name: params_case.h
 * Description: Declare CaseParams class used set parameters by users
 * Created by Hao-Xin on 2023/04/03.
 *
 */


#ifndef TJMODEL_SRC_PARAMS_CASE_H
#define TJMODEL_SRC_PARAMS_CASE_H

#include "qlmps/case_params_parser.h"

using qlmps::CaseParamsParserBasic;

struct CaseParams : public CaseParamsParserBasic {
  CaseParams(const char *pf) : CaseParamsParserBasic(pf) {
    Geometry = ParseStr("Geometry");
    Ly = ParseInt("Ly");
    Lx = ParseInt("Lx");
    Numhole = ParseInt("NumHole");
    t = ParseDouble("t");
    t_perp = ParseDouble("t_perp");
    J = ParseDouble("J");
    J_perp = ParseDouble("J_perp");
//    phi = ParseDouble("phi");
    Sweeps = ParseInt("Sweeps");
    Dmin = ParseInt("Dmin");
    Dmax = ParseInt("Dmax");
    CutOff = ParseDouble("CutOff");
    LanczErr = ParseDouble("LanczErr");
    MaxLanczIter = ParseInt("MaxLanczIter");
    Threads = ParseInt("Threads");
    Perturbation = ParseDouble("Perturbation");
    wavelength = ParseDouble("wavelength");
    noise = ParseDoubleVec("noise");
  }

  std::string Geometry; // Cylinder, Torus, OBC, Rotated, Ladder (useless actually)
  size_t Ly;
  size_t Lx;
  int Numhole;     //Numhole = 0 means quarter filling. Negative means electron dope
  double t;
  double t_perp;
  double J;
  double J_perp;
  double phi = 0;     // twist boundary condition, Phys. Rev. B 107, 075127 Eq.(5)
  size_t Sweeps;
  size_t Dmin;
  size_t Dmax;
  double CutOff;
  double LanczErr;
  size_t MaxLanczIter;
  size_t Threads;
  double Perturbation;
  double wavelength;
  std::vector<double> noise;

};

struct DoubleLayertJModelParamters {
 public :
  DoubleLayertJModelParamters(const double t,
                              const double t_perp,
                              const double J,
                              const double J_perp,
                              const double phi
  ) : t(t), t_perp(t_perp), J(J), J_perp(J_perp), phi(phi) {}

  DoubleLayertJModelParamters(CaseParams &params) :
      DoubleLayertJModelParamters(params.t,
                                  params.t_perp,
                                  params.J,
                                  params.J_perp,
                                  params.phi
      ) {}

  inline void Print(void) {
    using std::cout;
    cout << " ****** tJV Model Parameter List ****** " << "\n";
    cout << "intralayer NN hopping t  = " << t << "\n";
    cout << "interlayer hopping t_perp = " << t_perp << "\n";
    cout << "intralayer NN super-exchange J  = " << J << "\n";
    cout << "interlayer super-exchange J_perp = " << J_perp << "\n";
    cout << "Twist angle phi = " << phi << "\n";
  }

  double t;
  double t_perp;
  double J;
  double J_perp;
  double phi = 0;
};

#endif //TJMODEL_SRC_PARAMS_CASE_H
