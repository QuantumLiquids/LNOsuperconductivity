#ifndef TJ_MODEL_H
#define TJ_MODEL_H

#include "qlmps/one_dim_tn/mpo/mpo.h"
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen.h"
#include "../src_single_orbital/tJ_type_hilbert_space.h"
#include "squarelattice.h"
#include "../src_single_orbital/params_case.h"
#include "tJ_operators.h"

struct tJModelParamters {
 public :
  tJModelParamters(const double t,
                   const double J,
                   const double delta
  ) : t(t), J(J), delta(delta) {}

  tJModelParamters(CaseParams &params) :
      tJModelParamters(params.t,
                       params.J,
                       params.delta
      ) {}

  inline void Print(void) {
    using std::cout;
    cout << " ****** tJV Model Parameter List ****** " << "\n";
    cout << "NN hopping t  = " << t << "\n";
    cout << "NN super-exchange J  = " << J << "\n";
    cout << "delta  = " << delta << "\n";
  }

  double t;
  double J;
  double delta;
};

inline void ConstructAnitJMPOGenerator(
    qlmps::MPOGenerator<TenElemT, QNT> &mpo_gen,
    const SquareLattice &lattice,
    const tJModelParamters &model_params
) {
  if (lattice.Ly % 2 != 0) {
    std::cout << "Note well define for Ly : " << lattice.Ly << std::endl;
  }

  tJOperators ops;
  // We assume the tilted lattice.
  for (const Link &link : lattice.nearest_neighbor_links) {
    size_t site1 = std::get<0>(link);
    size_t site2 = std::get<1>(link);
    if (site1 > site2) {
      std::swap(site1, site2);
    }
    int y1 = site1 % lattice.Ly, y2 = site2 % lattice.Ly;
    int sign_of_anisotropy; // +1 or -1
    if (std::abs(y2 - y1) == 1) {
      //not winding
      if (std::min(y1, y2) % 2 == 0) {
        sign_of_anisotropy = +1;
      } else {
        sign_of_anisotropy = -1;
      }
    } else {//winding
      sign_of_anisotropy = -1;
    }
    const double delta = model_params.delta;
    const double t1 = model_params.t * (1 + sign_of_anisotropy * delta),
        J = model_params.J * (1 + sign_of_anisotropy * delta) * (1 + sign_of_anisotropy * delta);

    mpo_gen.AddTerm(-t1, {ops.bupc, ops.bupa}, {site1, site2}, {ops.f});
    mpo_gen.AddTerm(-t1, {ops.bdnc, ops.bdna}, {site1, site2}, {ops.f});
    mpo_gen.AddTerm(-t1, {ops.bupa, ops.bupc}, {site1, site2}, {ops.f});
    mpo_gen.AddTerm(-t1, {ops.bdna, ops.bdnc}, {site1, site2}, {ops.f});
    mpo_gen.AddTerm(J, {ops.sz, ops.sz}, {site1, site2});
    mpo_gen.AddTerm(J / 2, {ops.sp, ops.sm}, {site1, site2});
    mpo_gen.AddTerm(J / 2, {ops.sm, ops.sp}, {site1, site2});
    mpo_gen.AddTerm(-J / 4, {ops.nf, ops.nf}, {site1, site2});
  }

}

#endif //TJ_MODEL_H