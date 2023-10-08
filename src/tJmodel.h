#ifndef TJ_MODEL_H
#define TJ_MODEL_H

#include "gqmps2/one_dim_tn/mpo/mpo.h"
#include "gqmps2/one_dim_tn/mpo/mpogen/mpogen.h"
#include "gqdouble.h"
#include "squarelattice.h"
#include "params_case.h"
#include "operators.h"

inline void ConstructDoubleLayertJMPOGenerator(
    gqmps2::MPOGenerator<TenElemT, U1U1QN> &mpo_gen,
    const DoubleLayerSquareLattice &lattice,
    const DoubleLayertJModelParamters &model_params
) {
  const double t1 = model_params.t,
      t2 = model_params.t_perp,
      J = model_params.J,
      J2 = model_params.J_perp,
      phi = model_params.phi;
  OperatorInitial();
  if (std::fabs(phi) < 1e-15) {
    for (const Link &link: lattice.intralayer_links) {
      size_t site1 = std::get<0>(link);
      size_t site2 = std::get<1>(link);
      assert(site1 < site2);
      mpo_gen.AddTerm(-t1, {bupc, bupa}, {site1, site2}, {f});
      mpo_gen.AddTerm(-t1, {bdnc, bdna}, {site1, site2}, {f});
      mpo_gen.AddTerm(-t1, {bupa, bupc}, {site1, site2}, {f});
      mpo_gen.AddTerm(-t1, {bdna, bdnc}, {site1, site2}, {f});
      mpo_gen.AddTerm(J, {sz, sz}, {site1, site2});
      mpo_gen.AddTerm(J / 2, {sp, sm}, {site1, site2});
      mpo_gen.AddTerm(J / 2, {sm, sp}, {site1, site2});
      mpo_gen.AddTerm(-J / 4, {nf, nf}, {site1, site2});
#ifndef NDEBUG
      std::cout << "Add  -t = " << -t1 << ", J = " << J << "between sites" << site1 <<","<<site2<<std::endl;
#endif
    }

    for (const Link &link: lattice.interlayer_links) {
      size_t site1 = std::get<0>(link);
      size_t site2 = std::get<1>(link);
      assert(site1 < site2);
      mpo_gen.AddTerm(-t2, {bupc, bupa}, {site1, site2}, {f});
      mpo_gen.AddTerm(-t2, {bdnc, bdna}, {site1, site2}, {f});
      mpo_gen.AddTerm(-t2, {bupa, bupc}, {site1, site2}, {f});
      mpo_gen.AddTerm(-t2, {bdna, bdnc}, {site1, site2}, {f});
      mpo_gen.AddTerm(J2, {sz, sz}, {site1, site2});
      mpo_gen.AddTerm(J2 / 2, {sp, sm}, {site1, site2});
      mpo_gen.AddTerm(J2 / 2, {sm, sp}, {site1, site2});
      mpo_gen.AddTerm(-J2 / 4, {nf, nf}, {site1, site2});
      std::cout << "Add  -t_perp = " << -t2 << ", J_perp = " << J2 << "between sites" << site1 << "," << site2 << std::endl;
    }
  } else {
    std::cout << "not support" << std::endl;
    exit(1);
  }
}

#endif //TJ_MODEL_H