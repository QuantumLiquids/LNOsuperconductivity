#ifndef DOUBLE_LAYER_TJ_MODEL_H
#define DOUBLE_LAYER_TJ_MODEL_H

#include "qlmps/one_dim_tn/mpo/mpo.h"
#include "qlmps/one_dim_tn/mpo/mpogen/mpogen.h"
#include "tJ_type_hilbert_space.h"
#include "double_layer_squarelattice.h"
#include "params_case.h"

inline void ConstructDoubleLayertJMPOGenerator(
    qlmps::MPOGenerator<TenElemT, QNT> &mpo_gen,
    const DoubleLayerSquareLattice &lattice,
    const DoubleLayertJModelParamters &model_params
) {
  const double t1 = model_params.t,
      t2 = model_params.t_perp,
      J = model_params.J,
      J2 = model_params.J_perp,
      phi = model_params.phi,
      delta = model_params.delta;
  qlmps::tJOperators<TenElemT, QNT> ops;
  if (std::fabs(phi) < 1e-15) {
    if (std::fabs(delta) < 1e-15) {
      for (const Link &link : lattice.intralayer_links) {
        size_t site1 = std::get<0>(link);
        size_t site2 = std::get<1>(link);
        assert(site1 < site2);
        qlmps::AddTJHoppingTerms(mpo_gen, TenElemT(t1), site1, site2, ops);
        qlmps::AddHeisenbergCoupling(mpo_gen, J, site1, site2, ops);
//      mpo_gen.AddTerm(-J / 4, {nf, nf}, {site1, site2});
#ifndef NDEBUG
        std::cout << "Add  -t = " << -t1 << ", J = " << J << "between sites" << site1 << "," << site2 << std::endl;
#endif
      }
    } else {
      std::cout << "Note the current code only work for OBC!" << std::endl;
      for (const Link &link : lattice.intralayer_links) {
        size_t site1 = std::get<0>(link);
        size_t site2 = std::get<1>(link);
        assert(site1 < site2);
        size_t Ly = lattice.Ly;
        const size_t y = (site1 % (2 * Ly)) / 2;
        const size_t x = site1 / (2 * Ly); //x coordinate of site i
        double t_eff, J_eff;
        if ((x + y) % 2 == 0) {
          t_eff = t1 * (1 + delta);
          J_eff = J * (1 + delta) * (1 + delta);
        } else {
          t_eff = t1 * (1 - delta);
          J_eff = J * (1 - delta) * (1 - delta);
        }
        qlmps::AddTJHoppingTerms(mpo_gen, TenElemT(t_eff), site1, site2, ops);
        qlmps::AddHeisenbergCoupling(mpo_gen, J_eff, site1, site2, ops);
//      mpo_gen.AddTerm(-J_eff / 4, {nf, nf}, {site1, site2});
#ifndef NDEBUG
        std::cout << "Add  -t = " << -t_eff << ", J = " << J_eff << "between sites" << site1 << "," << site2
                  << std::endl;
#endif
      }
    }

    for (const Link &link : lattice.interlayer_links) {
      size_t site1 = std::get<0>(link);
      size_t site2 = std::get<1>(link);
      assert(site1 < site2);
      qlmps::AddTJHoppingTerms(mpo_gen, TenElemT(t2), site1, site2, ops);
      qlmps::AddHeisenbergCoupling(mpo_gen, J2, site1, site2, ops);
//      mpo_gen.AddTerm(-J2 / 4, {nf, nf}, {site1, site2});
#ifndef NDEBUG
      std::cout << "Add  -t_perp = " << -t2 << ", J_perp = " << J2 << "between sites" << site1 << "," << site2
                << std::endl;
#endif
    }
    if (model_params.pinning_field) {
      mpo_gen.AddTerm(1.0, ops.sz, 0);
    }
  } else {
    std::cout << "not support" << std::endl;
    exit(1);
  }
}

#endif //DOUBLE_LAYER_TJ_MODEL_H