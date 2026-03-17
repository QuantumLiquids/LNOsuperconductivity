/**
 * DMRG for LNO two-layer, two-orbital model code
 * Upto July 3rd, 2025, the code only support for Ly = 2
 *
 */
#include "qlmps/qlmps.h"
#include <fstream>
#include <stdexcept>
#include <vector>

#include "./hilbert_space.h"
#include "./myutil.h"
#include "./params_case.h"

/**
 *  Ly = 2
 *  O : d_x^-y^2, o: d_z^2
 *           ------------> x
 *  Layer1   O--o--O--o--O--o--O--o  (y = 0)
 *           |  |  |  |  |  |  |  |
 *  Layer1   O--o--O--o--O--o--O--o  (y = 1)
 *           |  |  |  |  |  |  |  |
 *  Layer2   O--o--O--o--O--o--O--o  (y = 0)
 *           |  |  |  |  |  |  |  |
 *  Layer2   O--o--O--o--O--o--O--o  (y = 1)
 */

using FiniteMPST = qlmps::FiniteMPS<TenElemT, QNT>;

namespace {

std::vector<size_t> BuildMaxLanczIterSchedule(size_t stage_count, size_t max_lancz_iter) {
  if (stage_count == 0) {
    return {};
  }
  std::vector<size_t> schedule(stage_count, max_lancz_iter);
  if (stage_count == 1) {
    return schedule;
  }

  const size_t first_stage_iter = std::min<size_t>(3, max_lancz_iter);
  schedule.front() = first_stage_iter;
  schedule.back() = max_lancz_iter;
  const size_t stage_step =
      (max_lancz_iter > first_stage_iter) ? (max_lancz_iter - first_stage_iter) / (stage_count - 1) : 0;
  for (size_t i = 1; i + 1 < stage_count; ++i) {
    schedule[i] = schedule[i - 1] + stage_step;
  }
  return schedule;
}

template <typename T>
std::string SerializeJsonArray(const std::vector<T> &values) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

void WriteStageMetadata(const CaseParams &params,
                        const std::vector<size_t> &dmax_schedule,
                        size_t stage_index,
                        size_t stage_dmax,
                        size_t stage_max_lancz_iter,
                        const std::string &active_mps_path,
                        const std::string &backup_mps_path) {
  const std::string stage_tag = params.BuildStageTag(stage_index, stage_dmax);
  const std::string metadata_path = "dmrg_stage_metadata_" + stage_tag + ".json";
  std::ofstream ofs(metadata_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open metadata file: " + metadata_path);
  }

  ofs << "{\n"
      << "  \"StageTag\": \"" << stage_tag << "\",\n"
      << "  \"StageNumber\": " << (stage_index + 1) << ",\n"
      << "  \"StageCount\": " << dmax_schedule.size() << ",\n"
      << "  \"Geometry\": \"" << params.Geometry << "\",\n"
      << "  \"Lx\": " << params.Lx << ",\n"
      << "  \"Ly\": " << params.Ly << ",\n"
      << "  \"t1\": " << params.t1 << ",\n"
      << "  \"t2\": " << params.t2 << ",\n"
      << "  \"Jh\": " << params.Jh << ",\n"
      << "  \"U\": " << params.U << ",\n"
      << "  \"delta\": " << params.delta << ",\n"
      << "  \"mu1\": " << params.mu1 << ",\n"
      << "  \"mu2\": " << params.mu2 << ",\n"
      << "  \"NumElectronsDx2Y2\": " << params.NumElectronsDx2Y2 << ",\n"
      << "  \"NumElectronsDz2\": " << params.NumElectronsDz2 << ",\n"
      << "  \"InterOrbitalHybridization\": " << params.InterOrbitalHybridization << ",\n"
      << "  \"Dx2Y2InterlayerHopping\": " << params.Dx2Y2InterlayerHopping << ",\n"
      << "  \"PinningField\": " << (params.PinningField ? "true" : "false") << ",\n"
      << "  \"Sweeps\": " << params.Sweeps << ",\n"
      << "  \"RequestedDmin\": " << params.Dmin << ",\n"
      << "  \"EffectiveDmin\": " << params.EffectiveDmin(stage_dmax) << ",\n"
      << "  \"Dmax\": " << stage_dmax << ",\n"
      << "  \"DmaxSchedule\": " << SerializeJsonArray(dmax_schedule) << ",\n"
      << "  \"CutOff\": " << params.CutOff << ",\n"
      << "  \"LanczErr\": " << params.LanczErr << ",\n"
      << "  \"RequestedMaxLanczIter\": " << params.MaxLanczIter << ",\n"
      << "  \"StageMaxLanczIter\": " << stage_max_lancz_iter << ",\n"
      << "  \"noise\": " << SerializeJsonArray(params.noise) << ",\n"
      << "  \"ActiveMpsPath\": \"" << active_mps_path << "\",\n"
      << "  \"BackupMpsPath\": \"" << backup_mps_path << "\"\n"
      << "}\n";
}

}  // namespace

int main(int argc, char *argv[]) {
  using namespace qlmps;
  using namespace qlten;
  MPI_Init(nullptr, nullptr);
  MPI_Comm comm = MPI_COMM_WORLD;
  int rank, mpi_size;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &rank);

  if (argc == 1) {
    if (rank == 0) {
      std::cout
          << "Usage: \n mpirun -np <num_proc> ./dmrg <params file> [--D=<list of bond dimensions, connected by comma>]\n";
    }
    return 0;
  } else if (argc == 2) {
    if (rank == 0)
      std::cout
          << "The complete usage can be: \n mpirun -np <num_proc> ./dmrg <params file> [--D=<list of bond dimensions, connected by comma>]"
          << std::endl;
  }

  CaseParams params(argv[1]);

#ifndef USE_GPU
  qlten::hp_numeric::SetTensorManipulationThreads(params.TotalThreads);
#endif

  /******** Model parameter ********/
  size_t Lx = params.Lx, Ly = params.Ly;
  if (Lx < Ly) {
    std::swap(Lx, Ly);
    std::swap(params.Lx, params.Ly);
    std::cout << "Swap Lx and Ly" << std::endl;
  }
  if (Ly != 2) {
    std::cout << "Do not support Ly : " << Ly << std::endl;
    exit(1);
  }
  size_t N = 4 * Lx * Ly;//two orbital, two layer
  double t1 = params.t1, t2 = params.t2, J_H = params.Jh;
  double U = params.U;
  if (rank == 0) {
    std::cout << "System size = (" << Lx << "," << Ly << ")" << std::endl;
    std::cout << "The number of electron sites =" << N << std::endl;
    std::cout << "Model parameter: t1 :" << t1 << ", t2 :" << t2
              << ", U : " << U
              << ", J_H : " << J_H
              << std::endl;
  }

  clock_t startTime, endTime;
  startTime = clock();

  qlmps::HubbardOperators<TenElemT, QNT> ops;
  const SiteVec<TenElemT, QNT> sites = SiteVec<TenElemT, QNT>(N, pb_out);
  const std::string active_mps_path = qlmps::kMpsPath;

  std::vector<size_t> input_D_set;
  bool has_bond_dimension_parameter = ParserBondDimension(
      argc, argv,
      input_D_set);
  const std::vector<size_t> dmax_schedule = has_bond_dimension_parameter ? input_D_set : params.Dmax;
  const size_t dmrg_stage_count = dmax_schedule.size();
  const std::vector<size_t> max_lancz_iter_schedule =
      BuildMaxLanczIterSchedule(dmrg_stage_count, params.MaxLanczIter);
  if (rank == 0) {
    if (has_bond_dimension_parameter) {
      std::cout << "Override Dmax schedule from --D argument." << std::endl;
    }
    std::cout << "Dmax schedule = " << SerializeJsonArray(dmax_schedule) << std::endl;
    std::cout << "Stage MaxLanczIter = " << SerializeJsonArray(max_lancz_iter_schedule) << std::endl;
  }

  /****** Initialize MPS ******/
  FiniteMPST mps(sites);
  if (rank == 0) {
    if (!IsPathExist(kMpsPath) || !(N == GetNumofMps())) {
      std::cout << "Initial mps as direct product state." << std::endl;
      try {
        const auto stat_labs2 = BuildInitialDz2StateLabels(N / 2, params.NumElectronsDz2, Ly);
        const auto stat_labs1 = BuildInitialDx2Y2StateLabels(N / 2, params.NumElectronsDx2Y2, stat_labs2);
        const auto stat_labs = InterleaveOrbitalStateLabels(stat_labs1, stat_labs2, Ly);
        std::cout << "Initial filling: "
                  << "N_dx2y2 = " << params.NumElectronsDx2Y2
                  << ", N_dz2 = " << params.NumElectronsDz2 << std::endl;
        qlmps::DirectStateInitMps(mps, stat_labs);
        mps.Dump(active_mps_path, true);
      } catch (const std::exception &ex) {
        std::cout << "Failed to build initial state: " << ex.what() << std::endl;
        exit(1);
      }
    }
  }


  /*******  Creation MPO/MRO *******/
  qlmps::MPOGenerator<TenElemT, QNT> mpo_gen(sites, qn0);

  //Hund's coupling
  for (size_t x = 0; x < 2 * Lx; x += 2) {
    for (size_t y = 0; y < 2 * Ly; y++) {
      size_t site1 = x * (2 * Ly) + y;
      size_t site2 = (x + 1) * (2 * Ly) + y;
      mpo_gen.AddTerm(-2 * J_H, ops.sz, site1, ops.sz, site2);
      mpo_gen.AddTerm(-J_H, ops.sp, site1, ops.sm, site2);
      mpo_gen.AddTerm(-J_H, ops.sm, site1, ops.sp, site2);
    }
  }
  //t_perp hopping in d_z^2 orbital
  for (size_t x = 1; x < 2 * Lx; x += 2) {
    for (size_t y = 0; y < Ly; y++) {
      size_t site1 = x * (2 * Ly) + y;
      size_t site2 = site1 + Ly;
      mpo_gen.AddTerm(-t2, ops.bupcF, site1, ops.bupa, site2, ops.f);
      mpo_gen.AddTerm(t2, ops.bupaF, site1, ops.bupc, site2, ops.f);
      mpo_gen.AddTerm(-t2, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
      mpo_gen.AddTerm(t2, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
    }
  }

  //t_para Horizontal hopping in d_x^2-y^2 orbital
  for (size_t x = 0; x < 2 * Lx - 2; x += 2) {
    for (size_t y = 0; y < (2 * Ly); y++) {
      size_t y_phy = y % Ly; // physical value of y
      size_t x_phy = x / 2;// physical value of 2
      double t_eff;
      if ((x_phy + y_phy) % 2 == 0) {
        t_eff = t1 * (1 + params.delta);
      } else {
        t_eff = t1 * (1 - params.delta);
      }
      size_t site1 = x * (2 * Ly) + y;
      size_t site2 = site1 + (4 * Ly);
      mpo_gen.AddTerm(-t_eff, ops.bupcF, site1, ops.bupa, site2, ops.f);
      mpo_gen.AddTerm(t_eff, ops.bupaF, site1, ops.bupc, site2, ops.f);
      mpo_gen.AddTerm(-t_eff, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
      mpo_gen.AddTerm(t_eff, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
    }
  }

  //t_para Vertical hopping in d_x^2-y^2 orbital
  for (size_t x = 0; x < 2 * Lx; x += 2) {
    for (size_t y = 0; y < (2 * Ly); y++) {
      if (y % Ly < Ly - 1) { // OBC
        size_t y_phy = y % Ly; // physical value of y
        size_t x_phy = x / 2;  // physical value of 2
        double t_eff;
        if ((x_phy + y_phy) % 2 == 0) {
          t_eff = t1 * (1 + params.delta);
        } else {
          t_eff = t1 * (1 - params.delta);
        }
        size_t site1 = x * (2 * Ly) + y;
        size_t site2 = site1 + 1;
        mpo_gen.AddTerm(-t_eff, ops.bupcF, site1, ops.bupa, site2, ops.f);
        mpo_gen.AddTerm(t_eff, ops.bupaF, site1, ops.bupc, site2, ops.f);
        mpo_gen.AddTerm(-t_eff, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
        mpo_gen.AddTerm(t_eff, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
      } else if (Ly > 2) { // y% Ly == Ly-1
        //PBC winding code here

      }
    }
  }
  // optional on-site hybridization between d_z^2 and d_x^2-y^2 orbitals
  for (const auto &[site1, site2] : CollectOnsiteInterOrbitalBonds(Lx, Ly)) {
    const TenElemT t_noise = params.InterOrbitalHybridization;
    if (t_noise != 0.0) {
      mpo_gen.AddTerm(-t_noise, ops.bupcF, site1, ops.bupa, site2, ops.f);
      mpo_gen.AddTerm(t_noise, ops.bupaF, site1, ops.bupc, site2, ops.f);
      mpo_gen.AddTerm(-t_noise, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
      mpo_gen.AddTerm(t_noise, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
    }
  }
  // optional extra interlayer hopping in d_x^2-y^2 orbital
  for (const auto &[site1, site2] : CollectInterlayerDx2Y2Bonds(Lx, Ly)) {
    const TenElemT t_noise = params.Dx2Y2InterlayerHopping;
    if (t_noise != 0.0) {
      mpo_gen.AddTerm(-t_noise, ops.bupcF, site1, ops.bupa, site2, ops.f);
      mpo_gen.AddTerm(t_noise, ops.bupaF, site1, ops.bupc, site2, ops.f);
      mpo_gen.AddTerm(-t_noise, ops.bdnc, site1, ops.Fbdna, site2, ops.f);
      mpo_gen.AddTerm(t_noise, ops.bdna, site1, ops.Fbdnc, site2, ops.f);
    }
  }

  for (size_t i = 0; i < N; i++) {
    mpo_gen.AddTerm(U, ops.Uterm, i);
    if ((i / (2 * Ly)) % 2 == 0) {
      mpo_gen.AddTerm(params.mu1, ops.nf, i); // d_x^2-y^2 orbital
    } else {
      mpo_gen.AddTerm(params.mu2, ops.nf, i); // d_z^2 orbital
    }
  }

  if (params.PinningField) {
    mpo_gen.AddTerm(1.0, ops.sz, 2 * Ly); // d_z^2 orbital
  }

  auto mro = mpo_gen.GenMatReprMPO(true);
  if (rank == 0)
    std::cout << "MRO generated." << std::endl;

  auto run_measurements = [&](const std::string &stage_tag, const std::string &mps_path) {
    const std::string file_postfix = "dmrg_two_layer_two_orbital_" + stage_tag;
    const std::vector<size_t> dx2y2_sites = CollectOrbitalSites(N, Ly, false);
    const std::vector<size_t> dz2_sites = CollectOrbitalSites(N, Ly, true);

    const std::vector<QLTensor<TenElemT, QNT>> one_site_ops = {ops.sz, ops.nf, ops.nupndn};

    if (rank == hp_numeric::kMPIMasterRank) {
      Timer one_site_timer("measure one site operators");
      for (bool dz2_orbital : {false, true}) {
        const auto &measurement_sites = dz2_orbital ? dz2_sites : dx2y2_sites;
        const std::string orbital_tag = OrbitalTag(dz2_orbital);
        const std::vector<std::string> one_site_labels = {
            std::string("sz_") + orbital_tag + "_" + file_postfix,
            std::string("nf_") + orbital_tag + "_" + file_postfix,
            std::string("nupndn_") + orbital_tag + "_" + file_postfix};
        try {
          const auto one_site_measu_res_set =
              MeasureOneSiteOp(mps, mps_path, one_site_ops, measurement_sites, one_site_labels);
          if (one_site_measu_res_set.size() != one_site_ops.size()) {
            throw std::runtime_error("Unexpected number of one-site measurement outputs.");
          }
          DumpMeasuRes(DeriveSingleOccupancyMeasuRes(one_site_measu_res_set[1], one_site_measu_res_set[2]),
                       std::string("single_occ_") + orbital_tag + "_" + file_postfix);
          DumpMeasuRes(DeriveChargeVarianceMeasuRes(one_site_measu_res_set[1], one_site_measu_res_set[2]),
                       std::string("charge_var_") + orbital_tag + "_" + file_postfix);
        } catch (const std::exception &ex) {
          std::cerr << "Failed to derive one-site observables for " << orbital_tag
                    << ": " << ex.what() << std::endl;
          MPI_Abort(comm, 1);
        }
      }
      one_site_timer.PrintElapsed();
      std::cout << "measured one point function.<====" << std::endl;
    }

    struct TwoSiteTask {
      std::string label;
      const QLTensor<TenElemT, QNT> &op1;
      const QLTensor<TenElemT, QNT> &op2;
    };

    const std::vector<TwoSiteTask> two_site_tasks = {
        {"szsz", ops.sz, ops.sz},
        {"spsm", ops.sp, ops.sm},
        {"smsp", ops.sm, ops.sp},
        {"nfnf", ops.nf, ops.nf},
        {"nupndn_nupndn", ops.nupndn, ops.nupndn}
    };

    for (bool dz2_orbital : {false, true}) {
      const auto &orbital_sites = dz2_orbital ? dz2_sites : dx2y2_sites;
      if (orbital_sites.size() < 2) {
        continue;
      }
      const std::string orbital_tag = OrbitalTag(dz2_orbital);
      const size_t ref_site = orbital_sites[orbital_sites.size() / 2];
      std::vector<size_t> target_sites;
      for (size_t site : orbital_sites) {
        if (site > ref_site) {
          target_sites.push_back(site);
        }
      }
      for (size_t idx = 0; idx < two_site_tasks.size(); ++idx) {
        if (idx % mpi_size == rank) {
          const auto &task = two_site_tasks[idx];
          auto measu_res = MeasureTwoSiteOpGroup(mps,
                                                 mps_path,
                                                 task.op1,
                                                 task.op2,
                                                 ref_site,
                                                 target_sites);
          DumpMeasuRes(measu_res,
                       task.label + "_" + orbital_tag + "_ref" + std::to_string(ref_site) + "_" + file_postfix);
        }
      }
    }
    MPI_Barrier(comm);
  };

  // dmrg
  for (size_t stage_index = 0; stage_index < dmrg_stage_count; ++stage_index) {
    const size_t stage_dmax = dmax_schedule[stage_index];
    const size_t stage_dmin = params.EffectiveDmin(stage_dmax);
    if (rank == 0) {
      std::cout << "Stage " << (stage_index + 1) << "/" << dmrg_stage_count
                << ": Dmin = " << stage_dmin
                << ", Dmax = " << stage_dmax
                << ", MaxLanczIter = " << max_lancz_iter_schedule[stage_index]
                << std::endl;
      if (stage_dmin != params.Dmin) {
        std::cout << "Clamp Dmin from " << params.Dmin << " to " << stage_dmin
                  << " for this stage." << std::endl;
      }
    }

    qlmps::FiniteVMPSSweepParams stage_sweep_params(
        params.Sweeps,
        stage_dmin, stage_dmax, params.CutOff,
        qlmps::LanczosParams(params.LanczErr, max_lancz_iter_schedule[stage_index]),
        params.noise
    );
    stage_sweep_params.mps_path = active_mps_path;

    const std::string stage_tag = params.BuildStageTag(stage_index, stage_dmax);
    (void)qlmps::FiniteDMRG(mps, mro, stage_sweep_params, comm);

    const std::string backup_mps_path = BuildStageBackupPath(stage_sweep_params.mps_path, stage_tag);
    if (rank == 0) {
      try {
        CopyDirectoryRecursively(stage_sweep_params.mps_path, backup_mps_path);
        WriteStageMetadata(params,
                           dmax_schedule,
                           stage_index,
                           stage_dmax,
                           max_lancz_iter_schedule[stage_index],
                           stage_sweep_params.mps_path,
                           backup_mps_path);
      } catch (const std::exception &ex) {
        std::cerr << "Failed to backup stage MPS or write metadata: " << ex.what() << std::endl;
        MPI_Abort(comm, 1);
      }
    }
    MPI_Barrier(comm);
    run_measurements(stage_tag, stage_sweep_params.mps_path);
  }
  endTime = clock();
  std::cout << "CPU Time : " << (double) (endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
  MPI_Finalize();
  return 0;
}
