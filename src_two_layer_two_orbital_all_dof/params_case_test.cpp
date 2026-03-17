#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "params_case.h"

namespace {

std::string WriteTempParamsFile(const std::string &json_text, const std::string &suffix) {
  std::string path = "/tmp/lno_4band_params_" + suffix + ".json";
  std::ofstream ofs(path);
  ofs << json_text;
  ofs.close();
  return path;
}

const char kSharedFields[] = R"json(
    "Geometry": "OBC",
    "Lx": 6,
    "Ly": 2,
    "t1": 1.0,
    "t2": 0.6,
    "Jh": 4.0,
    "U": 12.0,
    "delta": 0.0,
    "mu1": 0.0,
    "mu2": -0.5,
    "Sweeps": 2,
    "Dmin": 10,
    "Dmax": 20,
    "CutOff": 1e-8,
    "LanczErr": 1e-10,
    "MaxLanczIter": 4,
    "TotalThreads": 1,
    "PinningField": false
)json";

bool CheckElectronCounts(const CaseParams &params, size_t expected_dx2y2, size_t expected_dz2) {
  return params.NumElectronsDx2Y2 == expected_dx2y2 &&
         params.NumElectronsDz2 == expected_dz2;
}

bool CheckExtraHoppings(const CaseParams &params,
                        double expected_inter_orbital_hybridization,
                        double expected_dx2y2_interlayer_hopping) {
  return params.InterOrbitalHybridization == expected_inter_orbital_hybridization &&
         params.Dx2Y2InterlayerHopping == expected_dx2y2_interlayer_hopping;
}

bool CheckNoise(const CaseParams &params, const std::vector<double> &expected_noise) {
  return params.noise == expected_noise;
}

bool ContainsAll(const std::string &text, const std::vector<std::string> &expected_fragments) {
  for (const auto &fragment : expected_fragments) {
    if (text.find(fragment) == std::string::npos) {
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  const std::string explicit_json =
      std::string("{\"CaseParams\": {") + kSharedFields +
      R"json(,
    "NumElectronsDx2Y2": 12,
    "NumElectronsDz2": 24,
    "InterOrbitalHybridization": 0.15,
    "Dx2Y2InterlayerHopping": 0.05,
    "noise": [0.2, 0.1]
  }})json";
  const std::string default_json =
      std::string("{\"CaseParams\": {") + kSharedFields +
      R"json(
  }})json";
  const std::string staged_json =
      std::string("{\"CaseParams\": {") + kSharedFields +
      R"json(,
    "Dmin": 30,
    "Dmax": [20, 40]
  }})json";

  const std::string explicit_path = WriteTempParamsFile(explicit_json, "explicit");
  const std::string default_path = WriteTempParamsFile(default_json, "default");
  const std::string staged_path = WriteTempParamsFile(staged_json, "staged");

  const CaseParams explicit_params(explicit_path.c_str());
  const CaseParams default_params(default_path.c_str());
  const CaseParams staged_params(staged_path.c_str());

  if (!CheckElectronCounts(explicit_params, 12, 24)) {
    std::cerr << "Explicit electron-count keys parsed incorrectly." << std::endl;
    return EXIT_FAILURE;
  }
  if (!CheckExtraHoppings(explicit_params, 0.15, 0.05)) {
    std::cerr << "Explicit hopping parameters parsed incorrectly." << std::endl;
    return EXIT_FAILURE;
  }
  if (!CheckElectronCounts(default_params, 12, 24)) {
    std::cerr << "Default manuscript filling parsed incorrectly." << std::endl;
    return EXIT_FAILURE;
  }
  if (!CheckExtraHoppings(default_params, 0.0, 0.0)) {
    std::cerr << "Unset hopping parameters should default to zero." << std::endl;
    return EXIT_FAILURE;
  }
  if (!CheckNoise(explicit_params, {0.2, 0.1})) {
    std::cerr << "Explicit noise schedule parsed incorrectly." << std::endl;
    return EXIT_FAILURE;
  }
  if (!CheckNoise(default_params, {0.0})) {
    std::cerr << "Unset noise should default to a disabled [0.0] schedule." << std::endl;
    return EXIT_FAILURE;
  }
  if (explicit_params.Dmax != std::vector<size_t>({20})) {
    std::cerr << "Scalar Dmax should be promoted to a single-stage schedule." << std::endl;
    return EXIT_FAILURE;
  }
  if (staged_params.Dmax != std::vector<size_t>({20, 40})) {
    std::cerr << "Vector Dmax should be preserved as a staged schedule." << std::endl;
    return EXIT_FAILURE;
  }
  if (staged_params.EffectiveDmin(20) != 20 || staged_params.EffectiveDmin(40) != 30) {
    std::cerr << "Per-stage Dmin clamping is incorrect." << std::endl;
    return EXIT_FAILURE;
  }
  const std::string stage_tag = staged_params.BuildStageTag(1, 40);
  if (!ContainsAll(stage_tag,
                   {"GeometryOBC", "Lx6", "Ly2", "t11", "t20.6", "Jh4",
                    "U12", "delta0", "mu10", "mu2-0.5", "Ndx2y212",
                    "Ndz224", "stage2", "Dmin30", "Dmax40"})) {
    std::cerr << "Stage tag should include the parameter setup and staged D information." << std::endl;
    return EXIT_FAILURE;
  }

  std::remove(explicit_path.c_str());
  std::remove(default_path.c_str());
  std::remove(staged_path.c_str());
  return EXIT_SUCCESS;
}
