import json
import pathlib
import sys

import numpy as np


THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
MODULE_DIR = REPO_ROOT / "plot" / "two_layer_two_orbital_all_dof"
if str(MODULE_DIR) not in sys.path:
  sys.path.insert(0, str(MODULE_DIR))

import referee_b_localization_preview as preview


DATA_DIR = REPO_ROOT / "data"
STAGE_TAG = (
    "GeometryOBC_Lx6_Ly2_t11_t21.28623_Jh1.43113_U5.36673_delta0_mu10_mu20_"
    "Ndx2y212_Ndz224_Vmix0_Vdx2y2Perp0_Pin0_stage1_Dmin400_Dmax1200"
)
METADATA_PATH = DATA_DIR / f"dmrg_stage_metadata_{STAGE_TAG}.json"


def test_load_stage_bundle_maps_observables_to_expected_grids():
  bundle = preview.load_stage_bundle(METADATA_PATH)

  assert bundle.stage_tag == STAGE_TAG
  assert bundle.lx == 6
  assert bundle.ly == 2
  assert bundle.one_site["nf_dz2"].shape == (4, 6)
  assert bundle.one_site["single_occ_dx2y2"].shape == (4, 6)

  np.testing.assert_allclose(bundle.one_site["nf_dz2"], np.ones((4, 6)), atol=1e-10)
  assert bundle.summary["nf_dz2"]["mean"] == bundle.one_site["nf_dz2"].mean()
  assert abs(bundle.summary["single_occ_dz2"]["mean"] - 0.9722456899397501) < 1e-10
  assert abs(bundle.summary["charge_var_dz2"]["mean"] - 0.027754310060248644) < 1e-10


def test_build_output_paths_use_stage_tag():
  metadata = json.loads(METADATA_PATH.read_text())
  png_path, pdf_path = preview.build_output_paths(REPO_ROOT / "plot" / "tmp", metadata)

  assert png_path.name == f"referee_b_localization_preview_{STAGE_TAG}.png"
  assert pdf_path.name == f"referee_b_localization_preview_{STAGE_TAG}.pdf"


def test_find_best_stage_metadata_by_u_picks_highest_dmax(tmp_path):
  def write_metadata(name: str, u: float, dmax: int) -> None:
    payload = {
        "StageTag": name.removeprefix("dmrg_stage_metadata_").removesuffix(".json"),
        "U": u,
        "Dmax": dmax,
        "Lx": 6,
        "Ly": 2,
        "NumElectronsDx2Y2": 12,
        "NumElectronsDz2": 24,
    }
    (tmp_path / name).write_text(json.dumps(payload))

  write_metadata("dmrg_stage_metadata_u3_stage1.json", 5.3, 800)
  write_metadata("dmrg_stage_metadata_u3_stage2.json", 5.3, 2000)
  write_metadata("dmrg_stage_metadata_u4_stage1.json", 7.1, 1000)
  write_metadata("dmrg_stage_metadata_u4_stage2.json", 7.1, 3000)

  best = preview.find_best_stage_metadata_by_u(tmp_path)

  assert set(best) == {5.3, 7.1}
  assert best[5.3].name == "dmrg_stage_metadata_u3_stage2.json"
  assert best[7.1].name == "dmrg_stage_metadata_u4_stage2.json"


def test_find_best_stage_metadata_by_u_jh_keeps_distinct_hund_anchors(tmp_path):
  def write_metadata(name: str, u: float, jh: float, dmax: int) -> None:
    payload = {
        "StageTag": name.removeprefix("dmrg_stage_metadata_").removesuffix(".json"),
        "U": u,
        "Jh": jh,
        "Dmax": dmax,
        "Lx": 6,
        "Ly": 2,
        "NumElectronsDx2Y2": 12,
        "NumElectronsDz2": 24,
    }
    (tmp_path / name).write_text(json.dumps(payload))

  write_metadata("dmrg_stage_metadata_u3_jh08_stage1.json", 5.3, 1.4, 800)
  write_metadata("dmrg_stage_metadata_u3_jh08_stage2.json", 5.3, 1.4, 2000)
  write_metadata("dmrg_stage_metadata_u3_jh20_stage1.json", 5.3, 3.6, 1200)
  write_metadata("dmrg_stage_metadata_u3_jh20_stage2.json", 5.3, 3.6, 1800)
  write_metadata("dmrg_stage_metadata_u4_jh08_stage1.json", 7.1, 1.4, 3000)

  best = preview.find_best_stage_metadata_by_u_jh(tmp_path)

  assert set(best) == {(5.3, 1.4), (5.3, 3.6), (7.1, 1.4)}
  assert best[(5.3, 1.4)].name == "dmrg_stage_metadata_u3_jh08_stage2.json"
  assert best[(5.3, 3.6)].name == "dmrg_stage_metadata_u3_jh20_stage2.json"
  assert best[(7.1, 1.4)].name == "dmrg_stage_metadata_u4_jh08_stage1.json"


def test_column_average_returns_expected_x_profile():
  bundle = preview.load_stage_bundle(METADATA_PATH)

  profile = preview.column_average(bundle, "charge_var_dz2")

  assert profile.shape == (6,)
  np.testing.assert_allclose(
      profile,
      np.array([0.056035403244, 0.0, 0.0, 0.0, 0.0, 0.110490456877]),
      atol=1e-9,
  )


def test_exported_figures_do_not_use_editorial_suptitles():
  assert preview.preview_suptitle() is None
  assert preview.comparison_suptitle() is None
