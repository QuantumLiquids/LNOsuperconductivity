import json
import tempfile
import unittest
from pathlib import Path

from plot.kondo_two_layer_zigzag import postprocess


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name)
        self.spec = postprocess.CaseSpec(
            jperp=0.5,
            jk=-4.0,
            t2=-0.6,
            u=18.0,
            lx=20,
            ly=4,
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _write_json(self, prefix: str, d: int, rows):
        name = f"{prefix}{postprocess.build_file_postfix(self.spec, d)}.json"
        (self.data_dir / name).write_text(json.dumps(rows), encoding="utf-8")

    def _write_complete_spin_set(self, layer_prefix: str, d: int, zz, pm, mp):
        # ref (layer0 local, x=5,y=0) = 81; same-layer targets 97, 101
        self._write_json(layer_prefix + "szsz", d, [[[81, 97], zz], [[81, 101], zz]])
        self._write_json(layer_prefix + "spsm", d, [[[81, 97], pm], [[81, 101], pm]])
        self._write_json(layer_prefix + "smsp", d, [[[81, 97], mp], [[81, 101], mp]])

    def test_find_complete_ds_filters_incomplete_sets(self):
        self._write_complete_spin_set("l0", 500, zz=-0.2, pm=0.1, mp=0.1)
        self._write_complete_spin_set("l1", 500, zz=-0.25, pm=0.12, mp=0.12)

        # Incomplete D=1000 (missing l1smsp)
        self._write_complete_spin_set("l0", 1000, zz=-0.1, pm=0.05, mp=0.05)
        self._write_json("l1szsz", 1000, [[[83, 99], -0.2]])
        self._write_json("l1spsm", 1000, [[[83, 99], 0.1]])

        ds = postprocess.find_complete_ds(self.data_dir, self.spec)
        self.assertEqual(ds, [500])

    def test_layer_purity_check_rejects_mixed_layer_targets(self):
        # layer0 ref 81 but target is layer1 localized (99) => should fail
        self._write_json("l0szsz", 500, [[[81, 99], -0.2]])
        self._write_json("l0spsm", 500, [[[81, 99], 0.1]])
        self._write_json("l0smsp", 500, [[[81, 99], 0.1]])

        with self.assertRaisesRegex(ValueError, "layer 0"):
            postprocess.load_spin_profile_for_layer(self.data_dir, self.spec, 500, layer=0)

    def test_extrapolate_to_infinite_d_linear_in_inverse_d(self):
        # y = 2 + 40*(1/D)
        points = {
            500: {1: 2.08, 2: 1.08},
            1000: {1: 2.04, 2: 1.04},
            2000: {1: 2.02, 2: 1.02},
        }
        extrap = postprocess.extrapolate_vs_inverse_d(points)
        self.assertAlmostEqual(extrap[1], 2.0, places=8)
        self.assertAlmostEqual(extrap[2], 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
