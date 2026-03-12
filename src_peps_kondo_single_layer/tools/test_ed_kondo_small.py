import importlib.util
import pathlib
import sys
import unittest


def _load_ed_module():
    module_path = pathlib.Path(__file__).with_name("ed_kondo_small.py")
    spec = importlib.util.spec_from_file_location("ed_kondo_small", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ed_kondo_small = _load_ed_module()


class EdKondoSmallTest(unittest.TestCase):
    def test_ground_energy_accepts_checkerboard_t2(self):
        energy_with_t2, dim_with_t2 = ed_kondo_small.ground_energy_ed(
            Lx=2,
            Ly=2,
            Ne=4,
            Sz2_total=0,
            t=1.0,
            t2=1.0,
            U=4.0,
            JK=-1.0,
            mu=0.0,
        )
        energy_uniform, dim_uniform = ed_kondo_small.ground_energy_ed(
            Lx=2,
            Ly=2,
            Ne=4,
            Sz2_total=0,
            t=1.0,
            U=4.0,
            JK=-1.0,
            mu=0.0,
        )
        self.assertEqual(dim_with_t2, dim_uniform)
        self.assertAlmostEqual(energy_with_t2, energy_uniform, places=12)

    def test_checkerboard_hopping_matches_peps_pattern(self):
        expected = {
            (0, 1): 1.0,
            (2, 3): 0.3,
            (0, 2): 0.3,
            (1, 3): 1.0,
        }
        bonds = ed_kondo_small.bonds_obc_square(Lx=2, Ly=2, t=1.0, t2=0.3)
        actual = {(site1, site2): hop for site1, site2, hop in bonds}
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
