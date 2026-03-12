import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
SU_BIN = BUILD_DIR / "peps_kondo_square_simple_update"


def write_case_params(path: pathlib.Path, case_params: dict) -> None:
    path.write_text(json.dumps({"CaseParams": case_params}, indent=2) + "\n")


def read_configuration(config_path: pathlib.Path) -> list[list[int]]:
    rows = []
    for line in config_path.read_text().strip().splitlines():
        rows.append([int(x) for x in line.split()])
    return rows


def run_simple_update(physics_params: dict, algo_params: dict) -> tuple[int, str, pathlib.Path]:
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="peps_init_state_test.", dir="/tmp"))
    write_case_params(tmpdir / "physics.json", physics_params)
    write_case_params(tmpdir / "algo.json", algo_params)
    proc = subprocess.run(
        [str(SU_BIN), "physics.json", "algo.json"],
        cwd=tmpdir,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr, tmpdir


class SimpleUpdateInitStateTest(unittest.TestCase):
    def test_random_initializer_keeps_neel_local_spins_and_counts(self):
        physics = {
            "Lx": 4,
            "Ly": 4,
            "t": 1.0,
            "t2": 0.3,
            "U": 0.0,
            "Jk": -1.0,
            "Mu": 0.0,
            "ElectronNum": 8,
            "ElectronSz2": 0,
        }
        algo = {
            "Dmin": 1,
            "Dmax": 2,
            "TruncErr": 1e-10,
            "Tau": 0.01,
            "Step": 1,
            "ThreadNum": 1,
        }
        rc, output, tmpdir = run_simple_update(physics, algo)
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

        self.assertEqual(rc, 0, msg=output)
        actual = read_configuration(tmpdir / "tpsfinal" / "configuration0")
        num_up = 0
        num_down = 0
        for row, row_values in enumerate(actual):
            for col, combined in enumerate(row_values):
                local_up = (combined % 2) == 0
                self.assertEqual(local_up, (row + col) % 2 == 0, msg=output)
                electron = combined // 2
                if electron == 1:
                    num_up += 1
                elif electron == 2:
                    num_down += 1
                else:
                    self.assertEqual(electron, 3, msg=output)
        self.assertEqual(num_up, 4, msg=output)
        self.assertEqual(num_down, 4, msg=output)

    def test_stripe_pi2pi2_initial_configuration(self):
        physics = {
            "Lx": 4,
            "Ly": 4,
            "t": 1.0,
            "t2": 0.3,
            "U": 0.0,
            "Jk": -1.0,
            "Mu": 0.0,
            "ElectronNum": 8,
            "ElectronSz2": 0,
            "InitState": "stripe_pi2pi2",
        }
        algo = {
            "Dmin": 1,
            "Dmax": 2,
            "TruncErr": 1e-10,
            "Tau": 0.01,
            "Step": 1,
            "ThreadNum": 1,
        }
        rc, output, tmpdir = run_simple_update(physics, algo)
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

        self.assertEqual(rc, 0, msg=output)
        actual = read_configuration(tmpdir / "tpsfinal" / "configuration0")
        expected = [
            [5, 7, 2, 6],
            [6, 5, 7, 2],
            [2, 6, 5, 7],
            [7, 2, 6, 5],
        ]
        self.assertEqual(actual, expected, msg=output)

    def test_stripe_pi0_initial_configuration(self):
        physics = {
            "Lx": 4,
            "Ly": 4,
            "t": 1.0,
            "t2": 0.3,
            "U": 0.0,
            "Jk": -1.0,
            "Mu": 0.0,
            "ElectronNum": 8,
            "ElectronSz2": 0,
            "InitState": "stripe_pi0",
        }
        algo = {
            "Dmin": 1,
            "Dmax": 2,
            "TruncErr": 1e-10,
            "Tau": 0.01,
            "Step": 1,
            "ThreadNum": 1,
        }
        rc, output, tmpdir = run_simple_update(physics, algo)
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

        self.assertEqual(rc, 0, msg=output)
        actual = read_configuration(tmpdir / "tpsfinal" / "configuration0")
        expected = [
            [2, 5, 6, 7],
            [5, 6, 7, 2],
            [6, 7, 2, 5],
            [7, 2, 5, 6],
        ]
        self.assertEqual(actual, expected, msg=output)

    def test_stripe_pi2pi2_doped_hole_removal(self):
        physics = {
            "Lx": 4,
            "Ly": 4,
            "t": 1.0,
            "t2": 0.3,
            "U": 0.0,
            "Jk": -1.0,
            "Mu": 0.0,
            "ElectronNum": 6,
            "ElectronSz2": 0,
            "InitState": "stripe_pi2pi2",
        }
        algo = {
            "Dmin": 1,
            "Dmax": 2,
            "TruncErr": 1e-10,
            "Tau": 0.01,
            "Step": 1,
            "ThreadNum": 1,
        }
        rc, output, tmpdir = run_simple_update(physics, algo)
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

        self.assertEqual(rc, 0, msg=output)
        actual = read_configuration(tmpdir / "tpsfinal" / "configuration0")
        expected = [
            [7, 7, 6, 6],
            [6, 5, 7, 2],
            [2, 6, 5, 7],
            [7, 2, 6, 5],
        ]
        self.assertEqual(actual, expected, msg=output)

    def test_stripe_init_state_rejects_nonzero_electron_sz(self):
        physics = {
            "Lx": 4,
            "Ly": 4,
            "t": 1.0,
            "t2": 0.3,
            "U": 0.0,
            "Jk": -1.0,
            "Mu": 0.0,
            "ElectronNum": 8,
            "ElectronSz2": 2,
            "InitState": "stripe_pi2pi2",
        }
        algo = {
            "Dmin": 1,
            "Dmax": 2,
            "TruncErr": 1e-10,
            "Tau": 0.01,
            "Step": 1,
            "ThreadNum": 1,
        }
        rc, output, tmpdir = run_simple_update(physics, algo)
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

        self.assertNotEqual(rc, 0, msg=output)
        self.assertIn("ElectronSz2 = 0", output)

    def test_stripe_init_state_rejects_unsupported_geometry(self):
        physics = {
            "Lx": 2,
            "Ly": 2,
            "t": 1.0,
            "t2": 0.3,
            "U": 0.0,
            "Jk": -1.0,
            "Mu": 0.0,
            "ElectronNum": 2,
            "ElectronSz2": 0,
            "InitState": "stripe_pi2pi2",
        }
        algo = {
            "Dmin": 1,
            "Dmax": 2,
            "TruncErr": 1e-10,
            "Tau": 0.01,
            "Step": 1,
            "ThreadNum": 1,
        }
        rc, output, tmpdir = run_simple_update(physics, algo)
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))

        self.assertNotEqual(rc, 0, msg=output)
        self.assertIn("cannot realize total Sz_total = 0", output)


if __name__ == "__main__":
    unittest.main()
