import json
import pathlib
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
EXACT_SUM_BIN = BUILD_DIR / "peps_kondo_2x2_exact_sum_optimize"
VMC_BIN = BUILD_DIR / "peps_kondo_square_vmc_optimize"
PHYSICS_FILE = REPO_ROOT / "src_peps_kondo_single_layer" / "params" / "tests_2x2" / "physics_full_2x2.json"


def write_case_params(path: pathlib.Path, case_params: dict) -> None:
    path.write_text(json.dumps({"CaseParams": case_params}, indent=2) + "\n")


def run_binary(binary: pathlib.Path, algo_params: dict) -> tuple[int, str, pathlib.Path]:
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="peps_opt_params.", dir="/tmp"))
    algo_path = tmpdir / "algo.json"
    write_case_params(algo_path, algo_params)
    proc = subprocess.run(
        [str(binary), str(PHYSICS_FILE), str(algo_path)],
        cwd=tmpdir,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout + proc.stderr, tmpdir


class OptimizerParamsCompatibilityTest(unittest.TestCase):
    def run_and_cleanup(self, binary: pathlib.Path, algo_params: dict) -> tuple[int, str]:
        rc, output, tmpdir = run_binary(binary, algo_params)
        self.addCleanup(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
        return rc, output

    def test_exact_sum_rejects_ambiguous_cg_tolerance_keys(self):
        algo = {
            "OptimizerType": "SR",
            "MaxIterations": 2,
            "LearningRate": 0.1,
            "CGMaxIter": 20,
            "CGTol": 1e-8,
            "CGRelativeTolerance": 1e-4,
            "CGResidualRecomputeInterval": 10,
            "SRDiagShift": 1e-4,
            "Db_min": 4,
            "Db_max": 4,
            "TruncErr": 0.0,
            "MPSCompressScheme": 0,
        }
        rc, output = self.run_and_cleanup(EXACT_SUM_BIN, algo)

        self.assertNotEqual(rc, 0, msg=output)
        self.assertIn("Ambiguous", output)
        self.assertNotIn("missing ./tpsfinal", output)

    def test_exact_sum_rejects_ambiguous_sr_diag_shift_keys(self):
        algo = {
            "OptimizerType": "StochasticReconfiguration",
            "MaxIterations": 2,
            "LearningRate": 0.1,
            "CGMaxIter": 20,
            "CGRelativeTolerance": 1e-4,
            "CGResidualRecomputeInterval": 10,
            "CGDiagShift": 1e-3,
            "SRDiagShift": 1e-4,
            "Db_min": 4,
            "Db_max": 4,
            "TruncErr": 0.0,
            "MPSCompressScheme": 0,
        }
        rc, output = self.run_and_cleanup(EXACT_SUM_BIN, algo)

        self.assertNotEqual(rc, 0, msg=output)
        self.assertIn("Ambiguous", output)
        self.assertNotIn("missing ./tpsfinal", output)

    def test_exact_sum_rejects_ambiguous_minsr_rpinv_keys(self):
        algo = {
            "OptimizerType": "MinSR",
            "MaxIterations": 2,
            "LearningRate": 0.1,
            "MinSRRPinv": 1e-12,
            "MinSRRelativePInv": 1e-10,
            "MinSRAPinv": 0.0,
            "Db_min": 4,
            "Db_max": 4,
            "TruncErr": 0.0,
            "MPSCompressScheme": 0,
        }
        rc, output = self.run_and_cleanup(EXACT_SUM_BIN, algo)

        self.assertNotEqual(rc, 0, msg=output)
        self.assertIn("Ambiguous", output)
        self.assertNotIn("missing ./tpsfinal", output)

    def test_exact_sum_rejects_negative_minsr_rpinv(self):
        algo = {
            "OptimizerType": "MinSR",
            "MaxIterations": 2,
            "LearningRate": 0.1,
            "MinSRRPinv": -1.0,
            "MinSRAPinv": 0.0,
            "Db_min": 4,
            "Db_max": 4,
            "TruncErr": 0.0,
            "MPSCompressScheme": 0,
        }
        rc, output = self.run_and_cleanup(EXACT_SUM_BIN, algo)

        self.assertNotEqual(rc, 0, msg=output)
        self.assertIn("MinSRRPinv must be >= 0", output)
        self.assertNotIn("missing ./tpsfinal", output)

    def test_exact_sum_rejects_negative_minsr_apinv(self):
        algo = {
            "OptimizerType": "MinSR",
            "MaxIterations": 2,
            "LearningRate": 0.1,
            "MinSRRPinv": 1e-12,
            "MinSRAPinv": -1.0,
            "Db_min": 4,
            "Db_max": 4,
            "TruncErr": 0.0,
            "MPSCompressScheme": 0,
        }
        rc, output = self.run_and_cleanup(EXACT_SUM_BIN, algo)

        self.assertNotEqual(rc, 0, msg=output)
        self.assertIn("MinSRAPinv must be >= 0", output)
        self.assertNotIn("missing ./tpsfinal", output)

    def test_vmc_rejects_ambiguous_cg_tolerance_keys(self):
        algo = {
            "OptimizerType": "SR",
            "MaxIterations": 2,
            "LearningRate": 0.1,
            "CGMaxIter": 20,
            "CGTol": 1e-8,
            "CGRelativeTolerance": 1e-4,
            "CGResidualRecomputeInterval": 10,
            "SRDiagShift": 1e-4,
            "WavefunctionBase": "tps",
            "ConfigurationLoadDir": "tpsfinal",
            "ConfigurationDumpDir": "tpsfinal",
            "MC_samples": 10,
            "WarmUp": 2,
            "MCLocalUpdateSweepsBetweenSample": 1,
            "Db_min": 4,
            "Db_max": 4,
            "TruncErr": 1e-10,
            "MPSCompressScheme": 0,
            "ThreadNum": 1,
            "ElectronNum": 4,
            "ElectronSz2": 0,
        }
        rc, output = self.run_and_cleanup(VMC_BIN, algo)

        self.assertNotEqual(rc, 0, msg=output)
        self.assertIn("Ambiguous", output)
        self.assertNotIn("Missing wavefunction directory", output)


if __name__ == "__main__":
    unittest.main()
