#!/usr/bin/env python3
"""Run explicit cold, checkpoint-continuation and CTM-only cent campaigns.

Manifest schema lno.cent.followup.v1:
  {campaign, cases:[{id, kind, point, physics, algorithm, start_total_time,
                    restart_checkpoint? (evolve), checkpoint? (measure)}]}
Inherited time is declared by the manifest. A measure case never evolves its
checkpoint, including during same-Chi convergence retries.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_exchange import characterize


COMPLETE = {"complete", "complete_ctm_recovered"}
BILAYER_EXECUTABLE = "ipeps_bilayer_simple_update_u1"
SINGLE_LAYER_EXECUTABLE = "ipeps_single_layer_simple_update_u1"
THREAD_KEYS = ("OMP_NUM_THREADS", "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS", "GOTO_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identifier(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]*", value):
        raise ValueError(f"Unsafe {name}: {value!r}")


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid {name}: {value!r}")
    return float(value)


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def sites_per_tensor(case):
    return 1 if case.get("executable") == SINGLE_LAYER_EXECUTABLE else 2


def characterize_single_layer(measurement):
    fourier = measurement["fourier"]
    sx, sz = fourier["sx_total"], fourier["sz_total"]
    if not sx or sx.keys() != sz.keys():
        raise ValueError("Inconsistent single-layer Fourier momenta")
    scores = {q: number(sx[q]["abs_squared"], f"Sx power at {q}")
                 + number(sz[q]["abs_squared"], f"Sz power at {q}") for q in sorted(sx)}
    dominant = max(scores, key=scores.get)
    amplitude = math.sqrt(scores[dominant])
    return {"dominant_q": dominant if amplitude >= .05 else "weak_moment",
            "largest_average_q": dominant, "spin_components": ["sz", "sx"],
            "dominant_amplitude": amplitude,
            "q_amplitudes": {q: math.sqrt(value) for q, value in scores.items()}}


def effective_algorithm(case):
    algorithm = dict(case["algorithm"])
    uniform_seed = algorithm.get("ChargePattern") in {
        "uniform_x", "uniform_y", "uniform_type_x", "uniform_type_y"}
    measure_initial = (case["kind"] == "evolve" and not case.get("restart_checkpoint")
                       and uniform_seed and algorithm.get("MeasureInitial", False))
    algorithm.update(ThreadNum=1, MeasureInitial=measure_initial, MeasureStageEnds=False, CheckpointEvery=5)
    if case["kind"] == "measure":
        algorithm["Schedule"] = []  # Explicitly discarded: native --measure never evolves.
        algorithm["MeasureEvery"] = 1
    else:
        algorithm["MeasureEvery"] = sum(stage["Steps"] for stage in algorithm["Schedule"]) + 1
    algorithm.setdefault("MaxCTMSweeps", 200)
    algorithm.setdefault("MinCTMSweeps", 2)
    return algorithm


def timing(case):
    inherited = float(case.get("start_total_time", 0))
    evolution = (sum(stage["Steps"] * stage["Tau"] for stage in case["algorithm"]["Schedule"])
                 if case["kind"] == "evolve" else 0.0)
    return {"inherited_time": inherited, "evolution_time": evolution,
            "planned_total_time": inherited + evolution}


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != "lno.cent.followup.v1":
        raise ValueError("Expected manifest schema lno.cent.followup.v1")
    identifier(manifest.get("campaign"), "campaign")
    if not isinstance(manifest.get("cases"), list) or not manifest["cases"]:
        raise ValueError("Manifest requires a nonempty cases list")
    seen, physics_by_point = set(), {}
    for case in manifest["cases"]:
        if not isinstance(case, dict):
            raise ValueError("Every case must be an object")
        identifier(case.get("id"), "case id")
        identifier(case.get("point"), "point")
        if case["id"] in seen:
            raise ValueError("Duplicate case id: " + case["id"])
        seen.add(case["id"])
        if case.get("kind") not in {"evolve", "measure"}:
            raise ValueError("Case kind must be evolve or measure")
        executable = case.get("executable", BILAYER_EXECUTABLE)
        if executable not in {BILAYER_EXECUTABLE, "ipeps_bilayer_components_u1", SINGLE_LAYER_EXECUTABLE}:
            raise ValueError("Unsupported follow-up executable")
        if executable == "ipeps_bilayer_components_u1" and case["kind"] != "measure":
            raise ValueError("Energy decomposition only measures an unchanged checkpoint")
        if not isinstance(case.get("physics"), dict) or not isinstance(case.get("algorithm"), dict):
            raise ValueError("Case physics and algorithm must be objects")
        for key, value in case["physics"].items():
            number(value, key, minimum=-math.inf)
        if sites_per_tensor(case) == 1 and any(case["physics"].get(key, 0.) != 0.
                                               for key in ("TzPerp", "JzPerp")):
            raise ValueError("Single-layer cases require zero interlayer couplings")
        if case["point"] in physics_by_point and physics_by_point[case["point"]] != case["physics"]:
            raise ValueError("Cases at a named point must have identical physics")
        physics_by_point[case["point"]] = case["physics"]
        algorithm = case["algorithm"]
        for key in ("Rows", "Cols", "Dmax", "Chi"):
            positive_integer(algorithm.get(key), key)
        if algorithm["Rows"] < 2 or algorithm["Cols"] < 2 or algorithm["Rows"] % 2 or algorithm["Cols"] % 2:
            raise ValueError("Checkerboard cells require even Rows and Cols, both at least two")
        if algorithm.get("ThreadNum", 1) != 1 or isinstance(algorithm.get("ThreadNum", 1), bool):
            raise ValueError("Follow-up cases must use ThreadNum=1")
        for key in ("MaxCTMSweeps", "MinCTMSweeps"):
            positive_integer(algorithm.get(key, 200 if key == "MaxCTMSweeps" else 2), key)
        if algorithm.get("MinCTMSweeps", 2) > algorithm.get("MaxCTMSweeps", 200):
            raise ValueError("MinCTMSweeps exceeds MaxCTMSweeps")
        inherited = number(case.get("start_total_time", 0), "start_total_time")
        path_key = "checkpoint" if case["kind"] == "measure" else "restart_checkpoint"
        if case.get(path_key) is not None and (not isinstance(case[path_key], str) or not case[path_key]):
            raise ValueError(f"{path_key} must be a nonempty path")
        if case["kind"] == "measure":
            if not case.get("checkpoint") or "start_total_time" not in case or case.get("restart_checkpoint"):
                raise ValueError("A measure case requires checkpoint and start_total_time, without restart_checkpoint")
        else:
            if case.get("checkpoint"):
                raise ValueError("Evolution uses restart_checkpoint, not checkpoint")
            if inherited != 0 and not case.get("restart_checkpoint"):
                raise ValueError("A cold start must have start_total_time=0")
            if case.get("restart_checkpoint") and "start_total_time" not in case:
                raise ValueError("Checkpoint evolution requires an explicit start_total_time")
            schedule = algorithm.get("Schedule")
            if not isinstance(schedule, list) or not schedule:
                raise ValueError("Evolution requires a nonempty Schedule")
            for stage in schedule:
                if not isinstance(stage, dict) or set(stage) != {"Steps", "Tau"}:
                    raise ValueError("Each stage requires exactly Steps and Tau")
                positive_integer(stage["Steps"], "Steps")
                number(stage["Tau"], "Tau")
        number(timing(case)["planned_total_time"], "planned_total_time")
    return manifest


def checkpoint_path(deployment, value):
    path = Path(value)
    return (path if path.is_absolute() else deployment / path).resolve()


def checkpoint_identity(path, algorithm):
    metadata_path = path / "metadata.txt"
    fields = dict(line.split("=", 1) for line in metadata_path.read_text().splitlines())
    expected = {"schema": "qlpeps.ipeps", "version": "1", "representation": "gamma_lambda",
                "symmetry": "fU1QN", "scalar": "real", "real_bytes": "8",
                "rows": str(algorithm["Rows"]), "cols": str(algorithm["Cols"])}
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint geometry, representation or N-only symmetry differs: " + str(path))
    files = [metadata_path]
    for component in ("Gamma", "lambda_horiz", "lambda_vert"):
        files.extend(path / f"{component}_{row}_{col}.qlten"
                     for row in range(algorithm["Rows"]) for col in range(algorithm["Cols"]))
    if not all(item.is_file() and item.stat().st_size > 0 for item in files):
        raise ValueError("Incomplete Gamma/lambda checkpoint: " + str(path))
    hashes = {item.name: checksum(item) for item in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"path": str(path), "sha256": digest, "files": hashes}


def one_cpu_affinity():
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Run follow-up compute cases on Linux with a one-CPU Slurm binding")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) != 1:
        raise RuntimeError(f"Expected one bound hardware thread, found {cpus}")
    return cpus


def run_case(deployment, manifest_path, index):
    manifest = validate_manifest(read_json(manifest_path))
    if index < 0 or index >= len(manifest["cases"]):
        raise IndexError("Case index outside the manifest")
    case = manifest["cases"][index]
    algorithm = effective_algorithm(case)
    binary = deployment / "build" / case.get("executable", BILAYER_EXECUTABLE)
    binary_hash, manifest_hash = checksum(binary), checksum(manifest_path)
    affinity = one_cpu_affinity()
    environment = dict(os.environ, **{key: "1" for key in THREAD_KEYS}, OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    source_key = "checkpoint" if case["kind"] == "measure" else "restart_checkpoint"
    source = (checkpoint_identity(checkpoint_path(deployment, case[source_key]), algorithm)
              if case.get(source_key) else None)
    if case.get("expected_checkpoint_sha256") and (
            source is None or source["sha256"] != case["expected_checkpoint_sha256"]):
        raise ValueError("Checkpoint differs from the frozen promotion manifest")
    root = deployment / "runs" / manifest["campaign"] / case["id"]
    result_path = root / "result.json"
    if root.exists():
        if result_path.exists():
            old = read_json(result_path)
            if (old.get("status") in COMPLETE and old.get("case") == case and
                    old.get("effective_algorithm") == algorithm and old.get("binary_sha256") == binary_hash and
                    old.get("manifest_sha256") == manifest_hash and old.get("source_checkpoint") == source and
                    checkpoint_identity(Path(old["checkpoint"]), algorithm)["sha256"] == old["checkpoint_identity"]["sha256"]):
                print("Already completed", case["id"], flush=True)
                return 0
        raise FileExistsError("Existing result retained; use a fresh campaign name: " + str(root))
    root.mkdir(parents=True)
    physics_path, algorithm_path = root / "physics.json", root / "algorithm.json"
    write_json(physics_path, {"CaseParams": case["physics"]})
    write_json(algorithm_path, {"CaseParams": algorithm})
    record = {"case": case, "effective_algorithm": algorithm, "status": "running", **timing(case),
              "time_accounting": "manifest-declared inherited time plus evolution schedule",
              "binary_sha256": binary_hash, "manifest_sha256": manifest_hash,
              "workflow_sha256": checksum(Path(__file__)), "source_checkpoint": source,
              "affinity": affinity, "thread_environment": {key: environment[key] for key in THREAD_KEYS},
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
              "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"), "attempts": []}
    write_json(result_path, record)
    started = time.monotonic()
    ni_per_tensor = sites_per_tensor(case)

    def attempt(mode, settings_path, output_path, checkpoint=None, log_name="solver.log"):
        command = ([str(binary), "--measure", str(physics_path), str(settings_path), str(checkpoint), str(output_path)]
                   if mode == "measure" else [str(binary), str(physics_path), str(settings_path), str(output_path)] +
                   ([str(checkpoint)] if checkpoint else []))
        with (root / log_name).open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, check=False)
        record["attempts"].append({"kind": mode, "command": command, "returncode": result.returncode, "log": log_name})
        write_json(result_path, record)
        return result.returncode

    try:
        final_checkpoint, measurement = None, None
        if case["kind"] == "evolve":
            state = root / "state"
            code = attempt("evolve", algorithm_path, state, source["path"] if source else None)
            planned_steps = sum(stage["Steps"] for stage in algorithm["Schedule"])
            latest = read_json(state / "latest_checkpoint.json") if (state / "latest_checkpoint.json").exists() else None
            if latest:
                identity = checkpoint_identity(Path(latest["checkpoint"]), algorithm)
                record.update(checkpoint=identity["path"], checkpoint_identity=identity,
                              checkpoint_step=latest["step"])
                if latest["step"] == planned_steps:
                    final_checkpoint = Path(latest["checkpoint"])
            if algorithm["MeasureInitial"]:
                initial = read_json(state / "checkpoint_step00000000.json")
                targets = ({"n_total": 1.5, "n_x": .5, "n_z": 1.} if ni_per_tensor == 1 else
                           {"n_top": 1.5, "n_bottom": 1.5, "n_top_x": .5, "n_bottom_x": .5,
                            "n_top_z": 1., "n_bottom_z": 1.})
                if (initial["step"] != 0 or initial["bond_dimension_max"] != 2
                        or initial.get("sites_per_tensor", 2) != ni_per_tensor
                        or len(initial["sites"]) != algorithm["Rows"] * algorithm["Cols"]
                        or any(not math.isclose(site[key], value, rel_tol=0, abs_tol=1e-7)
                               for site in initial["sites"] for key, value in targets.items())):
                    raise ValueError("Measured initial state does not have uniform per-layer orbital density")
                record["initial_measurement"] = initial
            if code == 0:
                summary = read_json(state / "summary.json")
                if summary["status"] != "schedule_completed" or summary["executed_steps"] != planned_steps or final_checkpoint is None:
                    raise ValueError("Evolution did not finish its planned schedule")
                measurement = summary["measurements"][-1]
                if measurement["step"] != planned_steps:
                    raise ValueError("No final-step CTM measurement")
        else:
            final_checkpoint = Path(source["path"])
            record.update(checkpoint=source["path"], checkpoint_identity=source)
            code = attempt("measure", algorithm_path, root / "measurement.json", final_checkpoint)
            if code == 0:
                measurement = read_json(root / "measurement.json")["measurement"]
        if measurement is None and final_checkpoint is not None and algorithm["MaxCTMSweeps"] < 500 and \
                "CTMRG did not converge" in (root / "solver.log").read_text():
            recovery = dict(algorithm, MaxCTMSweeps=500, Schedule=[])
            recovery_path = root / "recovery_algorithm.json"
            write_json(recovery_path, {"CaseParams": recovery})
            code = attempt("measure", recovery_path, root / "recovered_measurement.json", final_checkpoint, "recovery.log")
            if code == 0:
                measurement = read_json(root / "recovered_measurement.json")["measurement"]
                record["status"] = "complete_ctm_recovered"
        if measurement is None:
            raise RuntimeError("Final CTM failed; no energy fallback is assigned. See solver/recovery logs.")
        sector = measurement["exact_symmetry_sector"]
        expected_n = 3 * ni_per_tensor * algorithm["Rows"] * algorithm["Cols"] // 2
        if sector != {"particles_per_cell": expected_n, "twice_sz_per_cell": None, "symmetry": "fU1QN(N)"}:
            raise ValueError("Measurement has an incompatible N-only particle sector")
        if measurement.get("sites_per_tensor", 2) != ni_per_tensor:
            raise ValueError("Measurement has an incompatible number of Ni sites per tensor")
        if measurement["chi"] != algorithm["Chi"] or measurement.get("ctm_converged") is not True:
            raise ValueError("Measurement differs from requested Chi or is unconverged")
        if "TxzYSign" in case["physics"] and measurement.get("txz_orientation_signs") != {
                "horizontal": 1, "vertical": case["physics"]["TxzYSign"]}:
            raise ValueError("Measurement does not confirm the requested directional hopping sign")
        number(measurement["physical_energy_per_ni"], "measured energy", minimum=-math.inf)
        number(measurement["means"]["n_total"], "measured density")
        if not 1 <= measurement["bond_dimension_min"] <= measurement["bond_dimension_max"] <= algorithm["Dmax"]:
            raise ValueError("Measured bond dimension exceeds the case cap")
        if case["kind"] == "measure" and measurement["bond_dimension_max"] != algorithm["Dmax"]:
            raise ValueError("CTM-only case Dmax does not describe its unchanged checkpoint")
        phase = characterize_single_layer(measurement) if ni_per_tensor == 1 else characterize(measurement)
        record.update(measurement=measurement, checkpoint=str(final_checkpoint),
                      checkpoint_identity=checkpoint_identity(final_checkpoint, algorithm), **phase)
        if record["status"] != "complete_ctm_recovered":
            record["status"] = "complete"
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
    record["elapsed_seconds"] = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record["peak_child_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    write_json(result_path, record)
    print("END", case["id"], record["status"], flush=True)
    return 0 if record["status"] in COMPLETE else 1


def collect(deployment, manifest_path, output):
    manifest = validate_manifest(read_json(manifest_path))
    manifest_hash = checksum(manifest_path)
    records, rows = [], []
    for case in manifest["cases"]:
        path = deployment / "runs" / manifest["campaign"] / case["id"] / "result.json"
        record = read_json(path) if path.exists() else {"case": case, "status": "not_started", **timing(case)}
        if record["case"] != case or (path.exists() and record.get("manifest_sha256") != manifest_hash):
            raise ValueError("Result differs from manifest: " + case["id"])
        records.append(record)
        row = {"case_id": case["id"], "kind": case["kind"], "point": case["point"],
               "D": case["algorithm"]["Dmax"], "chi": case["algorithm"]["Chi"], **timing(case),
               "status": record["status"], "checkpoint": record.get("checkpoint"),
               "source_checkpoint": (record.get("source_checkpoint") or {}).get("path")}
        if record["status"] in COMPLETE:
            measurement = record["measurement"]
            ni_per_tensor = sites_per_tensor(case)
            row.update(energy_per_ni=measurement["physical_energy_per_ni"],
                       n_per_ni=measurement["means"]["n_total"] / ni_per_tensor,
                       Dmin=measurement["bond_dimension_min"], Dmax=measurement["bond_dimension_max"],
                       final_q=record["dominant_q"],
                       amplitude=record["dominant_amplitude"], elapsed_seconds=record["elapsed_seconds"])
            if ni_per_tensor == 1:
                deviations = [site["n_total"] - 1.5 for site in measurement["sites"]]
                row.update(max_site_density_deviation=max(map(abs, deviations)),
                           rms_site_density_deviation=math.sqrt(math.fsum(x*x for x in deviations) / len(deviations)))
            else:
                row.update(top_q=record["layer_q"][0], bottom_q=record["layer_q"][1])
        rows.append(row)
    groups = {}
    hashes = {record["binary_sha256"] for record in records if record["status"] in COMPLETE}
    if len(hashes) > 1:
        raise ValueError("Completed follow-up results use different numerical binaries")
    for row in rows:
        if row["status"] not in COMPLETE:
            continue
        key = (row["point"], row["D"], round(row["planned_total_time"], 12), row["chi"])
        energies = groups.setdefault(key, {})
        q = row["final_q"]
        if q not in energies or row["energy_per_ni"] < energies[q]["energy_per_ni"]:
            energies[q] = row
    gaps = []
    for (point, dimension, duration, chi), best in sorted(groups.items()):
        reference = best.get("0_pi")
        axial = min((best[q] for q in ("0_pi", "pi_0") if q in best), key=lambda row: row["energy_per_ni"], default=None)
        gaps.append({"point": point, "D": dimension, "planned_total_time": duration, "chi": chi,
                     "best_by_final_q": {q: {"case_id": row["case_id"], "energy_per_ni": row["energy_per_ni"]} for q, row in best.items()},
                     "reference_0_pi": reference["energy_per_ni"] if reference else None,
                     **{f"delta_{name}_vs_0_pi": best[q]["energy_per_ni"] - reference["energy_per_ni"]
                        if q in best and reference else None for name, q in [("plus", "pi2_pi2"), ("minus", "pi2_minus_pi2")]},
                     **{f"delta_{name}_vs_best_axial": best[q]["energy_per_ni"] - axial["energy_per_ni"]
                        if q in best and axial else None for name, q in [("plus", "pi2_pi2"), ("minus", "pi2_minus_pi2")]}})
    write_json(output, {"manifest": manifest, "records": records, "rows": rows, "gaps": gaps})
    columns = ["case_id", "kind", "point", "D", "chi", "inherited_time", "evolution_time", "planned_total_time",
               "status", "energy_per_ni", "n_per_ni", "Dmin", "Dmax", "final_q", "top_q", "bottom_q",
               "amplitude", "elapsed_seconds", "checkpoint", "source_checkpoint"]
    if any(sites_per_tensor(case) == 1 for case in manifest["cases"]):
        columns.extend(["max_site_density_deviation", "rms_site_density_deviation"])
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status_counts": {status: sum(row["status"] == status for row in rows)
                                      for status in sorted({row["status"] for row in rows})}, "gaps": gaps}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    run = commands.add_parser("run")
    run.add_argument("deployment", type=Path); run.add_argument("manifest", type=Path); run.add_argument("index", type=int)
    readout = commands.add_parser("collect")
    readout.add_argument("deployment", type=Path); readout.add_argument("manifest", type=Path); readout.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        manifest = validate_manifest(read_json(args.manifest))
        print(json.dumps({"campaign": manifest["campaign"], "cases": [dict(id=case["id"], kind=case["kind"], **timing(case))
                           for case in manifest["cases"]]}, indent=2))
        return 0
    if args.command == "run":
        return run_case(args.deployment.resolve(), args.manifest.resolve(), args.index)
    collect(args.deployment.resolve(), args.manifest.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
