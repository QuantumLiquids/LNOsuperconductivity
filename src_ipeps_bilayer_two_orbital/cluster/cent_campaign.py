#!/usr/bin/env python3
"""Prepare, execute and collect one-core N-only iPEPS Slurm campaigns."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scan_exchange import characterize


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    source = Path(__file__).resolve().parents[1]
    table = read_json(source / "params/physics_table.json")["CaseParams"]
    candidate = read_json(source / "params/physics_diagonal_candidate.json")["CaseParams"]
    base = read_json(source / "params/candidate_d2.json")["CaseParams"]
    base.update(ThreadNum=1, Chi=8, CTMTolerance=1e-5, MaxCTMSweeps=200,
                MinCTMSweeps=2, MeasureInitial=False, MeasureStageEnds=False,
                MeasureEvery=1000, CheckpointEvery=5,
                Schedule=[{"Tau": 0.1, "Steps": 10}, {"Tau": 0.03, "Steps": 10}])
    base.pop("GateCacheDir", None)
    points = []
    for name, fraction in [("table", 0.0), ("midpoint", 0.5), ("candidate", 1.0)]:
        physics = dict(table)
        for key in ["Jx1", "Jx2", "Jxz1", "Jxz2"]:
            physics[key] = (1 - fraction) * table[key] + fraction * candidate[key]
        points.append({"name": name, "candidate_fraction": fraction, "physics": physics})
    cases = []

    def add(point, dimension, seed, charge, spin=0, alignment=-1):
        name = f"{point['name']}_d{dimension}_{seed}_s{spin}_{charge}_layer{alignment:+d}"
        algorithm = dict(base, Dmax=dimension, InitState=seed, ChargePattern=charge,
                         ChargeOffset=0, SpinOffset=spin, LayerAlignment=alignment)
        cases.append({"id": name, "point": point["name"],
                      "candidate_fraction": point["candidate_fraction"],
                      "physics": point["physics"], "algorithm": algorithm})

    # First three tasks are matched D2/D3/D4 pilots at the table parameters.
    for point in points:
        for seed, charge in [("diagonal_minus", "stripe_x"),
                             ("stripe_y", "stripe_x"), ("diagonal_plus", "stripe_x")]:
            for dimension in [2, 3, 4]:
                add(point, dimension, seed, charge)
    # Guards requested in the preceding D3 screen, plus the other axial domain.
    for seed, charge in [("fm", "stripe_x"), ("stripe_pi2_x", "stripe_y"),
                         ("stripe_pi2_y", "stripe_x"), ("stripe_x", "stripe_y")]:
        add(points[-1], 3, seed, charge)
    # Nonzero total Sz is explicitly allowed by the N-only tensor symmetry.
    add(points[0], 2, "fm", "stripe_x", alignment=1)
    manifest = {"schema": "lno.cent.nonly.v1", "campaign": "nonly_d234_screen_20261105",
                "description": "FixedChi8 N-only cold-start comparison at table, midpoint and candidate. Singlephysicalcore/job, nohyperthreads. Collinear starts; Sx/Sz measured. T=1.3 initialscreen, noChi sweep.",
                "cases": cases}
    if output.exists():
        raise FileExistsError(f"Refusing to replace manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, manifest)
    print(f"Prepared {len(cases)} cases; Slurm array0-{len(cases)-1}")


def run_case(deployment, manifest_path, index):
    manifest = read_json(manifest_path)
    if index < 0 or index >= len(manifest["cases"]):
        raise IndexError("Case index outside the manifest")
    case = manifest["cases"][index]
    if (not re.fullmatch(r"[a-zA-Z0-9_.+-]+", case["id"]) or
        not re.fullmatch(r"[a-zA-Z0-9_.+-]+", manifest["campaign"]) or
        case["id"] in [".", ".."] or manifest["campaign"] in [".", ".."]):
        raise ValueError("Unsafe case identifier")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if affinity is not None and len(affinity) != 1:
        raise RuntimeError(f"Expected one bound hardware thread; actual affinity={affinity}")
    binary = deployment / "build/ipeps_bilayer_simple_update_u1"
    case_root = deployment / "runs" / manifest["campaign"] / case["id"]
    if case_root.exists():
        result_path = case_root / "result.json"
        if result_path.exists():
            old = read_json(result_path)
            if old.get("status") in ["complete", "complete_ctm_recovered"] and old["binary_sha256"] == checksum(binary) and old["case"] == case:
                print("Already completed", case["id"], flush=True)
                return 0
        raise FileExistsError(f"Existing case retained; use a fresh campaign for reruns: {case_root}")
    case_root.mkdir(parents=True)
    physics_path, algorithm_path = case_root / "physics.json", case_root / "algorithm.json"
    write_json(physics_path, {"CaseParams": case["physics"]})
    write_json(algorithm_path, {"CaseParams": case["algorithm"]})
    started = time.monotonic()
    record = {"case": case, "status": "running", "binary_sha256": checksum(binary),
              "manifest_sha256": checksum(manifest_path), "affinity": affinity,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
              "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
              "thread_environment": {key: os.environ.get(key) for key in
                                     ["OMP_NUM_THREADS", "BLIS_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}}
    write_json(case_root / "result.json", record)
    print("START", case["id"], "affinity", affinity, flush=True)
    output = case_root / "state"
    command = [str(binary), str(physics_path), str(algorithm_path), str(output)]
    with (case_root / "solver.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    record["solver_returncode"] = result.returncode
    measurement = None
    if result.returncode == 0:
        measurement = read_json(output / "summary.json")["measurements"][-1]
        record["status"] = "complete"
    else:
        latest_path = output / "latest_checkpoint.json"
        planned = sum(stage["Steps"] for stage in case["algorithm"]["Schedule"])
        if latest_path.exists() and read_json(latest_path)["step"] == planned:
            # Evolved state is complete. A longer same-Chi iteration budget is
            # a convergence retry, not a new chi study or further evolution.
            recovery = dict(case["algorithm"], MaxCTMSweeps=500)
            recovery_input = case_root / "recovery_algorithm.json"
            recovery_output = case_root / "recovered_measurement.json"
            write_json(recovery_input, {"CaseParams": recovery})
            checkpoint = read_json(latest_path)["checkpoint"]
            with (case_root / "recovery.log").open("w") as log:
                retried = subprocess.run([str(binary), "--measure", str(physics_path), str(recovery_input),
                                          checkpoint, str(recovery_output)], stdout=log, stderr=subprocess.STDOUT)
            record["recovery_returncode"] = retried.returncode
            if retried.returncode == 0:
                measurement = read_json(recovery_output)["measurement"]
                record["status"] = "complete_ctm_recovered"
    if measurement is not None:
        sector = measurement["exact_symmetry_sector"]
        if sector["particles_per_cell"] != 48 or sector["twice_sz_per_cell"] is not None or sector["symmetry"] != "fU1QN(N)":
            record["status"] = "failed"
            record["error"] = "Campaign must use N-only tensors at N48"
            measurement = None
        else:
            record["measurement"] = measurement
            record.update(characterize(measurement))
    else:
        record["status"] = "failed"
        failure = output / "failure.json"
        record["error"] = read_json(failure).get("error") if failure.exists() else "See solver log"
    record["elapsed_seconds"] = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record["peak_child_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    write_json(case_root / "result.json", record)
    print("END", case["id"], record["status"], "seconds", record["elapsed_seconds"],
          "rss_bytes", record["peak_child_rss_bytes"], flush=True)
    return 0 if measurement is not None else 1


def collect(deployment, manifest_path, output):
    manifest = read_json(manifest_path)
    records, rows = [], []
    for case in manifest["cases"]:
        path = deployment / "runs" / manifest["campaign"] / case["id"] / "result.json"
        record = read_json(path) if path.exists() else {"case": case, "status": "not_started"}
        records.append(record)
        row = {"case_id": case["id"], "point": case["point"],
               "candidate_fraction": case["candidate_fraction"], "D_requested": case["algorithm"]["Dmax"],
               "seed": case["algorithm"]["InitState"], "status": record["status"]}
        if "measurement" in record:
            m = record["measurement"]
            row.update(energy_per_ni=m["physical_energy_per_ni"], n_per_ni=m["means"]["n_total"] / 2,
                       Dmin=m["bond_dimension_min"], Dmax=m["bond_dimension_max"],
                       final_q=record["dominant_q"], amplitude=record["dominant_amplitude"],
                       elapsed_seconds=record["elapsed_seconds"], peak_rss_bytes=record["peak_child_rss_bytes"])
        rows.append(row)
    gaps = []
    for point in ["table", "midpoint", "candidate"]:
        for dimension in [2, 3, 4]:
            subset = [record for record in records if "measurement" in record and
                      record["case"]["point"] == point and record["case"]["algorithm"]["Dmax"] == dimension]
            energies = {}
            for record in subset:
                q = record["dominant_q"]
                energies[q] = min(energies.get(q, float("inf")), record["measurement"]["physical_energy_per_ni"])
            reference = energies.get("0_pi")
            gaps.append({"point": point, "D": dimension, "reference_0_pi": reference,
                         "delta_plus": energies["pi2_pi2"] - reference if "pi2_pi2" in energies and reference is not None else None,
                         "delta_minus": energies["pi2_minus_pi2"] - reference if "pi2_minus_pi2" in energies and reference is not None else None})
    write_json(output, {"manifest": manifest, "records": records, "gaps_relative_to_0_pi": gaps})
    columns = ["case_id", "point", "candidate_fraction", "D_requested", "seed", "status", "energy_per_ni",
               "n_per_ni", "Dmin", "Dmax", "final_q", "amplitude", "elapsed_seconds", "peak_rss_bytes"]
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status_counts": {status: sum(record["status"] == status for record in records)
                                       for status in sorted({record["status"] for record in records})},
                      "gaps": gaps}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("output", type=Path)
    p = sub.add_parser("run"); p.add_argument("deployment", type=Path); p.add_argument("manifest", type=Path); p.add_argument("index", type=int)
    p = sub.add_parser("collect"); p.add_argument("deployment", type=Path); p.add_argument("manifest", type=Path); p.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.output); return 0
    if args.command == "run": return run_case(args.deployment.resolve(), args.manifest.resolve(), args.index)
    collect(args.deployment.resolve(), args.manifest.resolve(), args.output.resolve()); return 0


if __name__ == "__main__":
    raise SystemExit(main())
