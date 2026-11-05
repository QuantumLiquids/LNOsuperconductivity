#!/usr/bin/env python3
"""Run competing seeds sequentially and collect final CTMRG measurements."""

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys


def read_params(path):
    value = json.loads(path.read_text())
    return value.get("CaseParams", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physics", type=Path)
    parser.add_argument("algorithm", type=Path)
    parser.add_argument("output", type=Path, help="New comparison directory")
    parser.add_argument("--binary", type=Path, default=Path("build/ipeps_bilayer_simple_update"))
    parser.add_argument("--seeds", nargs="+", default=[
        "diagonal_plus", "diagonal_minus", "stripe_x", "stripe_y", "neel"])
    parser.add_argument("--prepare-only", action="store_true", help="Write inputs without starting calculations")
    parser.add_argument("--charge-patterns", nargs="+", choices=["stripe_x", "stripe_y"])
    parser.add_argument("--charge-offsets", nargs="+", type=int, choices=range(4))
    args = parser.parse_args()
    physics = read_params(args.physics)
    algorithm = read_params(args.algorithm)
    if not isinstance(physics, dict) or not isinstance(algorithm, dict):
        parser.error("Parameter files must contain objects")
    rows, cols = algorithm.get("Rows", 4), algorithm.get("Cols", 4)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 4 or x % 4 for x in (rows, cols)):
        parser.error("Use one common rectangular cell with Rows/Cols divisible by 4")
    allowed = {"diagonal_plus", "diagonal_minus", "stripe_x", "stripe_y",
               "stripe_pi2_x", "stripe_pi2_y", "neel", "fm"}
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds) <= allowed:
        parser.error("Seeds must be unique supported names")
    patterns = args.charge_patterns or [algorithm.get("ChargePattern", "auto")]
    offsets = args.charge_offsets or [algorithm.get("ChargeOffset", 0)]
    if len(set(patterns)) != len(patterns) or len(set(offsets)) != len(offsets):
        parser.error("Charge patterns and offsets must be unique")
    label_charge = args.charge_patterns is not None or args.charge_offsets is not None
    binary = args.binary.resolve()
    if not args.prepare_only and not binary.is_file():
        parser.error(f"Executable not found: {binary}")
    # A new root avoids altering any earlier run, even when preparation fails.
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output.resolve()
    physics_path = root / "physics.json"
    physics_path.write_text(json.dumps({"CaseParams": physics}, indent=2) + "\n")
    summary = []
    failed = False
    for seed, charge_pattern, charge_offset in itertools.product(args.seeds, patterns, offsets):
        label = f"{seed}_charge_{charge_pattern}_shift{charge_offset}" if label_charge else seed
        parameters = dict(algorithm, InitState=seed, ChargePattern=charge_pattern, ChargeOffset=charge_offset)
        algorithm_path = root / f"algorithm_{label}.json"
        algorithm_path.write_text(json.dumps({"CaseParams": parameters}, indent=2) + "\n")
        if args.prepare_only:
            continue
        print(f"Running {label}; log: {root / (label + '.log')}", flush=True)
        with (root / f"{label}.log").open("w") as log:
            result = subprocess.run([str(binary), str(physics_path), str(algorithm_path), str(root / label)],
                                    stdout=log, stderr=subprocess.STDOUT, check=False)
        item = {"initial_seed": seed, "charge_pattern": charge_pattern,
                "charge_offset": charge_offset, "returncode": result.returncode}
        if result.returncode == 0:
            run = json.loads((root / label / "summary.json").read_text())
            measurement = run["measurements"][-1]
            item.update({key: measurement[key] for key in (
                "physical_energy_per_ni", "physical_energy_per_rung", "grand_energy_per_rung",
                "step", "chi", "ctm_sweeps", "bond_dimension_min", "bond_dimension_max", "checkpoint")})
            item["n_per_ni"] = measurement["means"]["n_total"] / 2
            item["top_z_spin_fourier"] = measurement["fourier"]["sz_top_z"]
            print(f"  E/Ni={item['physical_energy_per_ni']:.12g}, n/Ni={item['n_per_ni']:.9g}", flush=True)
        else:
            failed = True
            failure = root / label / "failure.json"
            item["error"] = json.loads(failure.read_text()).get("error") if failure.exists() else "See log"
            print(f"  Failed: {item['error']}", file=sys.stderr, flush=True)
        summary.append(item)
        (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.prepare_only:
        columns = ["initial_seed", "charge_pattern", "charge_offset", "returncode",
                   "physical_energy_per_ni", "n_per_ni", "step", "chi",
                   "bond_dimension_min", "bond_dimension_max", "checkpoint"]
        with (root / "comparison.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)
        print("Collected final states; inspect their spin patterns and Chi/tau dependence before ranking phases.")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
