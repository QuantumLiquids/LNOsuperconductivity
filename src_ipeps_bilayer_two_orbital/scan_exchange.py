#!/usr/bin/env python3
"""Screen explicit exchange-parameter points with common magnetic/charge starts."""

import argparse
import csv
import json
import os
import re
from pathlib import Path
import subprocess


def read_json(path):
    value = json.loads(path.read_text())
    return value.get("CaseParams", value) if isinstance(value, dict) else value


def write_json(path, value):
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def characterize(measurement):
    fourier = measurement["fourier"]
    names = sorted(set(fourier["sz_top_z"]) & set(fourier["sz_bottom_z"]))
    components = ["sz"] + (["sx"] if "sx_top_z" in fourier and "sx_bottom_z" in fourier else [])
    layer_scores = {layer: {q: sum(fourier[f"{component}_{layer}_z"][q]["abs_squared"]
                                    for component in components) for q in names}
                    for layer in ["top", "bottom"]}
    scores = {q: (layer_scores["top"][q] + layer_scores["bottom"][q]) / 2 for q in names}
    dominant = max(scores, key=scores.get)
    amplitude = scores[dominant] ** 0.5
    layers = []
    for layer in ["top", "bottom"]:
        q = max(names, key=lambda q: layer_scores[layer][q])
        layers.append(q if layer_scores[layer][q] ** 0.5 >= 0.05 else "weak_moment")
    phase = "weak_moment" if amplitude < 0.05 else (
        layers[0] if layers[0] == layers[1] else "mixed_layers")
    return {"dominant_q": phase, "largest_average_q": dominant,
            "spin_components": components,
            "dominant_amplitude": amplitude, "layer_q": layers,
            "q_amplitudes": {q: value ** 0.5 for q, value in scores.items()}}


def record_label(record):
    return (f"{record['initial_seed']}_spin{record['spin_offset']}_charge_"
            f"{record['charge_pattern']}_shift{record['charge_offset']}")


def summarize(records):
    groups = {key: [] for key in ["pi2_pi2", "pi2_minus_pi2", "axial", "quarter_axial",
              "mixed_wavevector", "neel", "fm", "weak_moment", "mixed_layers", "other"]}
    for record in records:
        if record.get("returncode") != 0:
            continue
        q = record["dominant_q"]
        group = ("axial" if q in ["pi_0", "0_pi"] else
                 "quarter_axial" if q in ["pi2_0", "0_pi2"] else
                 "mixed_wavevector" if q in ["pi2_pi", "pi_pi2"] else
                 "neel" if q == "pi_pi" else "fm" if q == "0_0" else q)
        groups[group if group in groups else "other"].append(record)
    best = {key: min(values, key=lambda r: r["energy_per_ni"]) if values else None
            for key, values in groups.items()}
    axial = best["axial"]
    competitors = [value for key, value in best.items()
                   if key not in ["pi2_pi2", "pi2_minus_pi2"] and value is not None]
    competitor = min(competitors, key=lambda r: r["energy_per_ni"]) if competitors else None
    diagonals = [best[key] for key in ["pi2_pi2", "pi2_minus_pi2"] if best[key] is not None]
    diagonal = min(diagonals, key=lambda r: r["energy_per_ni"]) if diagonals else None
    return {"best_by_final_q": best,
            "best_competitor": competitor,
            "delta_diagonal_minus_competitor": diagonal["energy_per_ni"] - competitor["energy_per_ni"]
            if diagonal and competitor else None,
            "delta_plus_minus_axial": best["pi2_pi2"]["energy_per_ni"] - axial["energy_per_ni"]
            if best["pi2_pi2"] and axial else None,
            "delta_minus_minus_axial": best["pi2_minus_pi2"]["energy_per_ni"] - axial["energy_per_ni"]
            if best["pi2_minus_pi2"] and axial else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physics", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, default=Path("build/ipeps_bilayer_simple_update"))
    parser.add_argument("--points", nargs="+")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path, help="Stop cleanly between cases when this file exists")
    parser.add_argument("--starts", nargs="+", help="Select generated start labels from an unchanged manifest")
    args = parser.parse_args()
    baseline, manifest = read_json(args.physics), read_json(args.manifest)
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"Executable not found: {binary}")
    if not isinstance(baseline, dict) or not isinstance(manifest, dict) or set(manifest) != {"description", "algorithm", "points", "starts"}:
        parser.error("Manifest requires description, algorithm, points and starts")
    names = [point["name"] for point in manifest["points"]]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name) for name in names):
        parser.error("Point names must be unique simple directory names")
    allowed_seeds = {"diagonal_plus", "diagonal_minus", "stripe_x", "stripe_y",
                     "stripe_pi2_x", "stripe_pi2_y", "neel", "fm"}
    labels = []
    for start in manifest["starts"]:
        if not isinstance(start, dict) or not set(start) <= {"InitState", "SpinOffset", "ChargePattern", "ChargeOffset", "LayerAlignment"}:
            parser.error("Starts must contain only initialization controls")
        if start.get("InitState") not in allowed_seeds or start.get("ChargePattern", "stripe_x") not in {"auto", "stripe_x", "stripe_y"}:
            parser.error("Unknown initial spin or charge pattern")
        for key in ["SpinOffset", "ChargeOffset"]:
            value = start.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value not in range(4):
                parser.error(f"{key} must be an integer in0..3")
        label = f"{start['InitState']}_spin{start.get('SpinOffset', 0)}_charge_{start.get('ChargePattern', 'stripe_x')}_shift{start.get('ChargeOffset', 0)}"
        labels.append(label)
    if not labels or len(labels) != len(set(labels)):
        parser.error("Start labels must be nonempty and unique")
    if args.starts and not set(args.starts) <= set(labels):
        parser.error("Unknown selected start")
    if args.points and not set(args.points) <= set(names):
        parser.error("Unknown selected parameter point")
    root = args.output.resolve()
    provenance = {"physics": baseline, "manifest": manifest}
    if root.exists():
        if not args.resume or read_json(root / "manifest.json") != provenance:
            parser.error("Existing output requires --resume and identical inputs")
    else:
        root.mkdir(parents=True)
        write_json(root / "manifest.json", provenance)
    selected = ([next(point for point in manifest["points"] if point["name"] == name)
                 for name in args.points] if args.points else manifest["points"])
    stop_file = args.stop_file.resolve() if args.stop_file else root / "stop_requested"
    write_json(root / "active_controller.json", {"pid": os.getpid(), "binary": str(binary),
               "selected_points": [point["name"] for point in selected], "stop_file": str(stop_file)})
    any_failed = False
    for point in selected:
        point_dir = root / point["name"]
        point_dir.mkdir(exist_ok=True)
        physics = dict(baseline, **point["overrides"])
        if not set(point["overrides"]) <= set(baseline):
            parser.error("Point overrides unknown physics fields")
        physics_path = point_dir / "physics.json"
        write_json(physics_path, {"CaseParams": physics})
        records = []
        previous_path = point_dir / "comparison.json"
        if args.starts and previous_path.exists():
            previous = read_json(previous_path)
            if previous["physics"] != physics:
                parser.error("Existing point readout has different physics")
            records = [record for record in previous["records"]
                       if record_label(record) not in args.starts]
        for start in manifest["starts"]:
            if stop_file.exists():
                write_json(root / "scan_state.json", {"status": "stopped_between_runs", "next_point": point["name"]})
                print(f"Stopped between runs because {stop_file} exists", flush=True)
                return int(any_failed)
            seed = start["InitState"]
            spin = start.get("SpinOffset", 0)
            charge = start.get("ChargePattern", "stripe_x")
            offset = start.get("ChargeOffset", 0)
            label = f"{seed}_spin{spin}_charge_{charge}_shift{offset}"
            if args.starts and label not in args.starts:
                continue
            algorithm = dict(manifest["algorithm"], **start,
                             GateCacheDir=str(point_dir / "gate_cache"))
            algorithm_path = point_dir / f"algorithm_{label}.json"
            write_json(algorithm_path, {"CaseParams": algorithm})
            destination = point_dir / label
            summary_path = destination / "summary.json"
            if destination.exists():
                if not args.resume or not summary_path.exists():
                    raise RuntimeError(f"Incomplete existing run retained: {destination}")
                run = read_json(summary_path)
                if run["status"] != "schedule_completed" or run["physics"] != physics or run["algorithm"] != algorithm:
                    raise RuntimeError(f"Existing run differs or is unfinished: {destination}")
                returncode = 0
            else:
                print(f"{point['name']}: {label}", flush=True)
                with (point_dir / f"{label}.log").open("w") as log:
                    result = subprocess.run([str(binary), str(physics_path), str(algorithm_path), str(destination)],
                                            stdout=log, stderr=subprocess.STDOUT, check=False)
                returncode = result.returncode
            record = {"point": point["name"], "initial_seed": seed, "spin_offset": spin,
                      "charge_pattern": charge, "charge_offset": offset, "returncode": returncode}
            if returncode == 0:
                run = read_json(summary_path)
                measurement = run["measurements"][-1]
                previous = run["measurements"][-2] if len(run["measurements"]) > 1 else measurement
                record.update(characterize(measurement))
                record.update(energy_per_ni=measurement["physical_energy_per_ni"],
                              n_per_ni=measurement["means"]["n_total"] / 2,
                              energy_change_last_stage=measurement["physical_energy_per_ni"] - previous["physical_energy_per_ni"],
                              dmin=measurement["bond_dimension_min"], dmax=measurement["bond_dimension_max"],
                              exact_symmetry_sector=measurement["exact_symmetry_sector"],
                              checkpoint=measurement["checkpoint"], step=measurement["step"],
                              elapsed_seconds=run["elapsed_seconds"])
                print(f"  E/Ni={record['energy_per_ni']:.9f}, finalq={record['dominant_q']}, "
                      f"m={record['dominant_amplitude']:.4f}", flush=True)
            else:
                any_failed = True
                failure = destination / "failure.json"
                record["error"] = read_json(failure)["error"] if failure.exists() else "See log"
                print(f"  FAILED: {record['error']}", flush=True)
            records.append(record)
            records.sort(key=lambda value: labels.index(record_label(value)))
            write_json(point_dir / "comparison.json", {"physics": physics, "records": records,
                       "completed_starts": len(records), "planned_starts": len(manifest["starts"]),
                       "complete": len(records) == len(manifest["starts"]),
                       "all_succeeded": len(records) == len(manifest["starts"]) and
                       all(value["returncode"] == 0 for value in records), **summarize(records)})
        aggregate = []
        for existing_point in manifest["points"]:
            existing = root / existing_point["name"] / "comparison.json"
            if existing.exists():
                data = read_json(existing)
                aggregate.append({"point": existing_point["name"], "physics": data["physics"],
                                  "complete": len(data["records"]) == len(manifest["starts"]),
                                  **summarize(data["records"])})
        write_json(root / "scan_summary.json", aggregate)
        columns = ["point", "initial_seed", "spin_offset", "charge_pattern", "charge_offset", "returncode",
                   "energy_per_ni", "dominant_q", "dominant_amplitude", "n_per_ni", "dmin", "dmax", "step",
                   "energy_change_last_stage", "elapsed_seconds", "checkpoint"]
        with (point_dir / "comparison.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
    write_json(root / "scan_state.json", {"status": "selected_points_finished", "failed": any_failed,
               "selected_points": [point["name"] for point in selected]})
    return int(any_failed)


if __name__ == "__main__":
    raise SystemExit(main())
