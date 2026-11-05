#!/usr/bin/env python3
"""Plot frozen-Chi tests, continuations and constrained exchange screens.

Each optional --chi/--time/--scan accepts one or more collection JSON files
from cent_followup.py, or manifest-only placeholders. --prior explicitly adds
the old N-only T=1.3 campaign to the time plot. No missing energy is imputed.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COMPLETE = {"complete", "complete_ctm_recovered"}
DIAGONALS = ("pi2_pi2", "pi2_minus_pi2")
Q_LABELS = {"pi2_pi2": r"$q_+=(\pi/2,\pi/2)$", "pi2_minus_pi2": r"$q_-=(\pi/2,-\pi/2)$"}
POINTS = ("table", "legacy_candidate")
COLORS = {"pi2_pi2": "#4477aa", "pi2_minus_pi2": "#dd8844"}


def load_rows(paths, dataset, inputs):
    merged = {}
    for path in paths:
        data = path.read_bytes(); document = json.loads(data)
        manifest = document.get("manifest", document)
        expected = "lno.cent.nonly.v1" if dataset == "prior" else "lno.cent.followup.v1"
        if manifest.get("schema") != expected:
            raise ValueError(f"Unexpected {dataset} schema in {path}")
        inputs.append({"dataset": dataset, "path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()})
        records = {record["case"]["id"]: record for record in document.get("records", [])}
        for case in manifest["cases"]:
            record = records.get(case["id"], {"case": case, "status": "not_started"})
            if record["case"] != case:
                raise ValueError("Case differs from manifest: " + case["id"])
            algorithm = case["algorithm"]; kind = case.get("kind", "prior")
            inherited = float(case.get("start_total_time", 0))
            schedule = algorithm.get("Schedule", []) if kind != "measure" else []
            total = inherited + sum(stage["Steps"] * stage["Tau"] for stage in schedule)
            point = "legacy_candidate" if dataset == "prior" and case["point"] == "candidate" else case["point"]
            row = {"dataset": dataset, "campaign": manifest["campaign"], "case_id": case["id"],
                   "point": point, "original_point": case["point"], "kind": kind,
                   "D": algorithm["Dmax"], "chi": algorithm["Chi"], "time": round(total, 12),
                   "inherited_time": inherited,
                   "last_tau": schedule[-1]["Tau"] if schedule else case.get("source_last_tau"),
                   "ctm_tolerance": algorithm.get("CTMTolerance"), "status": record["status"],
                   "seed": algorithm["InitState"], "spin_offset": algorithm.get("SpinOffset", 0),
                   "charge_pattern": algorithm.get("ChargePattern"), "physics": case["physics"],
                   "coordinates": case.get("scan_coordinates", {}),
                   "control": algorithm["InitState"] in {"fm", "ferro"} and algorithm.get("LayerAlignment", -1) == 1}
            if row["status"] in COMPLETE:
                measurement = record["measurement"]; energy = measurement["physical_energy_per_ni"]
                if not math.isfinite(energy) or not measurement.get("ctm_converged") or measurement["chi"] != row["chi"]:
                    raise ValueError("Invalid completed energy/CTM controls: " + case["id"])
                if "planned_total_time" in record and abs(record["planned_total_time"] - total) > 1e-10:
                    raise ValueError("Recorded total time differs from manifest: " + case["id"])
                sector = measurement["exact_symmetry_sector"]
                if (sector.get("symmetry") != "fU1QN(N)" or sector.get("twice_sz_per_cell") is not None
                        or sector.get("particles_per_cell") != 3 * algorithm["Rows"] * algorithm["Cols"]):
                    raise ValueError("Follow-up plots require N-only tensors with three electrons per rung")
                row.update(energy=energy, final_q=record["dominant_q"], layer_q=record["layer_q"],
                           n_per_ni=measurement["means"]["n_total"] / 2,
                           binary_sha256=record["binary_sha256"],
                           source_identity={key: (record.get("source_checkpoint") or {}).get(key) for key in ["path", "sha256"]},
                           checkpoint_identity={key: (record.get("checkpoint_identity") or {}).get(key) for key in ["path", "sha256"]})
                if row["final_q"] not in {"mixed_layers", "weak_moment"} and any(q != row["final_q"] for q in row["layer_q"]):
                    raise ValueError("Final-q label disagrees with layer-resolved classification: " + case["id"])
            key = (row["campaign"], row["case_id"])
            old = merged.get(key)
            if old and old["status"] in COMPLETE and row["status"] in COMPLETE and abs(old["energy"] - row["energy"]) > 1e-10:
                raise ValueError("Repeated completed case has different energies: " + case["id"])
            if not old or old["status"] not in COMPLETE or row["status"] in COMPLETE:
                merged[key] = row
    return list(merged.values())


def gap_rows(rows, reference):
    grouped, physics = {}, {}
    for row in rows:
        if row["control"]:
            continue
        if row["point"] in physics and row["physics"] != physics[row["point"]]:
            raise ValueError("Different Hamiltonians share point label: " + row["point"])
        physics[row["point"]] = row["physics"]
        key = (row["point"], row["D"], row["time"], row["chi"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (point, dimension, duration, chi), cases in sorted(grouped.items()):
        if len({row["ctm_tolerance"] for row in cases}) > 1 or len({row["last_tau"] for row in cases}) > 1:
            raise ValueError(f"Conflicting measurement/time-grid controls within {(point, dimension, duration, chi)}")
        finished = [row for row in cases if row["status"] in COMPLETE]
        best = {}
        for row in finished:
            q = row["final_q"]
            if q not in best or row["energy"] < best[q]["energy"]:
                best[q] = row
        refs = [best[q] for q in reference if q in best]
        axial = min(refs, key=lambda row: row["energy"]) if refs else None
        for q in DIAGONALS:
            diagonal = best.get(q)
            output.append({"point": point, "D": dimension, "time": duration, "chi": chi, "q": q,
                           "delta": diagonal["energy"] - axial["energy"] if diagonal and axial else None,
                           "diagonal_energy": diagonal["energy"] if diagonal else None,
                           "axial_energy": axial["energy"] if axial else None,
                           "diagonal_case": diagonal["case_id"] if diagonal else None,
                           "axial_case": axial["case_id"] if axial else None,
                           "completed": len(finished), "planned": len(cases),
                           "point_complete": len(finished) == len(cases)})
    return output


def frozen_audit(rows):
    identities, physics, binaries, tolerances = {}, {}, set(), set()
    for row in rows:
        if row["kind"] != "measure" or row["D"] != 4 or abs(row["time"] - 1.3) > 1e-10 or row["point"] not in POINTS:
            raise ValueError("Chi plots require frozen D4 T=1.3 table/legacy_candidate cases")
        if row["point"] in physics and physics[row["point"]] != row["physics"]:
            raise ValueError("Chi cases have different physics at a named point")
        physics[row["point"]] = row["physics"]
        tolerances.add(row["ctm_tolerance"])
        if row["status"] not in COMPLETE:
            continue
        binaries.add(row["binary_sha256"])
        source = row.get("source_identity") or {}; final = row.get("checkpoint_identity") or {}
        if not source.get("sha256") or final.get("sha256") != source["sha256"]:
            raise ValueError("Frozen measurement lacks an unchanged source-checkpoint fingerprint")
        if row["final_q"] not in [*DIAGONALS, "0_pi"] or row["control"]:
            continue
        key = (row["point"], row["D"], row["final_q"])
        item = identities.setdefault(key, {"sha256": source["sha256"], "chi": []})
        if item["sha256"] != source["sha256"]:
            raise ValueError(f"Different checkpoints would be mixed into frozen-Chi branch {key}")
        item["chi"].append(row["chi"])
    if len(binaries) > 1:
        raise ValueError("Frozen measurements use different numerical binaries")
    if len(tolerances) > 1:
        raise ValueError("Frozen measurements use different CTM convergence tolerances")
    return [{"point": key[0], "D": key[1], "final_q": key[2], **value} for key, value in identities.items()]


def finish(figure, output, name):
    figure.text(0.5, 0.035, "Preliminary sampled branches; convergence is not established for all states.",
                ha="center", fontsize=9)
    for suffix in ["png", "pdf"]:
        figure.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_chi(rows, gaps, output):
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    figure.subplots_adjust(bottom=0.21, top=0.76, wspace=0.15)
    chis = sorted({row["chi"] for row in rows}) or [8, 12, 16]
    for axis, point in zip(axes, POINTS):
        axis.axhline(0, color="#444444", linewidth=0.9)
        for q, marker in zip(DIAGONALS, ["o", "^"]):
            values = {row["chi"]: row["delta"] for row in gaps if row["point"] == point and row["q"] == q}
            axis.plot(chis, [np.nan if values.get(chi) is None else values[chi] for chi in chis],
                      marker=marker, color=COLORS[q], label=Q_LABELS[q])
        axis.set(title=point.replace("_", " "), xlabel=r"Environment dimension $\chi$", xticks=chis)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"$\Delta E/Ni=E_{\rm diagonal}-E_{(0,\pi)}$")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=2, frameon=False)
    figure.suptitle("Frozen D=4 checkpoints: T=1.3, no additional evolution", fontsize=12)
    finish(figure, output, "followup_chi")


def plot_time(rows, gaps, output):
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.7), sharey=True)
    figure.subplots_adjust(bottom=0.25, top=0.73, wspace=0.16)
    times = sorted({row["time"] for row in rows})
    for axis, dimension in zip(axes, [3, 4]):
        axis.axhline(0, color="#444444", linewidth=0.9)
        for point, color in zip(POINTS, ["#4477aa", "#228833"]):
            for q, style, marker in zip(DIAGONALS, ["-", "--"], ["o", "^"]):
                values = {row["time"]: row["delta"] for row in gaps
                          if row["point"] == point and row["D"] == dimension and row["q"] == q}
                axis.plot(times, [np.nan if values.get(t) is None else values[t] for t in times],
                          linestyle=style, marker=marker, color=color, label=point.replace("_", " ") + ": " + Q_LABELS[q])
        labels = []
        for duration in times:
            taus = sorted({row["last_tau"] for row in rows if row["time"] == duration and row["last_tau"] is not None})
            labels.append(f"{duration:g}\n" + (r"$\tau_{\rm last}=$" + ",".join(f"{tau:g}" for tau in taus)))
        axis.set(title=f"D={dimension}", xlabel="Accumulated imaginary time T", xticks=times)
        axis.set_xticklabels(labels)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"$\Delta E/Ni=E_{\rm diagonal}-E_{(0,\pi)}$")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    figure.suptitle("Matched continuations at χ=8: cold prior and checkpoint restarts", fontsize=12)
    finish(figure, output, "followup_time")


def plot_scan(rows, gaps, output):
    points = list(dict.fromkeys(row["point"] for row in rows))
    chis = sorted({row["chi"] for row in rows})
    figure, axes = plt.subplots(len(chis), 1, squeeze=False, figsize=(max(10.4, 1.3 * len(points)), 3.8 * len(chis) + 1.1))
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.20 if len(chis) == 1 else 0.16, top=0.88, hspace=0.75)
    for axis, chi in zip(axes[:, 0], chis):
        subset = [row for row in rows if row["chi"] == chi]
        values = [entry for entry in gaps if entry["chi"] == chi]
        axis.axhline(0, color="#444444", linewidth=0.9)
        for index, q in enumerate(DIAGONALS):
            for x, point in enumerate(points):
                item = next((entry for entry in values if entry["point"] == point and entry["q"] == q), None)
                if item and item["delta"] is not None:
                    axis.scatter(x + (-0.07 if index == 0 else 0.07), item["delta"], s=45,
                                 facecolors=COLORS[q] if item["point_complete"] else "none", edgecolors=COLORS[q])
            axis.scatter([], [], s=45, color=COLORS[q], label=Q_LABELS[q])
        labels = []
        for point in points:
            cases = [row for row in subset if row["point"] == point]
            coordinates = next(row["coordinates"] for row in rows if row["point"] == point)
            if "interpolation_fraction" in coordinates:
                name = f"λ={coordinates['interpolation_fraction']:g}\ns={coordinates['af_scale']:g}"
            elif "jxz2_over_jxz1" in coordinates:
                name = f"boundary r={coordinates['jxz2_over_jxz1']:g}\ns={coordinates['af_scale']:g}"
            else:
                name = point.replace("_", "\n")
            labels.append(name + f"\n{sum(row['status'] in COMPLETE for row in cases)}/{len(cases)} starts")
        axis.set_xticks(range(len(points)), labels, fontsize=9)
        axis.set_xlim(-0.5, len(points) - 0.5)
        axis.set_ylabel(r"$\Delta E/Ni=E_{\rm diagonal}-\min(E_{(0,\pi)},E_{(\pi,0)})$")
        axis.set_title(f"χ={chi}: final-q minima over available spin/charge starts", fontsize=11)
        axis.grid(axis="y", alpha=0.2)
        if any(not entry["point_complete"] for entry in values):
            axis.scatter([], [], s=45, facecolors="none", edgecolors="#666666", label="Partial starts")
        axis.legend(frameon=False, loc="best", fontsize=9)
    figure.suptitle("Constrained exchange scan: D=3, T=2.95", fontsize=12)
    finish(figure, output, "followup_constrained_scan")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["chi", "time", "scan"]:
        parser.add_argument("--" + name, type=Path, nargs="+", default=[])
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not (args.chi or args.time or args.scan):
        parser.error("Supply at least one --chi, --time or --scan input")
    if args.time and not args.prior:
        parser.error("--time requires an explicit --prior N-only campaign JSON")
    inputs = []
    data = {name: load_rows(getattr(args, name), name, inputs) for name in ["chi", "time", "scan"]}
    prior = load_rows([args.prior], "prior", inputs) if args.prior else []
    frozen = frozen_audit(data["chi"])
    for row in data["time"]:
        if row["kind"] != "evolve" or row["point"] not in POINTS or row["D"] not in [3, 4] or row["chi"] != 8:
            raise ValueError("Time curves require D3/D4 χ8 table/legacy_candidate continuations")
    time_rows = data["time"] + [row for row in prior if row["point"] in POINTS and row["D"] in [3, 4] and row["chi"] == 8]
    for row in data["scan"]:
        if row["D"] != 3 or abs(row["time"] - 2.95) > 1e-10 or (row["kind"] != "measure" and row["inherited_time"] != 0):
            raise ValueError("Constrained scan requires cold D3 T=2.95 cases or their frozen remeasurements")
    gaps = {"chi": gap_rows(data["chi"], ["0_pi"]), "time": gap_rows(time_rows, ["0_pi"]),
            "scan": gap_rows(data["scan"], ["0_pi", "pi_0"])}
    args.output.mkdir(parents=True, exist_ok=True)
    names = ["followup_" + name for name, paths in [("chi", args.chi), ("time", args.time), ("constrained_scan", args.scan)] if paths]
    if not args.overwrite and any((args.output / (name + suffix)).exists() for name in names for suffix in [".png", ".pdf"]):
        parser.error("Output figure exists; choose another directory or --overwrite")
    if args.chi: plot_chi(data["chi"], gaps["chi"], args.output)
    if args.time: plot_time(time_rows, gaps["time"], args.output)
    if args.scan: plot_scan(data["scan"], gaps["scan"], args.output)
    for name, entries in gaps.items():
        if not getattr(args, name): continue
        with (args.output / f"{name}_gaps.csv").open("w", newline="") as stream:
            columns = ["point", "D", "time", "chi", "q", "delta", "diagonal_energy", "axial_energy",
                       "diagonal_case", "axial_case", "completed", "planned", "point_complete"]
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(entries)
    all_rows = data["chi"] + time_rows + data["scan"]
    with (args.output / "followup_states.csv").open("w", newline="") as stream:
        columns = ["dataset", "campaign", "case_id", "point", "D", "time", "chi", "last_tau", "ctm_tolerance",
                   "status", "seed", "spin_offset", "charge_pattern", "final_q", "energy", "n_per_ni"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore"); writer.writeheader(); writer.writerows(all_rows)
    audit = {"inputs": inputs, "frozen_checkpoint_checks": frozen, "rows": all_rows, "gaps": gaps,
             "counts": {name: {"completed": sum(row["status"] in COMPLETE for row in data[name]),
                               "planned": len(data[name]),
                               "missing_gaps": sum(entry["delta"] is None for entry in gaps[name])}
                        for name in data},
             "policies": ["Missing energies stay missing; only completed measurements enter minima.",
                          "Phase groups use measured final q, excluding weak/mixed states from diagonals.",
                          "Prior candidate aliases only to legacy_candidate; constrained point IDs remain separate.",
                          "Hollow scan markers denote minima over incomplete sets of starts; no phase boundary is inferred."]}
    (args.output / "followup_plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit["counts"], indent=2))


if __name__ == "__main__":
    main()
