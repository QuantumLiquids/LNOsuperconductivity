#!/usr/bin/env python3
"""Plot compact cent N-only campaign results without filling incomplete cases.

Usage: plot_cent_screen.py campaign_results.json output_directory [--overwrite]
The primary comparison excludes the separately initialized aligned-FM control.
Branch identities always come from the measured final dominant_q field.
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


POINTS = ("table", "midpoint", "candidate")
DIMENSIONS = (2, 3, 4)
DIAGONALS = ("pi2_pi2", "pi2_minus_pi2")
COMPLETED = {"complete", "complete_ctm_recovered"}
COLORS = {"table": "#4477aa", "midpoint": "#dd9944", "candidate": "#228833"}
MARKERS = {"table": "o", "midpoint": "^", "candidate": "s"}
Q_LABELS = {
    "pi2_pi2": r"$(\pi/2,\pi/2)$", "pi2_minus_pi2": r"$(\pi/2,-\pi/2)$",
    "pi_0": r"$(\pi,0)$", "0_pi": r"$(0,\pi)$", "pi_pi": r"$(\pi,\pi)$",
    "pi2_0": r"$(\pi/2,0)$", "0_pi2": r"$(0,\pi/2)$", "0_0": r"$(0,0)$",
    "pi2_pi": r"$(\pi/2,\pi)$", "pi_pi2": r"$(\pi,\pi/2)$",
    "mixed_layers": "mixed layers", "weak_moment": "weak moment",
}
def is_aligned_fm(case):
    algorithm = case["algorithm"]
    return algorithm["InitState"] in {"fm", "ferro"} and algorithm.get("LayerAlignment", -1) == 1


def finite_number(value, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or nonnumeric {context}: {value!r}")
    return float(value)


def audit_campaign(document):
    manifest = document["manifest"]
    if manifest.get("schema") != "lno.cent.nonly.v1":
        raise ValueError("Expected the cent N-only campaign schema lno.cent.nonly.v1")
    cases = manifest["cases"]
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Manifest case identifiers must be unique")
    records = document["records"]
    identifiers = [record["case"]["id"] for record in records]
    if len(identifiers) != len(set(identifiers)) or not set(identifiers) <= set(case_ids):
        raise ValueError("Records have duplicate or unknown case identifiers")
    by_id = {record["case"]["id"]: record for record in records}
    completed, pending, controls = [], [], []
    physics_by_point = {}
    for case in cases:
        algorithm = case["algorithm"]
        duration = sum(finite_number(stage["Tau"], "Tau") * stage["Steps"]
                       for stage in algorithm["Schedule"])
        if abs(duration - 1.3) > 1e-10 or algorithm["Chi"] != 8:
            raise ValueError(f"Case does not share T=1.3, Chi=8: {case['id']}")
        if algorithm["Rows"] != 4 or algorithm["Cols"] != 4:
            raise ValueError(f"Case does not use the common 4x4 cell: {case['id']}")
        point, dimension = case["point"], algorithm["Dmax"]
        if point not in POINTS or dimension not in DIMENSIONS:
            raise ValueError(f"Unknown point or D in case: {case['id']}")
        if point in physics_by_point and case["physics"] != physics_by_point[point]:
            raise ValueError(f"Point has inconsistent physics: {point}")
        physics_by_point[point] = case["physics"]
        record = by_id.get(case["id"], {"case": case, "status": "not_started"})
        if record["case"] != case:
            raise ValueError(f"Record and manifest inputs differ: {case['id']}")
        basic = {"case_id": case["id"], "point": point, "D": dimension,
                 "seed": algorithm["InitState"], "status": record["status"],
                 "layer_alignment": algorithm.get("LayerAlignment", -1), "time": duration,
                 "chi": algorithm["Chi"], "aligned_fm_control": is_aligned_fm(case)}
        if record["status"] not in COMPLETED:
            pending.append(basic)
            if basic["aligned_fm_control"]:
                controls.append(basic)
            continue
        measurement = record.get("measurement")
        if measurement is None:
            raise ValueError(f"Completed record has no measurement: {case['id']}")
        sector = measurement["exact_symmetry_sector"]
        if (sector.get("particles_per_cell") != 48 or sector.get("twice_sz_per_cell") is not None or
                sector.get("symmetry") != "fU1QN(N)"):
            raise ValueError(f"Completed record does not have N-only N48 tensors: {case['id']}")
        if measurement.get("chi") != 8 or measurement.get("ctm_converged") is not True:
            raise ValueError(f"Completed record has different Chi or unconverged CTM: {case['id']}")
        actual_min, actual_max = measurement["bond_dimension_min"], measurement["bond_dimension_max"]
        if not 1 <= actual_min <= actual_max <= dimension:
            raise ValueError(f"Invalid retained bond dimensions: {case['id']}")
        # This field is the campaign collector's measured phase classification.
        # No seed-to-wavevector mapping is used for phase or energy grouping.
        final_q = record.get("dominant_q")
        if not isinstance(final_q, str) or not final_q:
            raise ValueError(f"Completed record has no final dominant_q: {case['id']}")
        n_per_ni = finite_number(measurement["means"]["n_total"], "density") / 2
        entry = dict(basic, final_q=final_q,
                     energy_per_ni=finite_number(measurement["physical_energy_per_ni"], "energy"),
                     D_actual_min=actual_min, D_actual_max=actual_max,
                     n_per_ni=n_per_ni, density_error=abs(n_per_ni - 1.5),
                     layer_q=record.get("layer_q"), amplitude=record.get("dominant_amplitude"),
                     checkpoint=measurement.get("checkpoint"),
                     binary_sha256=record.get("binary_sha256"),
                     recovered_ctm=record["status"] == "complete_ctm_recovered")
        if entry["aligned_fm_control"]:
            controls.append(entry)
        else:
            if entry["layer_alignment"] != -1:
                raise ValueError(f"Unexpected primary layer alignment: {case['id']}")
            completed.append(entry)
    gaps = []
    for point in POINTS:
        for dimension in DIMENSIONS:
            subset = [entry for entry in completed if entry["point"] == point and entry["D"] == dimension]
            best = {}
            for entry in subset:
                q = entry["final_q"]
                if q not in best or entry["energy_per_ni"] < best[q]["energy_per_ni"]:
                    best[q] = entry
            reference = best.get("0_pi")
            for q in DIAGONALS:
                diagonal = best.get(q)
                gap = (diagonal["energy_per_ni"] - reference["energy_per_ni"]
                       if diagonal is not None and reference is not None else None)
                gaps.append({"point": point, "D": dimension, "q": q, "delta_energy_per_ni": gap,
                             "diagonal_case_id": diagonal["case_id"] if diagonal else None,
                             "reference_case_id": reference["case_id"] if reference else None,
                             "reference_energy_per_ni": reference["energy_per_ni"] if reference else None})
    candidate_cases = [case for case in cases if case["point"] == "candidate" and
                       case["algorithm"]["Dmax"] == 3 and not is_aligned_fm(case)]
    required_seeds = {"diagonal_minus", "diagonal_plus", "stripe_x", "stripe_y",
                      "stripe_pi2_x", "stripe_pi2_y", "fm"}
    candidate_ids = {case["id"] for case in candidate_cases}
    candidate_entries = [entry for entry in completed if entry["case_id"] in candidate_ids]
    candidate_pending = sorted(candidate_ids - {entry["case_id"] for entry in candidate_entries})
    candidate_missing_seeds = sorted(required_seeds - {case["algorithm"]["InitState"] for case in candidate_cases})
    candidate_ready = (not candidate_pending and
                       not candidate_missing_seeds)
    return {"schema": "lno.cent.plot.v1", "campaign": manifest["campaign"],
            "common_time": 1.3, "common_chi": 8, "physics_by_point": physics_by_point,
            "completed_primary": completed, "pending_or_failed": pending,
            "aligned_fm_controls": controls, "gaps": gaps,
            "candidate_d3_ready": candidate_ready, "candidate_d3_pending": candidate_pending,
            "candidate_d3_missing_seed_definitions": candidate_missing_seeds,
            "candidate_d3_entries": candidate_entries,
            "max_primary_density_error": max((entry["density_error"] for entry in completed), default=None)}


def save_figure(figure, output, stem):
    for extension in ("png", "pdf"):
        figure.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_gaps(audit, output):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.76, wspace=0.16)
    for axis, q in zip(axes, DIAGONALS):
        axis.axhline(0, color="#444444", linewidth=1.0, zorder=1)
        for point in POINTS:
            by_d = {entry["D"]: entry["delta_energy_per_ni"] for entry in audit["gaps"]
                    if entry["point"] == point and entry["q"] == q}
            values = [by_d[d] if by_d[d] is not None else np.nan for d in DIMENSIONS]
            axis.plot(DIMENSIONS, values, color=COLORS[point], marker=MARKERS[point],
                      linewidth=1.5, markersize=6, label=point.capitalize())
        axis.set(title="Final diagonal q=" + Q_LABELS[q], xlabel="Bond-dimension cap D",
                 xticks=DIMENSIONS, xlim=(1.85, 4.15))
        axis.grid(alpha=0.2)
    axes[0].set_ylabel(r"$\Delta E/Ni=E_{\rm diagonal}-E_{(0,\pi)}$ (table units)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.87),
                  ncol=3, frameon=False)
    figure.suptitle("N-only bilayer iPEPS: preliminary fixed-time screening\n"
                   r"Common $T=1.3$, $\chi=8$; signed gaps from measured final q", fontsize=12, y=0.99)
    figure.text(0.5, 0.045, "Preliminary collinear branches; projection-time and χ convergence remain unchecked.",
                ha="center", va="center", fontsize=9)
    save_figure(figure, output, "cent_signed_gaps")


def plot_candidate(audit, output):
    entries = sorted(audit["candidate_d3_entries"], key=lambda entry: entry["energy_per_ni"])
    reference = min(entry["energy_per_ni"] for entry in entries)
    color_by_q = {"pi2_minus_pi2": "#dd8844", "pi2_pi2": "#aa7744", "0_pi": "#4477aa",
                  "pi_0": "#66aadd", "pi2_0": "#33aaaa", "0_pi2": "#66aa88", "0_0": "#aa66aa"}
    figure, axis = plt.subplots(figsize=(max(10.0, 1.35 * len(entries)), 4.9))
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.80)
    xs = np.arange(len(entries))
    values = [entry["energy_per_ni"] - reference for entry in entries]
    axis.bar(xs, values, width=0.65, color=[color_by_q.get(entry["final_q"], "#888888") for entry in entries])
    for x, value, entry in zip(xs, values, entries):
        axis.annotate(f"E/Ni={entry['energy_per_ni']:.7f}", (x, value), xytext=(0, 5),
                      textcoords="offset points", ha="center", va="bottom", fontsize=8)
    labels = []
    for entry in entries:
        label = Q_LABELS.get(entry["final_q"], entry["final_q"])
        if sum(other["final_q"] == entry["final_q"] for other in entries) > 1:
            label += "\n" + entry["seed"].replace("diagonal_", "d_").replace("stripe_", "s_")
        labels.append(label)
    axis.set_xticks(xs, labels, fontsize=9)
    axis.set_ylabel(r"$(E-E_{\rm lowest\ sampled})/Ni$ (table units)")
    axis.set_title("Candidate parameters: D=3 branch screen\n"
                   r"N-only tensors, common $T=1.3$, $\chi=8$", fontsize=12)
    axis.axhline(0, color="#444444", linewidth=0.9)
    axis.grid(axis="y", alpha=0.2)
    maximum = max(values)
    axis.set_ylim(0, max(0.01, maximum * 1.20))
    figure.text(0.5, 0.045, "Preliminary collinear branches; projection-time and χ convergence remain unchecked.",
                ha="center", va="center", fontsize=9)
    save_figure(figure, output, "cent_candidate_d3_branches")


def write_csv(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true", help="Refresh only this helper's named output files")
    args = parser.parse_args()
    data = args.input.read_bytes()
    audit = audit_campaign(json.loads(data))
    audit["input"] = str(args.input.resolve())
    audit["input_sha256"] = hashlib.sha256(data).hexdigest()
    audit["missing_data_policy"] = "Incomplete or failed cases remain missing; plotted lines do not bridge missing D values."
    audit["aligned_fm_policy"] = "Aligned-FM controls are listed separately and excluded from all primary gaps and candidate-branch comparisons."
    stems = ["cent_signed_gaps", "cent_candidate_d3_branches"]
    owned = [args.output / (stem + suffix) for stem in stems for suffix in [".png", ".pdf"]]
    owned += [args.output / name for name in ["cent_plot_audit.json", "cent_signed_gaps.csv",
                                             "cent_completed_states.csv", "cent_candidate_d3.csv"]]
    if not args.overwrite and any(path.exists() for path in owned):
        parser.error("A named output already exists; choose a new directory or use --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    plot_gaps(audit, args.output)
    if audit["candidate_d3_ready"]:
        plot_candidate(audit, args.output)
    elif args.overwrite:
        for suffix in [".png", ".pdf"]:
            (args.output / ("cent_candidate_d3_branches" + suffix)).unlink(missing_ok=True)
    write_csv(args.output / "cent_signed_gaps.csv", audit["gaps"],
              ["point", "D", "q", "delta_energy_per_ni", "diagonal_case_id", "reference_case_id", "reference_energy_per_ni"])
    write_csv(args.output / "cent_completed_states.csv", audit["completed_primary"],
              ["case_id", "point", "D", "seed", "final_q", "energy_per_ni", "n_per_ni", "density_error",
               "D_actual_min", "D_actual_max", "status", "checkpoint"])
    candidate_rows = audit["candidate_d3_entries"] + [entry for entry in audit["pending_or_failed"]
                      if entry["case_id"] in audit["candidate_d3_pending"]]
    write_csv(args.output / "cent_candidate_d3.csv", sorted(candidate_rows, key=lambda entry: entry["case_id"]),
              ["case_id", "seed", "status", "final_q", "energy_per_ni", "D", "chi", "time",
               "n_per_ni", "D_actual_min", "D_actual_max", "checkpoint"])
    (args.output / "cent_plot_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"completed_primary": len(audit["completed_primary"]),
                      "incomplete_or_failed": len(audit["pending_or_failed"]),
                      "candidate_d3_figure": audit["candidate_d3_ready"],
                      "candidate_d3_pending": audit["candidate_d3_pending"],
                      "candidate_d3_missing_seed_definitions": audit["candidate_d3_missing_seed_definitions"],
                      "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
