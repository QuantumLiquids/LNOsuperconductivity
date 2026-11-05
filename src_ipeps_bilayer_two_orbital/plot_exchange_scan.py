#!/usr/bin/env python3
"""Plot observed final-state energy gaps from the exploratory exchange scan."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POINTS = ["baseline", "scale025", "ratio10_mean", "ratio10_scale025", "strong_anisotropy"]
LABELS = ["Original\ntable", "All AF J\n×0.25", "Jxz ratio 10\nfixed mean", "Ratio 10\nAF J ×0.25",
          "Jxz1=0.154\nJxz2=0.0067\nJx ×0.25", "Jxz1=0.200\nJxz2=0.0067\nJx ×0.25"]
DIAGONALS = ("pi2_pi2", "pi2_minus_pi2")
AXIAL = {"pi_0", "0_pi"}


def audit_point(directory, expected_time=None):
    records = []
    for path in sorted(directory.glob("*/summary.json")):
        run = json.loads(path.read_text())
        if run["status"] != "schedule_completed":
            raise ValueError(f"Incomplete run: {path}")
        algorithm = run["algorithm"]
        measurement = run["measurements"][-1]
        duration = sum(stage["Steps"] * stage["Tau"] for stage in algorithm["Schedule"])
        if expected_time is not None and abs(duration - expected_time) > 1e-10:
            raise ValueError(f"Different imaginary-time schedule: {path}")
        if measurement["bond_dimension_min"] != 2 or measurement["bond_dimension_max"] != 2:
            raise ValueError(f"Actual bond dimension differs from D=2: {path}")
        if measurement["chi"] != 8:
            raise ValueError(f"Environment dimension differs from the fixed Chi=8 screen: {path}")
        if measurement["exact_symmetry_sector"] != {"particles_per_cell": 48, "twice_sz_per_cell": 0}:
            raise ValueError(f"Different canonical sector: {path}")
        layer_q, layer_amplitudes = [], []
        for layer in ("top", "bottom"):
            fourier = measurement["fourier"][f"sz_{layer}_z"]
            q = max(fourier, key=lambda key: fourier[key]["abs"])
            layer_q.append(q)
            layer_amplitudes.append(fourier[q]["abs"])
        records.append({
            "run": str(path.parent.resolve()), "initial_seed": algorithm["InitState"],
            "q": layer_q[0] if layer_q[0] == layer_q[1] else "mixed_layers",
            "layer_q": layer_q, "layer_z_amplitudes": layer_amplitudes,
            "energy_per_ni": measurement["physical_energy_per_ni"],
            "n_per_ni": measurement["means"]["n_total"] / 2,
            "n_top": measurement["means"]["n_top"], "n_bottom": measurement["means"]["n_bottom"],
            "chi": measurement["chi"], "schedule_time": duration,
            "restart_checkpoint": run.get("restart_checkpoint"),
            "checkpoint": measurement["checkpoint"],
        })
    if not records:
        raise ValueError(f"No completed runs in {directory}")
    if any(record["q"] == "mixed_layers" for record in records):
        raise ValueError(f"A run has different dominant magnetic peaks on its layers: {directory}")
    by_q = {}
    for record in records:
        if record["q"] not in by_q or record["energy_per_ni"] < by_q[record["q"]]["energy_per_ni"]:
            by_q[record["q"]] = record
    axial = min((record for record in records if record["q"] in AXIAL), key=lambda r: r["energy_per_ni"])
    gaps = {q: by_q[q]["energy_per_ni"] - axial["energy_per_ni"] for q in DIAGONALS if q in by_q}
    physics_file = directory / "physics.json"
    if physics_file.exists():
        physics = json.loads(physics_file.read_text())
        physics = physics.get("CaseParams", physics)
    else:
        physics = json.loads(next(directory.glob("*/summary.json")).read_text())["physics"]
    return {"directory": str(directory.resolve()), "physics": physics, "records": records,
            "best_by_final_q": by_q, "best_axial": axial, "gaps": gaps,
            "best_sampled": min(records, key=lambda r: r["energy_per_ni"]),
            "max_density_error": max(abs(r["n_per_ni"] - 1.5) for r in records),
            "max_layer_density_error": max(abs(r[k] - 1.5) for r in records for k in ("n_top", "n_bottom"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--candidate-point", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path)
    parser.add_argument("--weak-refined-root", type=Path,
                        help="Matched continuation of the weak Jxz1=0.154 crossing")
    parser.add_argument("--d-controls", type=Path)
    parser.add_argument("--candidate-d-controls", type=Path)
    parser.add_argument("--weak-gap", type=float, default=0.001,
                        help="Visual near-tie guide; this is not an error estimate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output image or audit JSON already exists")
    points = [audit_point(args.scan_root / name, 2.95) for name in POINTS]
    points.append(audit_point(args.candidate_point, 2.95))
    refined = audit_point(args.refined_root, 2.2) if args.refined_root else None
    weak_refined = audit_point(args.weak_refined_root, 1.2) if args.weak_refined_root else None
    fig = plt.figure(figsize=(13.2, 6.2) if args.d_controls else (10.4, 6.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.25]) if args.d_controls else fig.add_gridspec(1, 1)
    ax = fig.add_subplot(grid[0, 0])
    colors = {"pi2_pi2": "#2665a8", "pi2_minus_pi2": "#d36a20"}
    q_labels = {"pi2_pi2": r"$q_+=(\pi/2,\pi/2)$", "pi2_minus_pi2": r"$q_-=(\pi/2,-\pi/2)$"}
    xs = np.arange(len(points))
    ax.axhspan(-args.weak_gap, args.weak_gap, color="#999999", alpha=0.12,
               label=rf"Near-tie guide $|\Delta E|<{args.weak_gap:g}$ (not an error bar)")
    ax.axhline(0, color="#555555", linewidth=0.9)
    for i, q in enumerate(DIAGONALS):
        offset = -0.08 if i == 0 else 0.08
        values = np.array([point["gaps"][q] for point in points])
        ax.scatter(xs + offset, values, s=55, c=colors[q], label=q_labels[q], zorder=3)
        for x, y in zip(xs + offset, values):
            if abs(y) < args.weak_gap:
                ax.scatter([x], [y], s=115, facecolors="none", edgecolors="#222222", linewidths=1.1, zorder=4)
    if weak_refined:
        value = weak_refined["gaps"]["pi2_minus_pi2"]
        ax.scatter([4.20], [value], marker="*", s=155, color=colors["pi2_minus_pi2"],
                   edgecolor="#222222", linewidth=0.6, zorder=5)
        suffix = " (sign reversed)" if value > 0 else ""
        ax.annotate(f"continued T=4.15:\n{value:+.6f}" + suffix, xy=(4.20, value),
                    xytext=(2.65, 0.009), fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8})
    else:
        ax.annotate("near tie; no robust ordering", xy=(4.08, points[4]["gaps"]["pi2_minus_pi2"]),
                    xytext=(2.6, -0.0055), fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8})
    if refined:
        for q in DIAGONALS:
            ax.scatter([5.20], [refined["gaps"][q]], marker="*", s=155,
                       color=colors[q], edgecolor="#222222", linewidth=0.6, zorder=5)
        ax.annotate("candidate continued\nto total T=5.15", xy=(5.20, refined["gaps"]["pi2_minus_pi2"]),
                    xytext=(3.50, -0.012), fontsize=9,
                    arrowprops={"arrowstyle": "->", "color": "#555555", "linewidth": 0.8})
        ax.scatter([], [], marker="*", s=130, color="#777777", edgecolor="#222222",
                   label="Continuations (different total T)")
    ax.set_xticks(xs, LABELS, fontsize=9)
    ax.set_xlim(-0.5, 5.6)
    ax.set_ylim(-0.015, max(point["gaps"][q] for point in points for q in DIAGONALS) + 0.014)
    ax.set_ylabel(r"$\Delta E/Ni=E_{\rm diagonal}-\min(E_{\pi,0},E_{0,\pi})$ (table units)")
    ax.set_title("Cold D=2 scan: total imaginary time T=2.95, fixed χ=8", fontsize=11)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left", fontsize=8.3, frameon=False)
    if args.d_controls:
        dax = fig.add_subplot(grid[0, 1])
        for file, label, color in [(args.d_controls, r"Original: $q_+$", "#2665a8"),
                                   (args.candidate_d_controls, r"Candidate: $q_-$", "#d36a20")]:
            if file is None:
                continue
            with file.open() as stream:
                data = list(csv.DictReader(stream))
            dax.plot([int(row["D"]) for row in data],
                     [float(row["gap_diagonal_minus_axial"]) for row in data], "o-", color=color, label=label)
        dax.axhline(0, color="#555555", linewidth=0.9)
        dax.set_xticks([2, 3, 4])
        dax.set_xlabel("Rung D")
        dax.set_ylabel("Diagonal minus axial / Ni")
        dax.set_title("Separate short D controls\nWarm starts; extra time 0.06", fontsize=10)
        dax.grid(alpha=0.2)
        dax.legend(fontsize=8, frameon=False, loc="center left")
    fig.suptitle("Projected bilayer iPEPS: exchange search and finite-time checks", fontsize=13)
    caption = "Cold starts sample spin/charge branches; final peaks checked separately on both layers.\n"
    caption += "Néel and FM starts were also sampled. Fixed finite χ; no phase boundary or ground-state convergence is established."
    fig.text(0.5, 0.015, caption, ha="center", fontsize=8.5)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.25, top=0.87, wspace=0.34)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    audit = {"cold_total_time": 2.95, "fixed_chi": 8, "cold_points": points,
             "refined_candidate": refined, "refined_weak_crossing": weak_refined,
             "weak_gap_guide": args.weak_gap,
             "interpretation": "Energy gaps among sampled branches at finite imaginary time; not a phase diagram"}
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    for point in points:
        print(Path(point["directory"]).name, point["gaps"])


if __name__ == "__main__":
    main()
