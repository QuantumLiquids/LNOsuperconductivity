#!/usr/bin/env python3
"""Plot local spin textures and CTMRG energy histories from completed runs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+", help="Run directories containing summary.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = [json.loads((path / "summary.json").read_text()) for path in args.runs]
    if any(run["status"] != "schedule_completed" for run in runs):
        parser.error("Plots require completed runs")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    fig = plt.figure(figsize=(max(7, 3 * len(runs)), 7))
    fig.subplots_adjust(left=0.09, right=0.94, bottom=0.08, top=0.82, hspace=0.45, wspace=0.35)
    grid = fig.add_gridspec(2, len(runs), height_ratios=[1, 1.1])
    for i, run in enumerate(runs):
        algorithm = run["algorithm"]
        rows, cols = algorithm.get("Rows", 4), algorithm.get("Cols", 4)
        last = run["measurements"][-1]
        spin = np.zeros((rows, cols))
        for site in last["sites"]:
            spin[site["row"], site["col"]] = site["sz_top_z"]
        ax = fig.add_subplot(grid[0, i])
        im = ax.imshow(spin, origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        charge_pattern = algorithm.get("ChargePattern", "stripe_x")
        if charge_pattern == "auto":
            charge_pattern = "stripe_x"
        ax.set_title(algorithm["InitState"].replace("_", " ") +
                     "\ncharge " + charge_pattern.replace("stripe_", ""))
        ax.set(xlabel="x (column)", ylabel="y (row)", xticks=range(cols), yticks=range(rows))
        if i == len(runs) - 1:
            fig.colorbar(im, ax=ax, label=r"$\langle S^z_{\mathrm{top},z}\rangle$")
    ax = fig.add_subplot(grid[1, :])
    for run in runs:
        records = run["measurements"]
        ax.plot([r["step"] for r in records], [r["physical_energy_per_ni"] for r in records],
                "o-", label=run["algorithm"]["InitState"].replace("_", " "), markersize=4)
    ax.set(xlabel="Simple-update sweep", ylabel="CTMRG energy / Ni (table units)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    dimensions = sorted({run["algorithm"].get("Dmax", 2) for run in runs})
    fig.suptitle(f"Projected bilayer iPEPS: short D={','.join(map(str, dimensions))} screening\n"
                 "Textures are seeded states; ground-state convergence is not established", fontsize=11)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
