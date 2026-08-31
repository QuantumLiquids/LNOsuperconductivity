#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_POSTFIX = "t20.3Jk-4U14Ly4Lx20D12001_OBC_Nh4.json"
POSITIVE_SPIN_COLOR = np.array([142, 139, 254]) / 256
NEGATIVE_SPIN_COLOR = np.array([232, 132, 130]) / 256
BASE_MARKER_SIZE = 300.0


def parse_args():
  parser = argparse.ArgumentParser(
      description="Plot one-point and two-point spin patterns for the Nh4 zigzag ladder run."
  )
  parser.add_argument("--ly", type=int, default=4)
  parser.add_argument("--lx", type=int, default=20)
  parser.add_argument("--postfix", default=DEFAULT_POSTFIX)
  parser.add_argument("--data-dir", default=None,
                      help="Directory containing measurement JSON files. Defaults to repo_root/data.")
  parser.add_argument("--out-dir", default=None,
                      help="Directory for output figures. Defaults to plot/kondo_coupled_zigzag_chain/figures.")
  return parser.parse_args()


def load_one_site(path: Path):
  with path.open() as f:
    data = json.load(f)
  return {int(entry[0][0]): float(entry[1]) for entry in data}


def load_two_site(path: Path):
  with path.open() as f:
    data = json.load(f)
  return {(int(entry[0][0]), int(entry[0][1])): float(entry[1]) for entry in data}


def x_y_to_phys_coord(x_chain: int, y_leg: int):
  base = x_chain // 2
  x_phys = base + y_leg
  if x_chain % 2 == 0:
    y_phys = base - y_leg
  else:
    y_phys = base + 1 - y_leg
  return x_phys, y_phys


def even_site_to_chain_leg(mps_site: int, ly: int):
  electron_index = mps_site // 2
  y_leg = electron_index % ly
  x_chain = electron_index // ly
  return x_chain, y_leg


def odd_site_to_chain_leg(mps_site: int, ly: int):
  site_index = (mps_site - 1) // 2
  y_leg = site_index % ly
  x_chain = site_index // ly
  return x_chain, y_leg


def build_site_grid(ly: int, lx: int):
  coords = []
  for x_chain in range(lx):
    for y_leg in range(ly):
      coords.append(x_y_to_phys_coord(x_chain, y_leg))
  return np.array(coords)


def draw_lattice(ax, ly: int, lx: int):
  for x_chain in range(lx - 1):
    for y_leg in range(ly):
      x1, y1 = x_y_to_phys_coord(x_chain, y_leg)
      x2, y2 = x_y_to_phys_coord(x_chain + 1, y_leg)
      ax.plot([x1, x2], [y1, y2], color="0.65", lw=1.0, zorder=1)

  for x_chain in range(lx - 1):
    delta = 1 if x_chain % 2 == 0 else -1
    for y_leg in range(ly):
      target = y_leg + delta
      if 0 <= target < ly:
        x1, y1 = x_y_to_phys_coord(x_chain, y_leg)
        x2, y2 = x_y_to_phys_coord(x_chain + 1, target)
        ax.plot([x1, x2], [y1, y2], color="0.78", lw=0.9, ls="--", zorder=1)


def format_panel(ax):
  ax.set_aspect("equal")
  ax.axis("off")


def add_panel_label(ax, label: str):
  x_lim = ax.get_xlim()
  y_lim = ax.get_ylim()
  width = x_lim[1] - x_lim[0]
  height = y_lim[1] - y_lim[0]
  ax.text(
      x_lim[0] + 0.02 * width,
      y_lim[1] - 0.04 * height,
      label,
      fontsize=16,
      fontweight="bold",
      va="top",
      ha="left",
  )


def bubble_sizes(values, max_abs):
  if max_abs <= 0:
    return np.zeros_like(values, dtype=float)
  return BASE_MARKER_SIZE * np.abs(values) / max_abs


def bubble_colors(values):
  return [POSITIVE_SPIN_COLOR if value >= 0 else NEGATIVE_SPIN_COLOR for value in values]


def format_magnitude(value: float) -> str:
  return f"{value:.2g}"


def rounded_legend_max(value: float) -> float:
  return float(f"{value:.2g}") if value > 0 else 1.0


def add_bubble_legend(fig, max_abs, title: str):
  legend_max = rounded_legend_max(max_abs)
  legend_values = legend_max * np.array([-1.0, -0.5, -0.1, 0.1, 0.5, 1.0])

  legend_ax = fig.add_axes([0.18, 0.01, 0.64, 0.12])
  legend_ax.set_axis_off()
  legend_ax.set_xlim(0, 1)
  legend_ax.set_ylim(0, 1)

  xs = np.linspace(0.08, 0.92, len(legend_values))
  legend_y = 0.65
  text_y = 0.25
  for x, value in zip(xs, legend_values):
    legend_ax.scatter(
        x,
        legend_y,
        s=BASE_MARKER_SIZE * abs(value) / legend_max,
        c=[POSITIVE_SPIN_COLOR if value >= 0 else NEGATIVE_SPIN_COLOR],
        edgecolors="k",
        linewidths=0.6,
    )
    legend_ax.text(x, text_y, format_magnitude(value), ha="center", va="top", fontsize=11)

  legend_ax.text(0.5, 0.96, title + "   (circle area ∝ magnitude)", ha="center", va="top", fontsize=12)


def plot_one_point_panel(ax, data, ly: int, lx: int, is_localized: bool, title: str,
                         label: str, global_max_abs: float):
  draw_lattice(ax, ly, lx)

  coords = []
  values = []
  for mps_site, value in sorted(data.items()):
    if is_localized:
      x_chain, y_leg = odd_site_to_chain_leg(mps_site, ly)
    else:
      x_chain, y_leg = even_site_to_chain_leg(mps_site, ly)
    coords.append(x_y_to_phys_coord(x_chain, y_leg))
    values.append(value)

  coords = np.array(coords)
  values = np.array(values)
  ax.scatter(coords[:, 0], coords[:, 1], s=bubble_sizes(values, global_max_abs),
             c=bubble_colors(values), edgecolors="k", linewidths=0.6, zorder=3)
  ax.set_title(title + "\n" + f"max|<Sz>| = {np.max(np.abs(values)):.3f}", fontsize=12)
  format_panel(ax)
  add_panel_label(ax, label)


def total_spin_correlation(szsz, spsm, smsp):
  corr = {}
  for key, value in szsz.items():
    corr[key] = value + 0.5 * (spsm[key] + smsp[key])
  return corr


def connected_spin_correlation(total_corr, one_point):
  connected = {}
  ref_site = next(iter(total_corr))[0]
  ref_one_point = one_point[ref_site]
  for (ref, target), value in total_corr.items():
    connected[(ref, target)] = value - ref_one_point * one_point[target]
  return connected


def plot_corr_panel(ax, corr_data, ly: int, lx: int, is_localized: bool, title: str,
                    label: str, global_max_abs: float):
  draw_lattice(ax, ly, lx)

  items = sorted(corr_data.items())
  ref_site = items[0][0][0]
  ref_chain, ref_leg = (
      odd_site_to_chain_leg(ref_site, ly) if is_localized else even_site_to_chain_leg(ref_site, ly)
  )
  ref_x, ref_y = x_y_to_phys_coord(ref_chain, ref_leg)

  coords = []
  values = []
  for (_, target), value in items:
    if is_localized:
      x_chain, y_leg = odd_site_to_chain_leg(target, ly)
    else:
      x_chain, y_leg = even_site_to_chain_leg(target, ly)
    coords.append(x_y_to_phys_coord(x_chain, y_leg))
    values.append(value)

  coords = np.array(coords)
  values = np.array(values)
  ax.scatter(coords[:, 0], coords[:, 1], s=bubble_sizes(values, global_max_abs),
             c=bubble_colors(values), edgecolors="k", linewidths=0.6, zorder=3)
  ax.plot(ref_x, ref_y, "kp", markersize=13, markerfacecolor="k", zorder=4)
  ax.set_title(title + "\n" + f"max|value| = {np.max(np.abs(values)):.3f}", fontsize=12)
  format_panel(ax)
  add_panel_label(ax, label)


def save_pair_figure(path_base: Path, fig, rect):
  fig.subplots_adjust(left=0.03, right=0.98, bottom=rect[1], top=rect[3], wspace=0.28)
  fig.savefig(path_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
  fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
  plt.close(fig)


def main():
  args = parse_args()
  script_dir = Path(__file__).resolve().parent
  repo_root = script_dir.parent.parent
  data_dir = Path(args.data_dir) if args.data_dir else repo_root / "data"
  out_dir = Path(args.out_dir) if args.out_dir else script_dir / "figures"
  out_dir.mkdir(parents=True, exist_ok=True)

  postfix = args.postfix
  ly, lx = args.ly, args.lx

  required = {
      "it_sz": data_dir / f"sz_local{postfix}",
      "loc_sz": data_dir / f"lsz_local{postfix}",
      "it_szsz": data_dir / f"szsz{postfix}",
      "it_spsm": data_dir / f"spsm{postfix}",
      "it_smsp": data_dir / f"smsp{postfix}",
      "loc_szsz": data_dir / f"lszsz{postfix}",
      "loc_spsm": data_dir / f"lspsm{postfix}",
      "loc_smsp": data_dir / f"lsmsp{postfix}",
  }
  missing = [str(path) for path in required.values() if not path.exists()]
  if missing:
    raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

  it_sz = load_one_site(required["it_sz"])
  loc_sz = load_one_site(required["loc_sz"])

  it_total = total_spin_correlation(
      load_two_site(required["it_szsz"]),
      load_two_site(required["it_spsm"]),
      load_two_site(required["it_smsp"]),
  )
  loc_total = total_spin_correlation(
      load_two_site(required["loc_szsz"]),
      load_two_site(required["loc_spsm"]),
      load_two_site(required["loc_smsp"]),
  )
  it_connected = connected_spin_correlation(it_total, it_sz)
  loc_connected = connected_spin_correlation(loc_total, loc_sz)

  one_point_max = max(
      np.max(np.abs(np.array(list(it_sz.values())))),
      np.max(np.abs(np.array(list(loc_sz.values())))),
  )
  raw_corr_max = max(
      np.max(np.abs(np.array(list(it_total.values())))),
      np.max(np.abs(np.array(list(loc_total.values())))),
  )
  connected_corr_max = max(
      np.max(np.abs(np.array(list(it_connected.values())))),
      np.max(np.abs(np.array(list(loc_connected.values())))),
  )

  title_base = (
      rf"$L_y={ly}$, $L_x={lx}$, OBC, $t^\prime=0.3$, $J_H=4$, $U=14$, "
      r"$N_h=4$, D=12000"
  )

  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  plot_one_point_panel(axes[0], it_sz, ly, lx, False, r"Itinerant $\langle S^z \rangle$", "(a)",
                       one_point_max)
  plot_one_point_panel(axes[1], loc_sz, ly, lx, True, r"Localized $\langle S^z \rangle$", "(b)",
                       one_point_max)
  fig.suptitle("Nh4 one-point spin texture\n" + title_base, fontsize=15, y=0.98)
  add_bubble_legend(fig, one_point_max, r"$\langle S^z \rangle$")
  save_pair_figure(out_dir / "kondo_ladder_nh4_one_point_sz", fig, rect=[0.0, 0.13, 1.0, 0.83])

  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  plot_corr_panel(axes[0], it_total, ly, lx, False, r"Itinerant $\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$",
                  "(a)", raw_corr_max)
  plot_corr_panel(axes[1], loc_total, ly, lx, True, r"Localized $\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$",
                  "(b)", raw_corr_max)
  fig.suptitle("Nh4 raw spin correlations\n" + title_base, fontsize=15, y=0.98)
  add_bubble_legend(fig, raw_corr_max, r"$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$")
  save_pair_figure(out_dir / "kondo_ladder_nh4_spin_corr_raw", fig, rect=[0.0, 0.13, 1.0, 0.83])

  fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  plot_corr_panel(axes[0], it_connected, ly, lx, False,
                  r"Itinerant connected $\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_c$",
                  "(a)", connected_corr_max)
  plot_corr_panel(axes[1], loc_connected, ly, lx, True,
                  r"Localized connected $\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_c$",
                  "(b)", connected_corr_max)
  fig.suptitle("Nh4 connected spin correlations\n" + title_base, fontsize=15, y=0.98)
  add_bubble_legend(fig, connected_corr_max, r"$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle_c$")
  save_pair_figure(out_dir / "kondo_ladder_nh4_spin_corr_connected", fig, rect=[0.0, 0.13, 1.0, 0.83])

  print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
  main()
