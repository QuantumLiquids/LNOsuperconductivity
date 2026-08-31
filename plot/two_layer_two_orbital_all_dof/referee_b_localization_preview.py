#!/usr/bin/env python3
"""Preview plots for Referee B localization/Mottness observables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ONE_SITE_OBSERVABLES = (
    "nf_dx2y2",
    "nupndn_dx2y2",
    "single_occ_dx2y2",
    "charge_var_dx2y2",
    "sz_dx2y2",
    "nf_dz2",
    "nupndn_dz2",
    "single_occ_dz2",
    "charge_var_dz2",
    "sz_dz2",
)

DISPLAY_NAMES = {
    "nf": r"$\langle n \rangle$",
    "nupndn": r"$\langle n_\uparrow n_\downarrow \rangle$",
    "single_occ": r"$P_\mathrm{single}$",
    "charge_var": r"$\delta n^2$",
}

ROW_LABELS = ("top y=0", "top y=1", "bottom y=0", "bottom y=1")
ORBITAL_LABELS = {"dx2y2": r"$d_{x^2-y^2}$", "dz2": r"$d_{z^2}$"}
SUMMARY_SPECS = (
    ("nf_dz2", r"$\langle n \rangle_{z^2}$", "#0f766e"),
    ("single_occ_dz2", r"$P_{\mathrm{single},z^2}$", "#14b8a6"),
    ("nupndn_dz2", r"$\langle n_\uparrow n_\downarrow \rangle_{z^2}$", "#ef4444"),
    ("charge_var_dz2", r"$\delta n^2_{z^2}$", "#0f766e"),
    ("nf_dx2y2", r"$\langle n \rangle_{x^2-y^2}$", "#b45309"),
    ("charge_var_dx2y2", r"$\delta n^2_{x^2-y^2}$", "#d97706"),
)


@dataclass(frozen=True)
class StageBundle:
  metadata: Dict[str, object]
  stage_tag: str
  lx: int
  ly: int
  one_site: Dict[str, np.ndarray]
  summary: Dict[str, Dict[str, float]]


def _load_json(path: Path):
  return json.loads(path.read_text())


def _load_one_site_array(path: Path, lx: int, ly: int) -> np.ndarray:
  raw = _load_json(path)
  values = [entry[1] for entry in raw]
  expected_size = 2 * lx * ly
  if len(values) != expected_size:
    raise ValueError(
        f"{path.name} has {len(values)} one-site entries; expected {expected_size} for Lx={lx}, Ly={ly}."
    )
  return np.asarray(values, dtype=float).reshape(lx, 2 * ly).T


def _summarize(array: np.ndarray) -> Dict[str, float]:
  return {
      "mean": float(array.mean()),
      "min": float(array.min()),
      "max": float(array.max()),
      "std": float(array.std()),
  }


def load_stage_bundle(metadata_path: Path) -> StageBundle:
  metadata = _load_json(metadata_path)
  stage_tag = str(metadata["StageTag"])
  data_dir = metadata_path.parent
  lx = int(metadata["Lx"])
  ly = int(metadata["Ly"])

  one_site = {}
  summary = {}
  for observable in ONE_SITE_OBSERVABLES:
    path = data_dir / f"{observable}_dmrg_two_layer_two_orbital_{stage_tag}.json"
    array = _load_one_site_array(path, lx, ly)
    one_site[observable] = array
    summary[observable] = _summarize(array)

  return StageBundle(
      metadata=metadata,
      stage_tag=stage_tag,
      lx=lx,
      ly=ly,
      one_site=one_site,
      summary=summary,
  )


def build_output_paths(output_dir: Path, metadata: Dict[str, object]) -> Tuple[Path, Path]:
  stage_tag = str(metadata["StageTag"])
  return (
      output_dir / f"referee_b_localization_preview_{stage_tag}.png",
      output_dir / f"referee_b_localization_preview_{stage_tag}.pdf",
  )


def build_comparison_output_paths(output_dir: Path) -> Tuple[Path, Path]:
  return (
      output_dir / "referee_b_localization_compare_best_by_u.png",
      output_dir / "referee_b_localization_compare_best_by_u.pdf",
  )


def find_best_stage_metadata_by_u(data_dir: Path) -> Dict[float, Path]:
  best: Dict[float, Tuple[int, Path]] = {}
  for path in sorted(data_dir.glob("dmrg_stage_metadata_*.json")):
    try:
      metadata = _load_json(path)
    except json.JSONDecodeError:
      continue
    if "U" not in metadata or "Dmax" not in metadata:
      continue
    if "NumElectronsDx2Y2" not in metadata or "NumElectronsDz2" not in metadata:
      continue
    u_value = float(metadata["U"])
    dmax = int(metadata["Dmax"])
    current_best = best.get(u_value)
    if current_best is None or dmax > current_best[0]:
      best[u_value] = (dmax, path)
  return {u_value: path for u_value, (_, path) in best.items()}


def find_best_stage_metadata_by_u_jh(data_dir: Path) -> Dict[Tuple[float, float], Path]:
  best: Dict[Tuple[float, float], Tuple[int, Path]] = {}
  for path in sorted(data_dir.glob("dmrg_stage_metadata_*.json")):
    try:
      metadata = _load_json(path)
    except json.JSONDecodeError:
      continue
    if "U" not in metadata or "Jh" not in metadata or "Dmax" not in metadata:
      continue
    if "NumElectronsDx2Y2" not in metadata or "NumElectronsDz2" not in metadata:
      continue
    key = (float(metadata["U"]), float(metadata["Jh"]))
    dmax = int(metadata["Dmax"])
    current_best = best.get(key)
    if current_best is None or dmax > current_best[0]:
      best[key] = (dmax, path)
  return {key: path for key, (_, path) in best.items()}


def column_average(bundle: StageBundle, observable: str) -> np.ndarray:
  if observable not in bundle.one_site:
    raise KeyError(f"Unknown observable {observable!r}")
  return bundle.one_site[observable].mean(axis=0)


def column_band(bundle: StageBundle, observable: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  if observable not in bundle.one_site:
    raise KeyError(f"Unknown observable {observable!r}")
  array = bundle.one_site[observable]
  return array.mean(axis=0), array.min(axis=0), array.max(axis=0)


def bundle_case_label(bundle: StageBundle) -> str:
  return (
      rf"$U/t_1={float(bundle.metadata['U']):.2f}$, "
      rf"$J_H/t_1={float(bundle.metadata['Jh']):.2f}$, "
      rf"$D={int(bundle.metadata['Dmax'])}$"
  )


def preview_suptitle() -> None:
  return None


def comparison_suptitle() -> None:
  return None


def _annotate_heatmap(ax: plt.Axes, array: np.ndarray, fmt: str) -> None:
  for row in range(array.shape[0]):
    for col in range(array.shape[1]):
      color = "white" if array[row, col] > (array.max() + array.min()) / 2 else "#1f2937"
      ax.text(col, row, format(array[row, col], fmt), ha="center", va="center", fontsize=8, color=color)


def _draw_scalar_heatmap(
    ax: plt.Axes,
    array: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    show_y: bool,
    fmt: str,
) -> None:
  im = ax.imshow(array, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
  ax.set_title(title, fontsize=11, pad=8)
  ax.set_xticks(range(array.shape[1]), [str(i + 1) for i in range(array.shape[1])], fontsize=9)
  if show_y:
    ax.set_yticks(range(array.shape[0]), ROW_LABELS, fontsize=9)
  else:
    ax.set_yticks(range(array.shape[0]), [])
  ax.set_xlabel("x", fontsize=9)
  _annotate_heatmap(ax, array, fmt)
  for spine in ax.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("#cbd5e1")
  return im


def _draw_summary_panel(ax: plt.Axes, bundle: StageBundle) -> None:
  metrics = ("nf", "nupndn", "single_occ", "charge_var")
  y = np.arange(len(metrics))
  dx_vals = [bundle.summary[f"{metric}_dx2y2"]["mean"] for metric in metrics]
  dz_vals = [bundle.summary[f"{metric}_dz2"]["mean"] for metric in metrics]
  width = 0.36

  ax.barh(y - width / 2, dx_vals, height=width, color="#d97706", label=ORBITAL_LABELS["dx2y2"])
  ax.barh(y + width / 2, dz_vals, height=width, color="#0f766e", label=ORBITAL_LABELS["dz2"])
  ax.set_yticks(y, [DISPLAY_NAMES[m] for m in metrics], fontsize=10)
  ax.set_xlim(0, 1.05)
  ax.set_xlabel("site average", fontsize=10)
  ax.set_title(r"$d_{z^2}$ already sits near the projected local-spin limit", fontsize=12, pad=10)
  ax.grid(axis="x", alpha=0.25)
  ax.legend(frameon=False, fontsize=10, loc="lower right")
  for values, y_shift in ((dx_vals, -width / 2), (dz_vals, width / 2)):
    for i, value in enumerate(values):
      ax.text(value + 0.015, y[i] + y_shift, f"{value:.3f}", va="center", fontsize=9)

  callout = "\n".join((
      rf"{ORBITAL_LABELS['dz2']}: "
      rf"$\langle n \rangle={bundle.summary['nf_dz2']['mean']:.3f}$, "
      rf"$\langle n_\uparrow n_\downarrow \rangle={bundle.summary['nupndn_dz2']['mean']:.3f}$",
      rf"$P_\mathrm{{single}}={bundle.summary['single_occ_dz2']['mean']:.3f}$, "
      rf"$\delta n^2={bundle.summary['charge_var_dz2']['mean']:.3f}$",
  ))
  ax.text(
      0.52,
      0.96,
      callout,
      transform=ax.transAxes,
      fontsize=9,
      color="#134e4a",
      va="top",
      ha="left",
      bbox={"boxstyle": "round,pad=0.3", "facecolor": "#f0fdfa", "edgecolor": "#99f6e4"},
  )


def _draw_sz_panel(ax: plt.Axes, array: np.ndarray, orbital_key: str, show_y: bool) -> None:
  title = (
      rf"{ORBITAL_LABELS[orbital_key]} $S^z$ "
      rf"(mean $|S^z|={np.mean(np.abs(array)):.3f}$)"
  )
  _draw_scalar_heatmap(ax, array, title, -0.5, 0.5, "RdBu_r", show_y, ".2f")


def render_preview(bundle: StageBundle, output_dir: Path) -> Tuple[Path, Path]:
  output_dir.mkdir(parents=True, exist_ok=True)
  png_path, pdf_path = build_output_paths(output_dir, bundle.metadata)

  fig = plt.figure(figsize=(17, 11), constrained_layout=True)
  gs = fig.add_gridspec(3, 4, height_ratios=(1.0, 1.0, 0.95))

  scalar_specs = (
      ("nf", 0.0, 1.0, "viridis", ".2f"),
      ("nupndn", 0.0, max(bundle.summary["nupndn_dx2y2"]["max"], bundle.summary["nupndn_dz2"]["max"]) or 1.0, "magma", ".3f"),
      ("single_occ", 0.0, 1.0, "viridis", ".2f"),
      ("charge_var", 0.0, max(bundle.summary["charge_var_dx2y2"]["max"], bundle.summary["charge_var_dz2"]["max"]), "magma_r", ".3f"),
  )

  for row, orbital_key in enumerate(("dx2y2", "dz2")):
    for col, (metric, vmin, vmax, cmap, fmt) in enumerate(scalar_specs):
      ax = fig.add_subplot(gs[row, col])
      array = bundle.one_site[f"{metric}_{orbital_key}"]
      mean_val = bundle.summary[f"{metric}_{orbital_key}"]["mean"]
      title = rf"{ORBITAL_LABELS[orbital_key]} {DISPLAY_NAMES[metric]} (mean {mean_val:.3f})"
      im = _draw_scalar_heatmap(ax, array, title, vmin, vmax, cmap, col == 0, fmt)
      cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
      cbar.ax.tick_params(labelsize=8)

  summary_ax = fig.add_subplot(gs[2, :2])
  _draw_summary_panel(summary_ax, bundle)

  ax_sz_dx = fig.add_subplot(gs[2, 2])
  _draw_sz_panel(ax_sz_dx, bundle.one_site["sz_dx2y2"], "dx2y2", False)

  ax_sz_dz = fig.add_subplot(gs[2, 3])
  _draw_sz_panel(ax_sz_dz, bundle.one_site["sz_dz2"], "dz2", False)

  metadata = bundle.metadata
  suptitle = preview_suptitle()
  if suptitle:
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
  fig.text(
      0.5,
      0.004,
      "Rows resolve the four sites in each x-slice: top-layer y=0, top-layer y=1, bottom-layer y=0, bottom-layer y=1.",
      ha="center",
      fontsize=10,
      color="#475569",
  )

  fig.savefig(png_path, dpi=220, bbox_inches="tight")
  fig.savefig(pdf_path, bbox_inches="tight")
  plt.close(fig)
  return png_path, pdf_path


def render_comparison(bundles: Iterable[StageBundle], output_dir: Path) -> Tuple[Path, Path]:
  bundle_list = list(bundles)
  if not bundle_list:
    raise ValueError("At least one StageBundle is required for comparison plotting.")

  output_dir.mkdir(parents=True, exist_ok=True)
  png_path, pdf_path = build_comparison_output_paths(output_dir)

  fig = plt.figure(figsize=(15.2, 3.9 * len(bundle_list)), constrained_layout=True)
  gs = fig.add_gridspec(len(bundle_list), 3, width_ratios=(1.25, 1.0, 1.0))

  for row, bundle in enumerate(bundle_list):
    x = np.arange(1, bundle.lx + 1)

    ax_summary = fig.add_subplot(gs[row, 0])
    y = np.arange(len(SUMMARY_SPECS))
    values = [bundle.summary[metric]["mean"] for metric, _, _ in SUMMARY_SPECS]
    colors = [color for _, _, color in SUMMARY_SPECS]
    labels = [label for _, label, _ in SUMMARY_SPECS]
    bars = ax_summary.barh(y, values, color=colors, height=0.66)
    ax_summary.set_yticks(y, labels, fontsize=10)
    ax_summary.invert_yaxis()
    ax_summary.set_xlim(0.0, 1.05)
    ax_summary.grid(axis="x", alpha=0.22)
    ax_summary.set_xlabel("site average", fontsize=10)
    ax_summary.set_title(bundle_case_label(bundle), fontsize=12, pad=8, loc="left")
    for bar, value in zip(bars, values):
      ax_summary.text(min(value + 0.018, 1.01), bar.get_y() + bar.get_height() / 2,
                      f"{value:.3f}", va="center", ha="left", fontsize=9)

    ax_occ = fig.add_subplot(gs[row, 1])
    occ_specs = (
        ("nf_dz2", r"$\langle n\rangle_{z^2}$", "#0f766e", "o"),
        ("single_occ_dz2", r"$P_{\mathrm{single},z^2}$", "#14b8a6", "s"),
        ("nupndn_dz2", r"$\langle n_\uparrow n_\downarrow\rangle_{z^2}$", "#ef4444", "^"),
    )
    for observable, label, color, marker in occ_specs:
      mean, lower, upper = column_band(bundle, observable)
      ax_occ.fill_between(x, lower, upper, alpha=0.10, color=color)
      ax_occ.plot(x, mean, marker=marker, linewidth=2.2, color=color, label=label)
    ax_occ.set_xticks(x)
    ax_occ.set_ylim(-0.02, 1.05)
    ax_occ.grid(alpha=0.22)
    ax_occ.set_xlabel("x")
    ax_occ.set_ylabel("column average")
    ax_occ.set_title(r"$d_{z^2}$ occupancy diagnostics", fontsize=11, pad=8)
    if row == 0:
      ax_occ.legend(frameon=False, fontsize=9, loc="lower center")

    ax_charge = fig.add_subplot(gs[row, 2])
    charge_specs = (
        ("charge_var_dz2", r"$\delta n^2_{z^2}$", "#0f766e", "o"),
        ("charge_var_dx2y2", r"$\delta n^2_{x^2-y^2}$", "#d97706", "s"),
    )
    for observable, label, color, marker in charge_specs:
      mean, lower, upper = column_band(bundle, observable)
      ax_charge.fill_between(x, lower, upper, alpha=0.12, color=color)
      ax_charge.plot(x, mean, marker=marker, linewidth=2.4, color=color, label=label)
    ax_charge.set_xticks(x)
    ax_charge.set_ylim(-0.01, 0.32)
    ax_charge.grid(alpha=0.22)
    ax_charge.set_xlabel("x")
    ax_charge.set_ylabel("column average")
    ax_charge.set_title(r"Charge fluctuation remains smaller in $d_{z^2}$", fontsize=11, pad=8)
    if row == 0:
      ax_charge.legend(frameon=False, fontsize=9, loc="upper center")

  suptitle = comparison_suptitle()
  if suptitle:
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")
  fig.text(
      0.5,
      0.004,
      "Lines show x-column means; shaded bands show the min-max spread over the four sites in each x-slice.",
      ha="center",
      fontsize=10,
      color="#475569",
  )
  fig.savefig(png_path, dpi=220, bbox_inches="tight")
  fig.savefig(pdf_path, bbox_inches="tight")
  plt.close(fig)
  return png_path, pdf_path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--metadata",
      type=Path,
      nargs="*",
      help="One or more dmrg_stage_metadata_*.json files. Omit together with --auto-best-by-u.",
  )
  parser.add_argument(
      "--auto-best-by-u",
      action="store_true",
      help="Scan --data-dir and pick the current highest-D stage for each U.",
  )
  parser.add_argument(
      "--auto-best-by-u-jh",
      action="store_true",
      help="Scan --data-dir and pick the current highest-D stage for each (U, Jh) pair.",
  )
  parser.add_argument(
      "--data-dir",
      type=Path,
      default=Path("data"),
      help="Directory used by --auto-best-by-u.",
  )
  parser.add_argument(
      "--comparison",
      action="store_true",
      help="Render the two-panel U-comparison figure instead of the single-dataset preview.",
  )
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=Path("plot/two_layer_two_orbital_all_dof/figures"),
      help="Directory for output figures.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  metadata_paths: List[Path]
  if args.auto_best_by_u:
    metadata_paths = [path for _, path in sorted(find_best_stage_metadata_by_u(args.data_dir).items())]
  elif args.auto_best_by_u_jh:
    metadata_paths = [path for _, path in sorted(find_best_stage_metadata_by_u_jh(args.data_dir).items())]
  else:
    metadata_paths = list(args.metadata or [])

  if not metadata_paths:
    raise SystemExit("Provide --metadata ... or use --auto-best-by-u.")

  bundles = [load_stage_bundle(path) for path in metadata_paths]
  if args.comparison:
    png_path, pdf_path = render_comparison(bundles, args.output_dir)
  else:
    if len(bundles) != 1:
      raise SystemExit("Single-dataset preview expects exactly one metadata file.")
    png_path, pdf_path = render_preview(bundles[0], args.output_dir)
  print(json.dumps({"png": str(png_path), "pdf": str(pdf_path)}, indent=2))


if __name__ == "__main__":
  main()
