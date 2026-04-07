"""plot/kondo_1d_chain/plot_corr_and_phase_2panel.py

Python port of plot_corr_and_phase_2panel.m.

Two-panel figure for the 1D Kondo chain:
  - Left:  phase diagram in (U, J_H) plane with FM and 2k_F-SDW regions.
  - Right: log-log spin correlation F(r) for two parameter sets and two
           orbital channels (itinerant d_{x^2-y^2} = circles,
           localized d_{z^2} = squares). Color encodes phase
           (FM = purple, 2k_F-SDW = teal). Sign: filled = +, hollow = -.
           Power-law fit on the itinerant SDW series. y = 1/4 reference
           shown as a custom y-tick on the right axis.
"""

from __future__ import annotations

import json
import os
import re
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, LogFormatterMathtext
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------------------
FM_COLOR  = np.array([27, 158, 119]) / 255.0   # teal
SDW_COLOR = np.array([117, 112, 179]) / 255.0  # purple

# Order matches the MATLAB params(1)=FM, params(2)=SDW; correlation panel
# swaps these so FM curves are drawn purple, SDW curves teal (matches the
# label coloring in the phase panel).
SERIES_COLORS_CORR = np.array([SDW_COLOR, FM_COLOR])

FONT_NAME = "Arial"
FS_AXES = 22
FS_LEGEND = 16
FS_LABEL = 18
LW_AXES = 1.8
LW_PLOT = 2.0
MARKER_SIZE = 7
MARKER_EDGE_WIDTH = 1.5

DATA_DIR = Path(__file__).resolve().parent / ".." / ".." / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _extract_D(name: str) -> int:
    m = re.search(r"D(\d+)\.json$", name)
    return int(m.group(1)) if m else 0


def _pick_file(files: list[str], prefer_suffix: str, fallback_name: str = "") -> str:
    for f in files:
        if f.endswith(prefer_suffix):
            return f
    if fallback_name:
        for f in files:
            if os.path.basename(f) == fallback_name:
                return f
    if files:
        return files[0]
    raise FileNotFoundError("no candidate files")


def _parse_corr(zz_path: str, pm_path: str):
    """Return (ref_idx, tgt_idx_array, corr_array) where corr = <S^z S^z> + <S^+ S^->.

    JSON layout: list of [[i, j], [re, im]] entries.
    """
    with open(zz_path) as f:
        zz = json.load(f)
    with open(pm_path) as f:
        pm = json.load(f)
    n = len(zz)
    ref_idx = zz[0][0][0]
    tgt = np.zeros(n, dtype=int)
    val = np.zeros(n, dtype=float)
    for i in range(n):
        tgt[i] = zz[i][0][1]
        val[i] = zz[i][1][0] + pm[i][1][0]
    return ref_idx, tgt, val


def load_orbital_phase(prefix_zz: str, prefix_pm: str, jk: int, U: int, L: int):
    """Load one (orbital, phase) series; return (dist, abs_corr, sign_corr)."""
    base_token = f"Jk{jk}U{U}L{L}"
    sz_files = sorted(glob(str(DATA_DIR / f"{prefix_zz}{base_token}*.json")))
    pm_files = sorted(glob(str(DATA_DIR / f"{prefix_pm}{base_token}*.json")))
    if not sz_files or not pm_files:
        raise FileNotFoundError(f"missing {prefix_zz}/{prefix_pm} for {base_token}")
    Ds = sorted({_extract_D(os.path.basename(f)) for f in sz_files} |
                {_extract_D(os.path.basename(f)) for f in pm_files})
    used_D = max(Ds)
    suffix = f"D{used_D}.json" if used_D > 0 else f"{base_token}.json"
    sz = _pick_file(sz_files, suffix, f"{base_token}.json")
    pm = _pick_file(pm_files, suffix, f"{base_token}.json")
    ref_idx, tgt, corr = _parse_corr(sz, pm)
    raw_dist = tgt - ref_idx
    if np.all(raw_dist % 2 == 0):
        dist = raw_dist // 2
    else:
        dist = raw_dist
    return dist.astype(float), np.abs(corr), np.sign(corr)


# ---------------------------------------------------------------------------
# Right panel: spin correlation
# ---------------------------------------------------------------------------
def draw_corr_panel(ax):
    series_colors = SERIES_COLORS_CORR  # row 0 = FM (purple), row 1 = SDW (teal)
    L = 100
    params = [
        dict(Jk=-10, U=10),  # k=0 → FM-like, drawn purple
        dict(Jk=-2,  U=4),   # k=1 → SDW-like, drawn teal
    ]
    orbitals = [
        dict(zz="szsz",  pm="spsm",  marker="o"),  # itinerant d_{x^2-y^2}
        dict(zz="lszsz", pm="lspsm", marker="s"),  # localized d_{z^2}
    ]
    n_orb = len(orbitals)
    n_set = len(params)

    dist_all = [[None] * n_set for _ in range(n_orb)]
    corr_all = [[None] * n_set for _ in range(n_orb)]
    sign_all = [[None] * n_set for _ in range(n_orb)]
    for o, orb in enumerate(orbitals):
        for k, p in enumerate(params):
            d, c, s = load_orbital_phase(orb["zz"], orb["pm"], p["Jk"], p["U"], L)
            dist_all[o][k] = d
            corr_all[o][k] = c
            sign_all[o][k] = s

    # Plot lines + markers per (orbital, phase). Color = phase, shape = orbital.
    for o, orb in enumerate(orbitals):
        mk = orb["marker"]
        for k in range(n_set):
            x = dist_all[o][k]
            y = corr_all[o][k]
            s = sign_all[o][k]
            mask = (x > 0) & (x < 60)
            x = x[mask]; y = y[mask]; s = s[mask]
            col = series_colors[k]
            ax.plot(x, y, "-", color=col, lw=LW_PLOT, zorder=2)
            pos = s >= 0
            neg = s < 0
            ax.plot(x[pos], y[pos], marker=mk, ls="none",
                    mec=col, mfc=col, mew=MARKER_EDGE_WIDTH, ms=MARKER_SIZE,
                    zorder=3)
            ax.plot(x[neg], y[neg], marker=mk, ls="none",
                    mec=col, mfc="none", mew=MARKER_EDGE_WIDTH, ms=MARKER_SIZE,
                    zorder=3)

    # --- Power-law fits on the SDW series for both orbitals (k=1) ---------
    # Drop the longest few points where the cylinder finite-size cutoff
    # contaminates the tail.
    k_sdw = 1
    fits = {}  # orb_idx -> dict(slope, intercept, alpha)
    x_target = np.arange(6, 41, 2)  # exclude r > 40 from the fit
    for o_fit in (0, 1):
        x_all = dist_all[o_fit][k_sdw]
        y_all = corr_all[o_fit][k_sdw]
        sel = np.isin(x_target, x_all)
        if not np.any(sel):
            continue
        x_fit = x_target[sel]
        idx_lookup = {int(d): i for i, d in enumerate(x_all)}
        y_fit = np.array([y_all[idx_lookup[int(xx)]] for xx in x_fit])
        slope, intercept = np.polyfit(np.log(x_fit), np.log(y_fit), 1)
        alpha = -slope
        xl = np.logspace(np.log10(x_fit.min()), np.log10(60), 200)
        yl = np.exp(intercept) * xl ** slope
        ax.plot(xl, yl, "--", color=series_colors[k_sdw],
                lw=LW_PLOT, zorder=2)
        fits[o_fit] = dict(slope=slope, intercept=intercept, alpha=alpha)

    # Axes setup
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 60)
    ax.set_xlabel(r"$r$", fontname=FONT_NAME, fontsize=FS_AXES)
    ax.set_ylabel(r"Spin correlation $F(r)$", fontname=FONT_NAME, fontsize=FS_AXES)
    ax.tick_params(axis="both", which="major", labelsize=FS_AXES, width=LW_AXES, length=6)
    ax.tick_params(axis="both", which="minor", width=LW_AXES * 0.7, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(LW_AXES)

    # --- Custom y-tick at 1/4 (the localized FM saturation value) ----------
    # We add it to the *minor* tick locator so it coexists with the default
    # log major ticks 1e-3, 1e-2, 1e-1, 1e0, and label it as "1/4".
    ax.figure.canvas.draw()  # finalize ylim
    y_lo, y_hi = ax.get_ylim()
    # Make sure 1/4 is visible.
    if y_hi < 0.4:
        ax.set_ylim(y_lo, 0.6)
    # Draw a horizontal reference line at y=1/4.
    ax.axhline(0.25, color=[0.35, 0.35, 0.35], ls=":", lw=1.4, zorder=1.5)
    # Add a tick mark + label on the right side of the axis.
    secax = ax.secondary_yaxis("right")
    secax.set_yscale("log")
    secax.set_ylim(ax.get_ylim())
    secax.yaxis.set_major_locator(FixedLocator([0.25]))
    secax.yaxis.set_major_formatter(FixedFormatter([r"$\mathbf{1/4}$"]))
    secax.tick_params(axis="y", which="major", labelsize=FS_AXES,
                      width=LW_AXES, length=8, colors=[0.25, 0.25, 0.25])
    secax.tick_params(axis="y", which="minor", length=0)
    for spine in secax.spines.values():
        spine.set_linewidth(LW_AXES)

    # --- Orbital labels at x ~ 30, sandwiched between FM and SDW curves -----
    # Each label has TWO leader lines: one to the FM (purple) curve above and
    # one to the SDW (teal) curve below for the same orbital.
    label_color = (0, 0, 0)

    def _y_at(x_arr, y_arr, x_target):
        """Return y at the data point closest to x_target."""
        i = int(np.argmin(np.abs(x_arr - x_target)))
        return float(y_arr[i])

    def _place_orbital_label(orb_idx, x_anchor, x_text, txt):
        y_fm  = _y_at(dist_all[orb_idx][0], corr_all[orb_idx][0], x_anchor)
        y_sdw = _y_at(dist_all[orb_idx][1], corr_all[orb_idx][1], x_anchor)
        # Geometric mean → midpoint in log space
        y_text = np.sqrt(y_fm * y_sdw)
        # Two leader lines, horizontally offset so they are clearly visible
        ax.plot([x_text, x_anchor], [y_text, y_fm], "-",
                color=label_color, lw=1.2, zorder=4, solid_capstyle="round")
        ax.plot([x_text, x_anchor], [y_text, y_sdw], "-",
                color=label_color, lw=1.2, zorder=4, solid_capstyle="round")
        ax.text(x_text, y_text, txt,
                color=label_color, fontname=FONT_NAME, fontsize=FS_LABEL,
                ha="left", va="center",
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=1.0, alpha=0.95),
                zorder=5)

    # anchor on the data at x≈22, label text floats out at x≈36
    _place_orbital_label(0, 22, 36, r"$d_{x^2-y^2}$")
    _place_orbital_label(1, 22, 36, r"$d_{z^2}$")

    # --- Rotate-aligned fit annotations ON the dashed fit lines ------------
    def _rotated_fit_label(orb_idx, x_lab):
        if orb_idx not in fits:
            return
        info = fits[orb_idx]
        y_lab = np.exp(info["intercept"]) * x_lab ** info["slope"]
        p1 = ax.transData.transform((x_lab, y_lab))
        p2 = ax.transData.transform(
            (x_lab * 1.5,
             np.exp(info["intercept"]) * (x_lab * 1.5) ** info["slope"]))
        angle_deg = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        ax.text(x_lab, y_lab * 1.18, rf"$\sim r^{{-{info['alpha']:.2f}}}$",
                color=series_colors[k_sdw], fontname=FONT_NAME,
                fontsize=FS_LABEL, ha="center", va="bottom",
                rotation=angle_deg, rotation_mode="anchor")

    _rotated_fit_label(1, x_lab=14)  # localized SDW (upper dashed)
    _rotated_fit_label(0, x_lab=14)  # itinerant SDW (lower dashed)

    # --- Phase-only legend (line, no marker) -------------------------------
    handles = []
    labels  = []
    for k, p in enumerate(params):
        handles.append(Line2D([], [], color=series_colors[k], lw=LW_PLOT))
        labels.append(rf"$J_H = {-p['Jk']}t,\; U = {p['U']}t$")
    ax.legend(handles, labels,
              loc="lower left", bbox_to_anchor=(0.0, 0.0),
              frameon=False, fontsize=FS_LEGEND, handlelength=2.2,
              borderpad=0.2, labelspacing=0.3)


# ---------------------------------------------------------------------------
# Chain schematic insets (Option B: arrows for spin, sites tinted by phase)
# ---------------------------------------------------------------------------
def _stair_chain_coords(n_sites: int, spacing: float = 1.0):
    """Return list of (x, y) site positions for a 45° stair zigzag.

    Site k: x = ((k+1)//2)*spacing, y = (k//2)*spacing.  Bonds connect
    consecutive sites and alternate horizontal/vertical.
    """
    return [(((k + 1) // 2) * spacing, (k // 2) * spacing)
            for k in range(n_sites)]


def draw_chain_inset(parent_ax, bbox, spins, region_color, *,
                     lattice_spacing=1.9, site_radius=0.78, arrow_len=1.20,
                     bond_lw=1.6, arrow_lw=1.6, arrow_angle_deg=110.0,
                     site_edge_lw=1.6, phase_label=None):
    """Draw a stair-zigzag chain schematic inside parent_ax.

    Parameters
    ----------
    parent_ax  : host Axes
    bbox       : [x0, y0, w, h] in axes-fraction coords of parent_ax
    spins      : sequence of +1/-1 for each site
    region_color : RGB triple — site fill = light tint of this color
    """
    from matplotlib.patches import Circle, FancyArrow

    ax = parent_ax.inset_axes(bbox)
    n = len(spins)
    coords = _stair_chain_coords(n, spacing=lattice_spacing)

    # Bonds (drawn first, lowest zorder)
    for k in range(n - 1):
        x0, y0 = coords[k]
        x1, y1 = coords[k + 1]
        ax.plot([x0, x1], [y0, y1], "-", color="black",
                lw=bond_lw, solid_capstyle="round", zorder=1)

    # Sites: white fill for arrow contrast, saturated phase color edge.
    edge_color = 0.85 * np.asarray(region_color) + 0.15 * np.zeros(3)
    for (x, y) in coords:
        ax.add_patch(Circle((x, y), site_radius,
                            facecolor="white", edgecolor=edge_color,
                            lw=site_edge_lw, zorder=2))

    # Arrows (one per site) drawn at `arrow_angle_deg` from the +x axis for
    # spin up; spin down is the antipodal direction. We use FancyArrow with
    # length_includes_head=True so the entire arrow (shaft + head) lies on
    # a segment of length `arrow_len` centered on the site, which we can
    # exactly bound by site_radius in data coordinates.
    theta = np.deg2rad(arrow_angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    half = arrow_len / 2.0
    head_len = arrow_len * 0.5   # head is 45% of total length
    head_wid = arrow_len * 0.4
    shaft_w  = arrow_len * 0.05
    for (x, y), s in zip(coords, spins):
        if s > 0:
            tx, ty = x - half * cos_t, y - half * sin_t
            dx, dy = arrow_len * cos_t, arrow_len * sin_t
        else:
            tx, ty = x + half * cos_t, y + half * sin_t
            dx, dy = -arrow_len * cos_t, -arrow_len * sin_t
        ax.add_patch(FancyArrow(
            tx, ty, dx, dy,
            width=shaft_w, head_width=head_wid, head_length=head_len,
            length_includes_head=True,
            facecolor="black", edgecolor="black", linewidth=0,
            zorder=3,
        ))

    # Frame and limits — extra top padding leaves room for the phase label;
    # widened x-padding so the diagonal arrows at the leftmost / rightmost
    # sites are not clipped.
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    pad_x = 1.0
    pad_y_bot = 1.0
    pad_y_top = 2.2
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y_bot, max(ys) + pad_y_top)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    # Phase label inside the inset (upper-left, in the padding above chain)
    if phase_label is not None:
        ax.text(0.02, 0.98, phase_label, transform=ax.transAxes,
                ha="left", va="top", fontname=FONT_NAME, fontsize=15,
                fontweight="bold", color=region_color)
    return ax


# ---------------------------------------------------------------------------
# Left panel: phase diagram
# ---------------------------------------------------------------------------
def draw_phase_panel(ax):
    # 2k_F-SDW (triangles, FM color)
    U_tri = np.concatenate([np.arange(0, 13, 2), [0, 2, 4, 6], [0, 2], [0]])
    Jh_tri = np.concatenate([np.zeros(7), 2*np.ones(4), 4*np.ones(2), [6]])
    ax.scatter(U_tri, Jh_tri, s=120, marker="^",
               facecolor=FM_COLOR, edgecolors="none", zorder=3, clip_on=False)

    # FM/(0,pi) state points (SDW color circles)
    U_opi = np.concatenate([[10, 12], [8, 10, 12], np.arange(6, 13, 2),
                            np.arange(4, 13, 2), [0], np.arange(0, 13, 2)])
    Jh_opi = np.concatenate([4*np.ones(2), 6*np.ones(3), 8*np.ones(4),
                             10*np.ones(5), [13], 15*np.ones(7)])
    ax.scatter(U_opi, Jh_opi, s=100, marker="o",
               facecolor=SDW_COLOR, edgecolors="none", zorder=3, clip_on=False)

    # Phase boundary (cubic spline through anchor points)
    x_anchor = np.array([0, 4, 8, 10, 12], dtype=float)
    y_anchor = np.array([12.3, 9, 5, 3.5, 2.5]) - 2
    order = np.argsort(y_anchor)
    cs = CubicSpline(y_anchor[order], x_anchor[order])
    y_all = np.concatenate([y_anchor, Jh_tri, Jh_opi])
    y_fine = np.linspace(y_all.min(), y_all.max(), 200)
    x_fine = cs(y_fine)

    x_max = 12.0
    # Background fills
    y0 = float(np.median(y_fine))
    x_mid = float(cs(y0))
    sdw_left = float(np.median(U_opi)) <= x_mid
    mix = 0.5
    sdw_bg = (1-mix)*SDW_COLOR + mix*np.ones(3)
    fm_bg  = (1-mix)*FM_COLOR  + mix*np.ones(3)
    if sdw_left:
        left_color, right_color = sdw_bg, fm_bg
    else:
        left_color, right_color = fm_bg, sdw_bg

    poly_left = np.column_stack([
        np.concatenate([np.zeros_like(y_fine), x_fine[::-1]]),
        np.concatenate([y_fine, y_fine[::-1]]),
    ])
    poly_right = np.column_stack([
        np.concatenate([x_fine, x_max*np.ones_like(y_fine)]),
        np.concatenate([y_fine, y_fine[::-1]]),
    ])
    ax.add_patch(Polygon(poly_left, facecolor=left_color, edgecolor="none",
                         alpha=0.25, zorder=1))
    ax.add_patch(Polygon(poly_right, facecolor=right_color, edgecolor="none",
                         alpha=0.25, zorder=1))

    ax.plot(x_fine, y_fine, "k-", lw=2.0, zorder=2)

    # FM / 2k_F-SDW text labels are placed *inside* the schematic insets
    # below; we therefore omit the in-region text labels here to avoid
    # collision with the insets.

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 15)
    ax.set_xlabel(r"$U/t$", fontname=FONT_NAME, fontsize=FS_AXES)
    ax.set_ylabel(r"$J_H/t$", fontname=FONT_NAME, fontsize=FS_AXES)
    ax.tick_params(axis="both", which="major", labelsize=FS_AXES,
                   width=LW_AXES, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(LW_AXES)

    # --- Chain schematic insets (Option B: arrows on tinted sites) ---------
    # In this panel: FM region label color = SDW_COLOR (purple);
    #                SDW region label color = FM_COLOR  (teal).
    fm_spins  = [+1] * 8
    draw_chain_inset(ax, [0.40, 0.62, 0.58, 0.34],
                     spins=fm_spins, region_color=SDW_COLOR,
                     phase_label="FM")
    sdw_spins = [+1, +1, -1, -1, +1, +1, -1, -1]
    draw_chain_inset(ax, [0.02, 0.13, 0.50, 0.30],
                     spins=sdw_spins, region_color=FM_COLOR,
                     phase_label=r"$2k_F$-SDW")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    plt.rcParams["font.family"] = FONT_NAME
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4),
                             gridspec_kw=dict(wspace=0.32))
    draw_phase_panel(axes[0])
    draw_corr_panel(axes[1])

    # Panel labels (a) and (b)
    for ax, lab in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.18, 1.02, lab, transform=ax.transAxes,
                fontname=FONT_NAME, fontsize=FS_AXES + 2,
                fontweight="bold", ha="left", va="bottom")

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    base = FIG_DIR / "kondo_1d_chain_corr_and_phase_2panel"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print("wrote", base.with_suffix(".pdf"))
    print("wrote", base.with_suffix(".png"))


if __name__ == "__main__":
    main()
