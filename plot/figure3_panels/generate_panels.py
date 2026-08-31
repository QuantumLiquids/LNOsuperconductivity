"""
Generate individual bubble-plot panels for Figure 3 (resubmission).

Panels:
  1. Single-layer Ly=4: itinerant spin corr
  2. Single-layer Ly=4: localized spin corr
  3. Double-layer Ly=2 Jperp=0.1: localized intra (L0)
  4. Double-layer Ly=2 Jperp=0.1: localized intra (L1)
  5. Double-layer Ly=2 Jperp=0.1: localized interlayer
  6. Double-layer Ly=2 Jperp=0.1: itinerant intra (L0)
  7. Double-layer Ly=2 Jperp=0.1: itinerant interlayer
  8. Standalone bubble legend

All panels share a single global_max so bubble sizes are directly comparable.
All use identical pts_per_bond so lattice spacings match physically.
Place PDFs in Illustrator at 100% — NO zooming — and bubbles are consistent.

Colors match the MATLAB scripts used in Figures 3/4 of the paper:
  positive (FM): [142, 139, 254]/256  (purple)
  negative (AFM): [232, 132, 130]/256 (red)
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ═══════════════════════════════════════════════════════════════════════════════
# Global style — match MATLAB scripts
# ═══════════════════════════════════════════════════════════════════════════════
# CMYK-safe colors — matched to the existing Figure 3 in the manuscript,
# which was assembled in an Illustrator CMYK document.
# Original RGB: purple (142,139,254), red (232,132,130)
# After RGB→CMYK→RGB round-trip in Illustrator:
POS_COLOR = np.array([132, 132, 192]) / 256   # purple (FM / positive corr)
NEG_COLOR = np.array([230, 128, 128]) / 256    # red    (AFM / negative corr)

# Physical layout: inches per lattice bond in the tilted coordinate system.
# All panels use the same value so bond lengths are identical across PDFs.
INCHES_PER_BOND = 0.22

# Marker area (in points²) for the largest |correlation| = global_max.
BASE_MARKER_SIZE = 220

# Lattice bond drawing
BOND_LW = 0.8          # solid (intra-chain) bond linewidth
WEAK_BOND_LW = 0.5     # dashed (inter-chain) bond linewidth

# Reference-site star
STAR_SIZE = 10

# Padding (inches) around each panel
PAD_INCHES = 0.15

# Font
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
})

# ═══════════════════════════════════════════════════════════════════════════════
# Tilted zigzag lattice geometry (replicates TiltedZigZagLattice.m)
# ═══════════════════════════════════════════════════════════════════════════════

def index_to_coord(idx, Ly):
    """Map geometry index → (x_phys, y_phys) in tilted coordinates."""
    x_chain = idx // Ly
    y_leg = idx % Ly
    base = x_chain // 2
    x_phys = base + y_leg
    if x_chain % 2 == 0:
        y_phys = base - y_leg
    else:
        y_phys = base + 1 - y_leg
    return x_phys, y_phys


def draw_lattice(ax, Ly, Lx):
    """Draw the tilted zigzag lattice bonds."""
    # Solid bonds (intra-chain)
    for x in range(Lx - 1):
        for y in range(Ly):
            x1, y1 = index_to_coord(y + Ly * x, Ly)
            x2, y2 = index_to_coord(y + Ly * (x + 1), Ly)
            ax.plot([x1, x2], [y1, y2], 'k-', lw=BOND_LW, zorder=1)
    # Dashed bonds (inter-chain, weak hopping)
    for x in range(Lx - 1):
        delta = 1 if x % 2 == 0 else -1
        for y in range(Ly):
            target = y + delta
            if 0 <= target < Ly:
                x1, y1 = index_to_coord(y + Ly * x, Ly)
                x2, y2 = index_to_coord(target + Ly * (x + 1), Ly)
                ax.plot([x1, x2], [y1, y2], 'k--', lw=WEAK_BOND_LW, zorder=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', '..', 'data')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_corr_dict(data):
    """JSON array [[site_pair, value], ...] → {(s1, s2): value}."""
    d = {}
    for entry in data:
        key = (entry[0][0], entry[0][1])
        d[key] = entry[1]
    return d


def full_corr(szsz, spsm):
    """S·S = SzSz + S+S- (smsp is conjugate, not needed for real corr)."""
    corr = {}
    for k, v in szsz.items():
        corr[k] = v
    for k, v in spsm.items():
        corr[k] = corr.get(k, 0) + v
    return corr


# ── Single-layer loaders ──
def load_single_layer_itinerant(t2, Jk, U, Ly, Lx, D):
    """Load szsz+spsm for itinerant electrons. Returns (corr_dict, ref_geom_idx)."""
    suffix = f"t2{t2}Jk{Jk}U{U}Ly{Ly}Lx{Lx}D{D}.json"
    szsz = build_corr_dict(load_json(os.path.join(DATA_ROOT, f"szsz{suffix}")))
    spsm = build_corr_dict(load_json(os.path.join(DATA_ROOT, f"spsm{suffix}")))
    corr = full_corr(szsz, spsm)
    # Raw MPS indices: even = itinerant → geom = raw/2
    some_key = next(iter(corr))
    ref_raw = some_key[0]
    assert ref_raw % 2 == 0, "Expected even indices for itinerant"
    ref_geom = ref_raw // 2
    # Convert all keys to geometry indices
    corr_geom = {}
    for (s1, s2), v in corr.items():
        corr_geom[(s1 // 2, s2 // 2)] = v
    return corr_geom, ref_geom


def load_single_layer_localized(t2, Jk, U, Ly, Lx, D):
    """Load lszsz+lspsm for localized spins. Returns (corr_dict, ref_geom_idx)."""
    suffix = f"t2{t2}Jk{Jk}U{U}Ly{Ly}Lx{Lx}D{D}.json"
    szsz = build_corr_dict(load_json(os.path.join(DATA_ROOT, f"lszsz{suffix}")))
    spsm = build_corr_dict(load_json(os.path.join(DATA_ROOT, f"lspsm{suffix}")))
    corr = full_corr(szsz, spsm)
    some_key = next(iter(corr))
    ref_raw = some_key[0]
    if ref_raw % 2 == 1:
        # Raw localized indices (odd) → geom = (raw-1)/2
        ref_geom = (ref_raw - 1) // 2
        corr_geom = {}
        for (s1, s2), v in corr.items():
            corr_geom[((s1 - 1) // 2, (s2 - 1) // 2)] = v
    else:
        # Already geometry indices
        ref_geom = ref_raw
        corr_geom = dict(corr)
    return corr_geom, ref_geom


# ── Double-layer loaders ──
DL_DIR = os.path.join(DATA_ROOT, 'kondo_two_layer_zigzag')


def dl_file(prefix, jp, Lx, Ly, D):
    """Find double-layer data file with fallback naming."""
    patterns = [
        f"{prefix}_tilted_zigzagJk-4Jperp{jp}U14t20.3Lx{Lx}Ly{Ly}D{D}_OBC.json",
        f"{prefix}Jperp{jp}Jk-4t20.3U14Ly{Ly}Lx{Lx}D{D}.json",
    ]
    subdir = os.path.join(DL_DIR, f"Jperp{jp}")
    # Also try Ly2/ subfolder
    for d in [subdir, os.path.join(DL_DIR, f"Ly{Ly}_Jperp{jp}"), DL_DIR]:
        for p in patterns:
            fp = os.path.join(d, p)
            if os.path.exists(fp):
                return fp
    raise FileNotFoundError(f"Cannot find {prefix} for Jperp={jp}, D={D} in {subdir}")


def load_double_layer_loc(jp, Lx, Ly, D, kind):
    """kind: 'l0' (intra L0), 'l1' (intra L1), 'l01' (interlayer)."""
    szsz = build_corr_dict(load_json(dl_file(f"{kind}szsz", jp, Lx, Ly, D)))
    spsm = build_corr_dict(load_json(dl_file(f"{kind}spsm", jp, Lx, Ly, D)))
    corr = full_corr(szsz, spsm)

    # MPS indices for double-layer: 4*(y + Ly*x) + 2*layer + 1 (localized)
    some_key = next(iter(corr))
    ref_raw = some_key[0]
    if kind == 'l0':
        ref_layer = 0
    elif kind == 'l1':
        ref_layer = 1
    else:  # l01
        ref_layer = 0  # ref on layer 0

    if kind == 'l01':
        tgt_layer = 1
    elif kind == 'l0':
        tgt_layer = 0
    else:
        tgt_layer = 1

    def mps_to_geom(mps_site, layer):
        base = (mps_site - 2 * layer - 1) // 4
        return base

    ref_geom = mps_to_geom(ref_raw, ref_layer)

    corr_geom = {}
    for (s1, s2), v in corr.items():
        g1 = mps_to_geom(s1, ref_layer)
        if kind == 'l01':
            g2 = mps_to_geom(s2, tgt_layer)
        else:
            g2 = mps_to_geom(s2, tgt_layer)
        corr_geom[(g1, g2)] = v

    return corr_geom, ref_geom


def load_double_layer_elec(jp, Lx, Ly, D, kind):
    """kind: 'intra' or 'inter'."""
    prefix_szsz = f"szsz_elec_{kind}"
    prefix_spsm = f"spsm_elec_{kind}"
    szsz = build_corr_dict(load_json(dl_file(prefix_szsz, jp, Lx, Ly, D)))
    spsm = build_corr_dict(load_json(dl_file(prefix_spsm, jp, Lx, Ly, D)))
    corr = full_corr(szsz, spsm)

    some_key = next(iter(corr))
    ref_raw = some_key[0]
    ref_layer = 0

    def mps_to_geom(mps_site, layer):
        base = (mps_site - 2 * layer) // 4
        return base

    ref_geom = mps_to_geom(ref_raw, ref_layer)

    tgt_layer = 1 if kind == 'inter' else 0
    corr_geom = {}
    for (s1, s2), v in corr.items():
        g1 = mps_to_geom(s1, ref_layer)
        g2 = mps_to_geom(s2, tgt_layer)
        corr_geom[(g1, g2)] = v

    return corr_geom, ref_geom


# ═══════════════════════════════════════════════════════════════════════════════
# Panel drawing
# ═══════════════════════════════════════════════════════════════════════════════

def compute_extent(Ly, Lx):
    """Compute (xmin, xmax, ymin, ymax) of tilted lattice in physical coords."""
    xs, ys = [], []
    for idx in range(Ly * Lx):
        x, y = index_to_coord(idx, Ly)
        xs.append(x)
        ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)


def draw_panel(corr_geom, ref_geom, Ly, Lx, global_max, label=None,
               is_interlayer=False):
    """
    Draw one bubble panel. Returns (fig, ax).
    Figure size is set so that lattice bonds have INCHES_PER_BOND physical size.
    """
    xmin, xmax, ymin, ymax = compute_extent(Ly, Lx)
    margin = 1.0  # lattice units of margin around the plot
    w_lattice = (xmax - xmin) + 2 * margin
    h_lattice = (ymax - ymin) + 2 * margin
    fig_w = w_lattice * INCHES_PER_BOND
    fig_h = h_lattice * INCHES_PER_BOND

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect('equal')
    ax.set_axis_off()

    # Draw lattice
    draw_lattice(ax, Ly, Lx)

    # Draw bubbles
    for (s1, s2), val in corr_geom.items():
        if s1 != ref_geom:
            continue
        if not is_interlayer and s2 == ref_geom:
            continue  # skip self-correlation for intralayer
        x, y = index_to_coord(s2, Ly)
        color = POS_COLOR if val >= 0 else NEG_COLOR
        size = BASE_MARKER_SIZE * abs(val) / global_max
        if size < 0.5:
            continue
        ax.scatter(x, y, s=size, c=[color],
                   edgecolors='k', linewidths=0.4, zorder=3)

    # Reference site star
    xr, yr = index_to_coord(ref_geom, Ly)
    ax.plot(xr, yr, 'k*', markersize=STAR_SIZE, zorder=5)

    # Panel label
    if label:
        ax.text(xmin - margin + 0.3, ymax + margin - 0.3, label,
                fontsize=12, fontweight='bold', va='top', ha='left')

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def draw_legend(global_max):
    """Draw a standalone bubble legend matching the shared scale."""
    legend_max_abs = float(f'{global_max:.1g}')
    if legend_max_abs == 0:
        legend_max_abs = 1
    legend_values = [v * legend_max_abs for v in [-1, -0.5, -0.1, 0.1, 0.5, 1]]

    fig, ax = plt.subplots(1, 1, figsize=(5, 0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    n = len(legend_values)
    dx = 0.12
    x0 = 0.5 - 0.5 * (n - 1) * dx

    for k, vv in enumerate(legend_values):
        xpos = x0 + k * dx
        color = POS_COLOR if vv >= 0 else NEG_COLOR
        sz = BASE_MARKER_SIZE * abs(vv) / global_max
        ax.scatter(xpos, 0.65, s=sz, c=[color],
                   edgecolors='k', linewidths=0.4, zorder=3)
        ax.text(xpos, 0.15, f'{vv:.2g}', ha='center', va='top', fontsize=9)

    ax.text(x0 - 0.06, 0.55,
            r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$',
            ha='right', va='center', fontsize=11)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def draw_legend_vertical(global_max):
    """Draw a standalone vertical bubble legend matching the shared scale."""
    legend_max_abs = float(f'{global_max:.1g}')
    if legend_max_abs == 0:
        legend_max_abs = 1
    legend_values = [v * legend_max_abs for v in [1, 0.5, 0.1, -0.1, -0.5, -1]]

    fig, ax = plt.subplots(1, 1, figsize=(1.2, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    n = len(legend_values)
    dy = 0.12
    y0 = 0.5 + 0.5 * (n - 1) * dy  # start from top

    for k, vv in enumerate(legend_values):
        ypos = y0 - k * dy
        color = POS_COLOR if vv >= 0 else NEG_COLOR
        sz = BASE_MARKER_SIZE * abs(vv) / global_max
        ax.scatter(0.35, ypos, s=sz, c=[color],
                   edgecolors='k', linewidths=0.4, zorder=3)
        ax.text(0.65, ypos, f'{vv:.2g}', ha='left', va='center', fontsize=9)

    ax.text(0.35, y0 + 0.10,
            r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$',
            ha='center', va='bottom', fontsize=11)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)

    # ── Parameters ──
    # Single-layer: same as current Figure 3
    sl_t2, sl_Jk, sl_U, sl_Ly, sl_Lx, sl_D = 0.3, -4, 14, 4, 20, 18000
    # Double-layer
    dl_jp, dl_Lx, dl_Ly, dl_D = 0.1, 20, 2, 20000

    print("Loading all datasets...")

    # 1. Single-layer itinerant
    sl_itin, sl_itin_ref = load_single_layer_itinerant(
        sl_t2, sl_Jk, sl_U, sl_Ly, sl_Lx, sl_D)

    # 2. Single-layer localized
    sl_loc, sl_loc_ref = load_single_layer_localized(
        sl_t2, sl_Jk, sl_U, sl_Ly, sl_Lx, sl_D)

    # 3-5. Double-layer localized (L0, L1, interlayer)
    dl_loc_l0, dl_loc_l0_ref = load_double_layer_loc(dl_jp, dl_Lx, dl_Ly, dl_D, 'l0')
    dl_loc_l1, dl_loc_l1_ref = load_double_layer_loc(dl_jp, dl_Lx, dl_Ly, dl_D, 'l1')
    dl_loc_inter, dl_loc_inter_ref = load_double_layer_loc(dl_jp, dl_Lx, dl_Ly, dl_D, 'l01')

    # 6-8. Double-layer itinerant (L0 intra, L1 intra = L0 by symmetry, interlayer)
    dl_elec_intra, dl_elec_intra_ref = load_double_layer_elec(dl_jp, dl_Lx, dl_Ly, dl_D, 'intra')
    # L1 intra = L0 intra by interlayer symmetry (verified numerically for loc spins)
    dl_elec_intra_l1, dl_elec_intra_l1_ref = dl_elec_intra, dl_elec_intra_ref
    dl_elec_inter, dl_elec_inter_ref = load_double_layer_elec(dl_jp, dl_Lx, dl_Ly, dl_D, 'inter')

    # ── Compute GLOBAL max across all datasets ──
    all_datasets = [
        (sl_itin, sl_itin_ref, False),
        (sl_loc, sl_loc_ref, False),
        (dl_loc_l0, dl_loc_l0_ref, False),
        (dl_loc_l1, dl_loc_l1_ref, False),
        (dl_loc_inter, dl_loc_inter_ref, True),
        (dl_elec_intra, dl_elec_intra_ref, False),
        (dl_elec_inter, dl_elec_inter_ref, True),
    ]

    global_max = 0
    for corr, ref, is_inter in all_datasets:
        for (s1, s2), val in corr.items():
            if s1 == ref:
                if is_inter or s2 != ref:
                    global_max = max(global_max, abs(val))

    print(f"Global max |S·S| = {global_max:.6f}")

    # ── Generate panels ──
    panels = [
        # (corr, ref, Ly, Lx, label, is_inter, filename)
        (sl_itin, sl_itin_ref, sl_Ly, sl_Lx,
         None, False, "sl_Ly4_itinerant"),

        (sl_loc, sl_loc_ref, sl_Ly, sl_Lx,
         None, False, "sl_Ly4_localized"),

        (dl_loc_l0, dl_loc_l0_ref, dl_Ly, dl_Lx,
         None, False, "dl_Ly2_loc_L0_intra"),

        (dl_loc_l1, dl_loc_l1_ref, dl_Ly, dl_Lx,
         None, False, "dl_Ly2_loc_L1_intra"),

        (dl_loc_inter, dl_loc_inter_ref, dl_Ly, dl_Lx,
         None, True, "dl_Ly2_loc_interlayer"),

        (dl_elec_intra, dl_elec_intra_ref, dl_Ly, dl_Lx,
         None, False, "dl_Ly2_elec_L0_intra"),

        (dl_elec_intra_l1, dl_elec_intra_l1_ref, dl_Ly, dl_Lx,
         None, False, "dl_Ly2_elec_L1_intra"),

        (dl_elec_inter, dl_elec_inter_ref, dl_Ly, dl_Lx,
         None, True, "dl_Ly2_elec_interlayer"),
    ]

    for corr, ref, Ly, Lx, label, is_inter, fname in panels:
        fig = draw_panel(corr, ref, Ly, Lx, global_max,
                         label=label, is_interlayer=is_inter)
        for ext in ('pdf', 'eps'):
            path = os.path.join(out_dir, f"{fname}.{ext}")
            fig.savefig(path, bbox_inches='tight', pad_inches=PAD_INCHES)
        plt.close(fig)
        print(f"  Saved {fname}.{{pdf,eps}}")

    # ── Legends ──
    fig_leg = draw_legend(global_max)
    for ext in ('pdf', 'eps'):
        fig_leg.savefig(os.path.join(out_dir, f"legend.{ext}"),
                        bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_leg)
    print(f"  Saved legend.{{pdf,eps}}")

    fig_leg_v = draw_legend_vertical(global_max)
    for ext in ('pdf', 'eps'):
        fig_leg_v.savefig(os.path.join(out_dir, f"legend_vertical.{ext}"),
                          bbox_inches='tight', pad_inches=0.05)
    plt.close(fig_leg_v)
    print(f"  Saved legend_vertical.{{pdf,eps}}")

    # ── Summary ──
    print(f"\nAll panels saved to {out_dir}/")
    print(f"Global max = {global_max:.6f}")
    print(f"INCHES_PER_BOND = {INCHES_PER_BOND}")
    print(f"BASE_MARKER_SIZE = {BASE_MARKER_SIZE}")
    print("\nIllustrator instructions:")
    print("  1. Place all PDFs at 100% scale (no zoom/resize)")
    print("  2. Bubbles are directly comparable across all panels")
    print("  3. Add (a), (b), ... labels in Illustrator")
    print("  4. Use legend.pdf as the shared legend bar")


def generate_figure4():
    """Generate Figure 4 panels: (π,0) columnar SSO, U=2t, Ly=4."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'output_fig4')
    os.makedirs(out_dir, exist_ok=True)

    # Parameters: same lattice as Fig 3 single-layer but U=2t, D=20000
    t2, Jk, U, Ly, Lx, D = 0.3, -4, 2, 4, 20, 20000

    print("\n=== Figure 4 panels ===")
    print(f"Parameters: Lx={Lx}, Ly={Ly}, t'={t2}t, J_H={-Jk}t, U={U}t, D={D}")
    print("Loading datasets...")

    fig4_itin, fig4_itin_ref = load_single_layer_itinerant(t2, Jk, U, Ly, Lx, D)
    fig4_loc, fig4_loc_ref = load_single_layer_localized(t2, Jk, U, Ly, Lx, D)

    # Global max across both datasets
    global_max = 0
    for corr, ref in [(fig4_itin, fig4_itin_ref), (fig4_loc, fig4_loc_ref)]:
        for (s1, s2), val in corr.items():
            if s1 == ref and s2 != ref:
                global_max = max(global_max, abs(val))
    print(f"Global max |S·S| = {global_max:.6f}")

    panels = [
        (fig4_itin, fig4_itin_ref, "fig4_itinerant"),
        (fig4_loc, fig4_loc_ref, "fig4_localized"),
    ]

    for corr, ref, fname in panels:
        fig = draw_panel(corr, ref, Ly, Lx, global_max)
        for ext in ('pdf', 'eps'):
            fig.savefig(os.path.join(out_dir, f"{fname}.{ext}"),
                        bbox_inches='tight', pad_inches=PAD_INCHES)
        plt.close(fig)
        print(f"  Saved {fname}.{{pdf,eps}}")

    # Legends (horizontal + vertical)
    for name, draw_fn in [("legend", draw_legend),
                           ("legend_vertical", draw_legend_vertical)]:
        fig_leg = draw_fn(global_max)
        for ext in ('pdf', 'eps'):
            fig_leg.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                            bbox_inches='tight', pad_inches=0.05)
        plt.close(fig_leg)
        print(f"  Saved {name}.{{pdf,eps}}")

    # README
    with open(os.path.join(out_dir, 'README.txt'), 'w') as f:
        f.write(f"""Figure 4 panels — (pi,0) columnar SSO spin correlations
=======================================================

Parameters: Lx={Lx}, Ly={Ly}, t'={t2}t, J_H={-Jk}t, U={U}t, D={D}, OBC

Global bubble scale:
  global_max |S·S| = {global_max:.6f}
  BASE_MARKER_SIZE = {BASE_MARKER_SIZE} pt²
  INCHES_PER_BOND  = {INCHES_PER_BOND}

Colors (same as all other figures):
  Positive (FM):  RGB = (142, 139, 254)/256  purple
  Negative (AFM): RGB = (232, 132, 130)/256  red

Files:
  fig4_itinerant      — itinerant d_{{x²-y²}} spin correlation
  fig4_localized      — localized d_{{z²}} spin correlation
  legend              — horizontal bubble legend
  legend_vertical     — vertical bubble legend

Illustrator: place at 100% scale, no zoom/resize.
""")
    print(f"  Saved README.txt")
    print(f"\nFigure 4 panels saved to {out_dir}/")


if __name__ == '__main__':
    main()
    generate_figure4()
