"""
Combined bubble plot of spin-spin correlations for Jperp=0.1, Ly=2.
2 rows x 3 columns:
  Row 1 (localized f-spin): (a) L0 intra, (b) L1 intra, (c) interlayer
  Row 2 (itinerant c-spin): (d) L0 intra, (e) L1 intra (=L0 by symmetry), (f) interlayer
Full spin correlation = SzSz + S+S-.
For referee reply (Q1: effect of finite J_perp at ambient pressure).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Parameters ──
Ly, Lx = 2, 20
jp = 0.1
D = 20000
base_marker_size = 250

pos_color = np.array([142, 139, 254]) / 256  # purple
neg_color = np.array([232, 132, 130]) / 256   # red

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'kondo_two_layer_zigzag', f'Jperp{jp}')


# ── Geometry ──
def index_to_coord(x_chain, y_leg):
    base = x_chain // 2
    x_phys = base + y_leg
    if x_chain % 2 == 0:
        y_phys = base - y_leg
    else:
        y_phys = base + 1 - y_leg
    return x_phys, y_phys


def loc_site_to_xy(mps_site, layer):
    """Localized spin: site = 4*(y + Ly*x) + 2*layer + 1"""
    base = (mps_site - 2 * layer - 1) // 4
    y_leg = base % Ly
    x_chain = base // Ly
    return x_chain, y_leg


def elec_site_to_xy(mps_site, layer):
    """Electron: site = 4*(y + Ly*x) + 2*layer"""
    base = (mps_site - 2 * layer) // 4
    y_leg = base % Ly
    x_chain = base // Ly
    return x_chain, y_leg


def loc_ref_mps(x_chain, y_leg, layer):
    return 4 * (y_leg + Ly * x_chain) + 2 * layer + 1


def elec_ref_mps(x_chain, y_leg, layer):
    return 4 * (y_leg + Ly * x_chain) + 2 * layer


def draw_lattice(ax):
    for x in range(Lx - 1):
        for y in range(Ly):
            x1, y1 = index_to_coord(x, y)
            x2, y2 = index_to_coord(x + 1, y)
            ax.plot([x1, x2], [y1, y2], 'k-', lw=0.6, zorder=1)
    for x in range(Lx - 1):
        delta = 1 if x % 2 == 0 else -1
        for y in range(Ly):
            target = y + delta
            if 0 <= target < Ly:
                x1, y1 = index_to_coord(x, y)
                x2, y2 = index_to_coord(x + 1, target)
                ax.plot([x1, x2], [y1, y2], 'k--', lw=0.4, zorder=1)


# ── Data loading ──
def find_file(prefix):
    patterns = [
        f"{prefix}_tilted_zigzagJk-4Jperp{jp}U14t20.3Lx{Lx}Ly{Ly}D{D}_OBC.json",
        f"{prefix}Jperp{jp}Jk-4t20.3U14Ly{Ly}Lx{Lx}D{D}.json",
    ]
    for p in patterns:
        fp = os.path.join(data_dir, p)
        if os.path.exists(fp):
            return fp
    return None


def load_corr(prefix_szsz, prefix_spsm):
    szsz_file = find_file(prefix_szsz)
    spsm_file = find_file(prefix_spsm)
    if not szsz_file or not spsm_file:
        print(f"  Warning: missing file for {prefix_szsz}")
        return None
    with open(szsz_file) as f:
        szsz_data = json.load(f)
    with open(spsm_file) as f:
        spsm_data = json.load(f)
    corr = {}
    for entry in szsz_data:
        key = (entry[0][0], entry[0][1])
        corr[key] = entry[1]
    for entry in spsm_data:
        key = (entry[0][0], entry[0][1])
        corr[key] = corr.get(key, 0) + entry[1]
    return corr


# ── Load all data ──
ref_x = Lx // 4  # = 5
ref_y = 0

# Localized spin correlations (per-layer files)
corr_loc_l0 = load_corr("l0szsz", "l0spsm")
corr_loc_l1 = load_corr("l1szsz", "l1spsm")
corr_loc_inter = load_corr("l01szsz", "l01spsm")

# Itinerant electron correlations (new naming)
corr_elec_intra = load_corr("szsz_elec_intra", "spsm_elec_intra")
corr_elec_inter = load_corr("szsz_elec_inter", "spsm_elec_inter")

# ── Find global max across ALL panels ──
global_max = 0

# Localized
for corr, layer in [(corr_loc_l0, 0), (corr_loc_l1, 1)]:
    if corr is None:
        continue
    ref_mps = loc_ref_mps(ref_x, ref_y, layer)
    for (s1, s2), val in corr.items():
        if s1 == ref_mps and s2 != ref_mps:
            global_max = max(global_max, abs(val))

if corr_loc_inter is not None:
    ref_mps = loc_ref_mps(ref_x, ref_y, 0)
    for (s1, s2), val in corr_loc_inter.items():
        if s1 == ref_mps:
            global_max = max(global_max, abs(val))

# Electron
if corr_elec_intra is not None:
    ref_mps = elec_ref_mps(ref_x, ref_y, 0)
    for (s1, s2), val in corr_elec_intra.items():
        if s1 == ref_mps and s2 != ref_mps:
            global_max = max(global_max, abs(val))

if corr_elec_inter is not None:
    ref_mps = elec_ref_mps(ref_x, ref_y, 0)
    for (s1, s2), val in corr_elec_inter.items():
        if s1 == ref_mps:
            global_max = max(global_max, abs(val))

if global_max == 0:
    global_max = 1
print(f"Global max |correlation| = {global_max:.6f}")

# ── Plot: 2 rows × 3 columns, compact ──
fig, axes = plt.subplots(2, 3, figsize=(10, 5.5),
                         gridspec_kw={'wspace': -0.3, 'hspace': 0.25})

# Panel specifications: (corr_dict, ref_layer, target_layer, site_to_xy_func, ref_mps_func, title)
# target_layer=None means interlayer (target is on the other layer)
panels = [
    # Row 0: localized spins
    (corr_loc_l0, 0, 0, loc_site_to_xy, loc_ref_mps,
     r'(a) $f$-spin L0 intra'),
    (corr_loc_l1, 1, 1, loc_site_to_xy, loc_ref_mps,
     r'(b) $f$-spin L1 intra'),
    (corr_loc_inter, 0, 1, loc_site_to_xy, loc_ref_mps,
     r'(c) $f$-spin interlayer'),
    # Row 1: itinerant electrons
    (corr_elec_intra, 0, 0, elec_site_to_xy, elec_ref_mps,
     r'(d) $c$-spin L0 intra'),
    (corr_elec_intra, 0, 0, elec_site_to_xy, elec_ref_mps,
     r'(e) $c$-spin L1 intra (=L0)'),
    (corr_elec_inter, 0, 1, elec_site_to_xy, elec_ref_mps,
     r'(f) $c$-spin interlayer'),
]

for idx, (corr, ref_layer, tgt_layer, site_to_xy, ref_func, title) in enumerate(panels):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    draw_lattice(ax)

    if corr is None:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10)
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(title, fontsize=9, pad=4)
        continue

    ref_mps = ref_func(ref_x, ref_y, ref_layer)
    is_interlayer = (ref_layer != tgt_layer)

    for (s1, s2), val in corr.items():
        if s1 != ref_mps:
            continue
        if not is_interlayer and s2 == ref_mps:
            continue  # skip self-correlation
        x_chain, y_leg = site_to_xy(s2, tgt_layer)
        x_phys, y_phys = index_to_coord(x_chain, y_leg)
        color = pos_color if val >= 0 else neg_color
        size = base_marker_size * abs(val) / global_max
        ax.scatter(x_phys, y_phys, s=size, c=[color],
                   edgecolors='k', linewidths=0.4, zorder=3)

    # Mark reference site
    x_ref_phys, y_ref_phys = index_to_coord(ref_x, ref_y)
    ax.plot(x_ref_phys, y_ref_phys, 'k*', markersize=10, zorder=5)

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=4)

# Row labels
for row, label in enumerate([r'Localized ($d_{z^2}$)', r'Itinerant ($d_{x^2-y^2}$)']):
    axes[row, 0].text(-0.05, 0.5, label,
                      transform=axes[row, 0].transAxes,
                      ha='right', va='center',
                      fontsize=9, fontweight='bold', rotation=90)

# ── Parameter annotation ──
param_str = (f"$L_x={Lx},\\ L_y={Ly},\\ D={D}$\n"
             f"$t'=0.3t,\\ J_H=4t,\\ U=14t,\\ J_\\perp={jp}t$")
fig.text(0.02, 0.97, param_str, fontsize=8, fontweight='bold',
         va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='black', linewidth=0.8))

# ── Bubble legend ──
legend_ax = fig.add_axes([0.20, -0.04, 0.6, 0.05])
legend_ax.set_axis_off()
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)

legend_max = float(f'{global_max:.1g}')
legend_vals = [-legend_max, -legend_max / 2, -0.1 * legend_max,
               0.1 * legend_max, legend_max / 2, legend_max]
n = len(legend_vals)
dx = 0.10
x0 = 0.5 - 0.5 * (n - 1) * dx

for k, vv in enumerate(legend_vals):
    xpos = x0 + k * dx
    color = pos_color if vv >= 0 else neg_color
    sz = base_marker_size * abs(vv) / global_max
    legend_ax.scatter(xpos, 0.7, s=sz, c=[color],
                      edgecolors='k', linewidths=0.4)
    legend_ax.text(xpos, 0.05, f'{vv:.2g}', ha='center', va='top', fontsize=8)

legend_ax.text(x0 - 0.05, 0.5,
               r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$',
               ha='right', va='center', fontsize=10)

# ── Save ──
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_base = 'spin_corr_combined_jperp0.1_Ly2'
plt.savefig(os.path.join(out_dir, out_base + '.png'), dpi=200, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, out_base + '.pdf'), bbox_inches='tight')
print(f"Saved to figures/{out_base}.{{png,pdf}}")
