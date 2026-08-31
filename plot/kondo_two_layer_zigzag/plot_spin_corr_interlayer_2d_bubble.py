"""
2D bubble plot of interlayer localized spin-spin correlations (l01) on the
tilted zigzag lattice for the two-layer Kondo model. Ly=2.

Reference site on layer 0, target sites on layer 1.
Layout: 3 columns (Jperp=0.1, 0.3, 0.5), single row.
Full spin correlation = SzSz + S+S-.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ── Parameters ──
Ly, Lx = 2, 20
jperp_values = [0.1, 0.3, 0.5]
base_marker_size = 300

pos_color = np.array([142, 139, 254]) / 256  # purple
neg_color = np.array([232, 132, 130]) / 256   # red

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'kondo_two_layer_zigzag')


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
    base = (mps_site - 2 * layer - 1) // 4
    y_leg = base % Ly
    x_chain = base // Ly
    return x_chain, y_leg


def ref_mps_site(x_chain, y_leg, layer):
    return 4 * (y_leg + Ly * x_chain) + 2 * layer + 1


def draw_lattice(ax, Ly, Lx):
    for x in range(Lx - 1):
        for y in range(Ly):
            x1, y1 = index_to_coord(x, y)
            x2, y2 = index_to_coord(x + 1, y)
            ax.plot([x1, x2], [y1, y2], 'k-', lw=0.8, zorder=1)
    for x in range(Lx - 1):
        delta = 1 if x % 2 == 0 else -1
        for y in range(Ly):
            target = y + delta
            if 0 <= target < Ly:
                x1, y1 = index_to_coord(x, y)
                x2, y2 = index_to_coord(x + 1, target)
                ax.plot([x1, x2], [y1, y2], 'k--', lw=0.6, zorder=1)


# ── Data loading ──
def find_corr_file(subdir, op, jp, D):
    patterns = [
        f"l01{op}_tilted_zigzagJk-4Jperp{jp}U14t20.3Lx{Lx}Ly{Ly}D{D}_OBC.json",
        f"l01{op}Jperp{jp}Jk-4t20.3U14Ly{Ly}Lx{Lx}D{D}.json",
    ]
    for p in patterns:
        fp = os.path.join(subdir, p)
        if os.path.exists(fp):
            return fp
    return None


def find_max_D(jp):
    subdir = os.path.join(data_dir, f"Jperp{jp}")
    if not os.path.isdir(subdir):
        return None
    d_vals = []
    for f in os.listdir(subdir):
        if f.startswith("l01szsz") and f.endswith('.json'):
            m = re.search(r'D(\d+)(?:_OBC)?\.json$', f)
            if m:
                d_vals.append(int(m.group(1)))
    return max(d_vals) if d_vals else None


def load_corr(jp, D):
    subdir = os.path.join(data_dir, f"Jperp{jp}")
    szsz_file = find_corr_file(subdir, "szsz", jp, D)
    spsm_file = find_corr_file(subdir, "spsm", jp, D)

    if not szsz_file or not spsm_file:
        print(f"  Warning: missing l01 file for Jperp={jp}, D={D}")
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


# ── Find D and load data ──
D_used = {}
all_corr = {}
global_max = 0

ref_x = Lx // 4
ref_y = 0
ref_mps = ref_mps_site(ref_x, ref_y, 0)  # reference on layer 0

for jp in jperp_values:
    D = find_max_D(jp)
    D_used[jp] = D
    if D is None:
        continue
    corr = load_corr(jp, D)
    if corr is None:
        continue
    all_corr[jp] = corr
    for (s1, s2), val in corr.items():
        if s1 == ref_mps:
            global_max = max(global_max, abs(val))

if global_max == 0:
    global_max = 1
print(f"D values used: {D_used}")
print(f"Global max |correlation| = {global_max:.6f}")

# ── Plot: single row, 3 columns ──
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

for col, jp in enumerate(jperp_values):
    ax = axes[col]
    draw_lattice(ax, Ly, Lx)
    D = D_used[jp]

    if D is None or jp not in all_corr:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_aspect('equal')
        ax.set_axis_off()
        continue

    corr = all_corr[jp]

    # Plot bubbles: target sites are on layer 1
    for (s1, s2), val in corr.items():
        if s1 != ref_mps:
            continue
        # s2 is a layer-1 localized site
        x_chain, y_leg = loc_site_to_xy(s2, 1)
        x_phys, y_phys = index_to_coord(x_chain, y_leg)
        color = pos_color if val >= 0 else neg_color
        size = base_marker_size * abs(val) / global_max
        ax.scatter(x_phys, y_phys, s=size, c=[color],
                   edgecolors='k', linewidths=0.5, zorder=3)

    # Mark reference site (layer 0) with black star
    x_ref_phys, y_ref_phys = index_to_coord(ref_x, ref_y)
    ax.plot(x_ref_phys, y_ref_phys, 'k*', markersize=14, zorder=5)

    ax.set_aspect('equal')
    ax.set_axis_off()
    D_str = f"{D}"
    ax.set_title(f'$J_\\perp = {jp}$,  D = {D_str}',
                 fontsize=13, fontweight='bold', pad=10)

# ── Parameter annotation ──
param_str = (f"$L_x = {Lx}$\n$L_y = {Ly}$\n"
             f"$t' = 0.3t$\n$J_H = 4t$\n$U = 14t$")
fig.text(0.01, 0.95, param_str, fontsize=11, fontweight='bold',
         va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='black', linewidth=1))

# ── Bubble legend ──
legend_ax = fig.add_axes([0.15, -0.04, 0.7, 0.06])
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
                      edgecolors='k', linewidths=0.5)
    legend_ax.text(xpos, 0.1, f'{vv:.2g}', ha='center', va='top', fontsize=9)

legend_ax.text(x0 - 0.06, 0.5, r'$\langle \mathbf{S}_i^{L0} \cdot \mathbf{S}_j^{L1} \rangle$',
               ha='right', va='center', fontsize=11)

fig.suptitle(
    r'Interlayer spin correlation $\langle \mathbf{S}_i^{\mathrm{L0}} \cdot \mathbf{S}_j^{\mathrm{L1}} \rangle$'
    r' — ref on layer 0, targets on layer 1',
    fontsize=14, y=1.02)

plt.tight_layout(rect=[0.03, 0.04, 1, 0.96])

# ── Save ──
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_base = 'spin_corr_interlayer_2d_bubble_Ly2'
plt.savefig(os.path.join(out_dir, out_base + '.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, out_base + '.pdf'), bbox_inches='tight')
print(f"Saved to figures/{out_base}.{{png,pdf}}")
