"""
2D visualization of local <Sz> on the tilted zigzag lattice for two-layer Kondo model.
Ly=2: plots both layers (localized + itinerant) for all Jperp values at highest available D.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Geometry ──
Ly, Lx = 2, 20


def index_to_coord(x_chain, y_leg):
    """Match TiltedZigZagLattice.m indexToCoord for tilted zigzag geometry."""
    base = x_chain // 2
    x_phys = base + y_leg
    if x_chain % 2 == 0:
        y_phys = base - y_leg
    else:
        y_phys = base + 1 - y_leg
    return x_phys, y_phys


def loc_site_to_xy(mps_site, layer):
    """Convert localized-spin MPS site index to (x_chain, y_leg).
    loc_site(x, y, layer) = 4*(y + Ly*x) + 2*layer + 1
    """
    base = (mps_site - 2 * layer - 1) // 4
    y_leg = base % Ly
    x_chain = base // Ly
    return x_chain, y_leg


def elec_site_to_xy(mps_site):
    """Convert electron MPS site index to (x_chain, y_leg, layer).
    elec_site(x, y, layer) = 4*(y + Ly*x) + 2*layer
    """
    layer = (mps_site % 4) // 2
    base = mps_site // 4
    y_leg = base % Ly
    x_chain = base // Ly
    return x_chain, y_leg, layer


def draw_lattice(ax, Ly, Lx, y_offset=0):
    """Draw the tilted zigzag lattice bonds."""
    for x in range(Lx - 1):
        for y in range(Ly):
            x1, y1 = index_to_coord(x, y)
            x2, y2 = index_to_coord(x + 1, y)
            ax.plot([x1, x2], [y1 + y_offset, y2 + y_offset],
                    'k-', lw=1.0, zorder=1)
    for x in range(Lx - 1):
        delta = 1 if x % 2 == 0 else -1
        for y in range(Ly):
            target = y + delta
            if 0 <= target < Ly:
                x1, y1 = index_to_coord(x, y)
                x2, y2 = index_to_coord(x + 1, target)
                ax.plot([x1, x2], [y1 + y_offset, y2 + y_offset],
                        'k--', lw=0.8, zorder=1)


def load_sz(filepath):
    """Load one-site <Sz> data: [[site], value] format."""
    with open(filepath) as f:
        data = json.load(f)
    return {d[0][0]: d[1] for d in data}


# ── Colors matching MATLAB: purple positive, red negative ──
pos_color = np.array([142, 139, 254]) / 256
neg_color = np.array([232, 132, 130]) / 256

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'kondo_two_layer_zigzag')

jperp_values = [0.1, 0.3, 0.5]


def find_max_D(jp, layer_label):
    d_vals = []
    subdir = os.path.join(data_dir, f"Jperp{jp}")
    if not os.path.isdir(subdir):
        return None
    for f in os.listdir(subdir):
        if f.startswith(f"sz_{layer_label}") and f.endswith('.json'):
            idx = f.rfind('D')
            d_str = f[idx+1:].replace('_OBC.json', '').replace('.json', '')
            try:
                d_vals.append(int(d_str))
            except ValueError:
                pass
    return max(d_vals) if d_vals else None


def find_sz_file(jp, prefix, D):
    subdir = os.path.join(data_dir, f"Jperp{jp}")
    patterns = [
        f"{prefix}_tilted_zigzagJk-4Jperp{jp}U14t20.3Lx{Lx}Ly{Ly}D{D}_OBC.json",
        f"{prefix}Jperp{jp}Jk-4t20.3U14Ly{Ly}Lx{Lx}D{D}.json",
    ]
    for p in patterns:
        fp = os.path.join(subdir, p)
        if os.path.exists(fp):
            return fp
    return None


# ── Plot: 3 columns (Jperp) × 4 rows (layer0 loc, layer0 elec, layer1 loc, layer1 elec) ──
row_labels = ['Layer 0 — localized', 'Layer 0 — itinerant',
              'Layer 1 — localized', 'Layer 1 — itinerant']
fig, axes = plt.subplots(4, 3, figsize=(18, 14))

# First pass: find global max |Sz|
global_max_sz = 0
D_used = {}
for jp in jperp_values:
    D = find_max_D(jp, 'loc0')
    D_used[jp] = D
    if D is None:
        continue
    for prefix in ['sz_loc0', 'sz_loc1', 'sz_elec']:
        fp = find_sz_file(jp, prefix, D)
        if fp:
            sz_data = load_sz(fp)
            max_val = max(abs(v) for v in sz_data.values())
            global_max_sz = max(global_max_sz, max_val)

if global_max_sz == 0:
    global_max_sz = 1

base_marker_size = 400

for col, jp in enumerate(jperp_values):
    D = D_used[jp]
    if D is None:
        for row in range(4):
            axes[row, col].text(0.5, 0.5, 'No data', transform=axes[row, col].transAxes, ha='center')
        continue

    # Localized spins: rows 0, 2
    for layer, row in [(0, 0), (1, 2)]:
        ax = axes[row, col]
        fp = find_sz_file(jp, f'sz_loc{layer}', D)
        if fp is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue
        sz_data = load_sz(fp)
        draw_lattice(ax, Ly, Lx)
        for mps_site, sz_val in sz_data.items():
            x_chain, y_leg = loc_site_to_xy(mps_site, layer)
            x_phys, y_phys = index_to_coord(x_chain, y_leg)
            color = pos_color if sz_val >= 0 else neg_color
            size = base_marker_size * abs(sz_val) / global_max_sz
            ax.scatter(x_phys, y_phys, s=size, c=[color],
                       edgecolors='k', linewidths=0.5, zorder=3)
        ax.set_aspect('equal')
        ax.set_axis_off()

    # Itinerant electrons: rows 1, 3
    fp_elec = find_sz_file(jp, 'sz_elec', D)
    if fp_elec:
        elec_data = load_sz(fp_elec)
        elec_by_layer = {0: {}, 1: {}}
        for mps_site, sz_val in elec_data.items():
            x_chain, y_leg, layer = elec_site_to_xy(mps_site)
            elec_by_layer[layer][(x_chain, y_leg)] = sz_val

        for layer, row in [(0, 1), (1, 3)]:
            ax = axes[row, col]
            draw_lattice(ax, Ly, Lx)
            for (x_chain, y_leg), sz_val in elec_by_layer[layer].items():
                x_phys, y_phys = index_to_coord(x_chain, y_leg)
                color = pos_color if sz_val >= 0 else neg_color
                size = base_marker_size * abs(sz_val) / global_max_sz
                ax.scatter(x_phys, y_phys, s=size, c=[color],
                           edgecolors='k', linewidths=0.5, zorder=3)
            ax.set_aspect('equal')
            ax.set_axis_off()
    else:
        for row in [1, 3]:
            axes[row, col].text(0.5, 0.5, 'No elec data',
                                transform=axes[row, col].transAxes, ha='center')

    # Column title
    axes[0, col].set_title(f'$J_\\perp = {jp}$, D={D}', fontsize=13, fontweight='bold')

# Row labels
for row, label in enumerate(row_labels):
    axes[row, 0].text(-0.05, 0.5, label,
                      transform=axes[row, 0].transAxes, ha='right', va='center',
                      fontsize=10, fontweight='bold', rotation=90)

# ── Bubble legend ──
legend_ax = fig.add_axes([0.15, -0.01, 0.7, 0.03])
legend_ax.set_axis_off()
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)

legend_max = float(f'{global_max_sz:.1g}')
legend_vals = [-legend_max, -legend_max/2, -0.1*legend_max,
               0.1*legend_max, legend_max/2, legend_max]
n = len(legend_vals)
dx = 0.12
x0 = 0.5 - 0.5 * (n - 1) * dx

for k, vv in enumerate(legend_vals):
    xpos = x0 + k * dx
    color = pos_color if vv >= 0 else neg_color
    sz = base_marker_size * abs(vv) / global_max_sz
    legend_ax.scatter(xpos, 0.7, s=sz, c=[color], edgecolors='k', linewidths=0.5)
    legend_ax.text(xpos, 0.15, f'{vv:.2g}', ha='center', va='top', fontsize=9)

legend_ax.text(x0 - 0.08, 0.5, r'$\langle S^z \rangle$',
               ha='right', va='center', fontsize=11)

fig.suptitle(r'Local $\langle S^z \rangle$ — two-layer Kondo on tilted zigzag lattice'
             '\n' r't=1, t$^\prime$=0.3, $J_K$=−4, U=14, $L_x$=20, $L_y$=2',
             fontsize=13, y=1.01)
plt.tight_layout(rect=[0, 0.02, 1, 0.97])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'local_sz_2d_Ly2_all_jperp.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'local_sz_2d_Ly2_all_jperp.pdf'), bbox_inches='tight')
print("Saved to plot/kondo_two_layer_zigzag/figures/local_sz_2d_Ly2_all_jperp.{png,pdf}")
