"""
Multi-panel convergence plot of local <Sz> on the tilted zigzag lattice
for the two-layer Kondo model at Ly=4, Jperp=0.1, across several bond dimensions D.

Columns: D values (2000, 4000, 6000, 8000)
Rows: Layer 0 localized, Layer 1 localized
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Parameters ──
Ly, Lx = 4, 20
D_values = [2000, 4000, 6000, 8000]
jperp = 0.1

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'kondo_two_layer_zigzag',
                        'Ly4_Jperp0.1')


# ── Geometry ──
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


def draw_lattice(ax, Ly, Lx, y_offset=0):
    """Draw the tilted zigzag lattice bonds."""
    # Along-chain bonds (solid)
    for x in range(Lx - 1):
        for y in range(Ly):
            x1, y1 = index_to_coord(x, y)
            x2, y2 = index_to_coord(x + 1, y)
            ax.plot([x1, x2], [y1 + y_offset, y2 + y_offset],
                    'k-', lw=0.6, zorder=1, alpha=0.3)
    # Zigzag bonds (dashed)
    for x in range(Lx - 1):
        delta = 1 if x % 2 == 0 else -1
        for y in range(Ly):
            target = y + delta
            if 0 <= target < Ly:
                x1, y1 = index_to_coord(x, y)
                x2, y2 = index_to_coord(x + 1, target)
                ax.plot([x1, x2], [y1 + y_offset, y2 + y_offset],
                        'k--', lw=0.5, zorder=1, alpha=0.25)


def load_sz(filepath):
    """Load one-site <Sz> data: [[site], value] format."""
    with open(filepath) as f:
        data = json.load(f)
    return {d[0][0]: d[1] for d in data}


def find_sz_file(layer, D):
    """Find the sz_loc file for given layer and D."""
    patterns = [
        f"sz_loc{layer}Jperp{jperp}Jk-4t20.3U14Ly{Ly}Lx{Lx}D{D}_OBC.json",
        f"sz_loc{layer}Jperp{jperp}Jk-4t20.3U14Ly{Ly}Lx{Lx}D{D}.json",
    ]
    for p in patterns:
        fp = os.path.join(data_dir, p)
        if os.path.exists(fp):
            return fp
    return None


# ── Colors: purple positive, red negative ──
pos_color = np.array([142, 139, 254]) / 256
neg_color = np.array([232, 132, 130]) / 256

# ── First pass: find global max |Sz| across all panels ──
global_max_sz = 0
for D in D_values:
    for layer in [0, 1]:
        fp = find_sz_file(layer, D)
        if fp:
            sz_data = load_sz(fp)
            max_val = max(abs(v) for v in sz_data.values())
            global_max_sz = max(global_max_sz, max_val)

if global_max_sz == 0:
    global_max_sz = 1

base_marker_size = 300

# ── Plot: 4 columns (D) x 2 rows (Layer 0 loc, Layer 1 loc) ──
n_cols = len(D_values)
fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 7))

for col, D in enumerate(D_values):
    for row, layer in enumerate([0, 1]):
        ax = axes[row, col]
        fp = find_sz_file(layer, D)
        if fp is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()
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

    # Column title
    axes[0, col].set_title(f'D = {D}', fontsize=13, fontweight='bold', pad=8)

# Row labels
row_labels = ['Layer 0 — localized', 'Layer 1 — localized']
for row, label in enumerate(row_labels):
    axes[row, 0].text(-0.08, 0.5, label,
                      transform=axes[row, 0].transAxes, ha='right', va='center',
                      fontsize=11, fontweight='bold', rotation=90)

# ── Bubble legend ──
legend_ax = fig.add_axes([0.15, -0.02, 0.7, 0.035])
legend_ax.set_axis_off()
legend_ax.set_xlim(0, 1)
legend_ax.set_ylim(0, 1)

legend_max = float(f'{global_max_sz:.1g}')
legend_vals = [-legend_max, -legend_max / 2, -0.1 * legend_max,
               0.1 * legend_max, legend_max / 2, legend_max]
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

fig.suptitle(r'Ly=4 convergence: local $\langle S^z \rangle$ vs D'
             '\n' r'$J_\perp$=0.1, $J_K$=$-$4, $t^\prime$=0.3, U=14, $L_x$=20',
             fontsize=14, y=1.02)
plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])

# ── Save ──
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'local_sz_2d_Ly4_convergence.png'),
            dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'local_sz_2d_Ly4_convergence.pdf'),
            bbox_inches='tight')
print("Saved to plot/kondo_two_layer_zigzag/figures/local_sz_2d_Ly4_convergence.{png,pdf}")
