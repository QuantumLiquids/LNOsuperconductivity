"""
2D visualization of local <Sz> for Ly=4 two-layer Kondo model — pinned run.
Reads from Ly4_Jperp0.1_pinned/ with tilted_zigzag naming convention.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Geometry ──
Ly, Lx = 4, 20
jp = 0.1


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


def elec_site_to_xy(mps_site):
    layer = (mps_site % 4) // 2
    base = mps_site // 4
    y_leg = base % Ly
    x_chain = base // Ly
    return x_chain, y_leg, layer


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


def load_sz(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return {d[0][0]: d[1] for d in data}


# ── Data directory ──
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', '..', 'data', 'kondo_two_layer_zigzag', 'Ly4_Jperp0.1_pinned')

# ── Find available D ──
d_vals = set()
for f in os.listdir(data_dir):
    if f.startswith('sz_loc0_') and f.endswith('.json'):
        idx = f.rfind('D')
        d_str = f[idx+1:].split('_')[0].replace('.json', '')
        try:
            d_vals.add(int(d_str))
        except ValueError:
            pass
available_D = sorted(d_vals)
print(f"Available D values: {available_D}")
D = max(available_D)
print(f"Plotting D = {D}")


def find_sz_file(prefix, D):
    """New naming: prefix_tilted_zigzagJk-4Jperp0.1U14t20.3Lx20Ly4D{D}_OBC.json"""
    patterns = [
        f"{prefix}_tilted_zigzagJk-4Jperp{jp}U14t20.3Lx{Lx}Ly{Ly}D{D}_OBC.json",
    ]
    for p in patterns:
        fp = os.path.join(data_dir, p)
        if os.path.exists(fp):
            return fp
    return None


# ── Colors ──
pos_color = np.array([142, 139, 254]) / 256
neg_color = np.array([232, 132, 130]) / 256

# ── Plot: 2 rows (layer 0, layer 1) × 2 columns (localized, itinerant) ──
fig, axes = plt.subplots(2, 2, figsize=(22, 12))

global_max_sz = 0
for prefix in ['sz_loc0', 'sz_loc1', 'sz_elec']:
    fp = find_sz_file(prefix, D)
    if fp:
        sz_data = load_sz(fp)
        max_val = max(abs(v) for v in sz_data.values())
        global_max_sz = max(global_max_sz, max_val)

if global_max_sz == 0:
    global_max_sz = 1

base_marker_size = 300

for row, (layer, prefix) in enumerate([(0, 'sz_loc0'), (1, 'sz_loc1')]):
    ax = axes[row, 0]
    fp = find_sz_file(prefix, D)
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
    ax.set_title(f'Layer {layer} — localized spins', fontsize=12, fontweight='bold')

fp_elec = find_sz_file('sz_elec', D)
if fp_elec:
    elec_data = load_sz(fp_elec)
    elec_by_layer = {0: {}, 1: {}}
    for mps_site, sz_val in elec_data.items():
        x_chain, y_leg, layer = elec_site_to_xy(mps_site)
        elec_by_layer[layer][(x_chain, y_leg)] = sz_val

    for row, layer in enumerate([0, 1]):
        ax = axes[row, 1]
        draw_lattice(ax, Ly, Lx)

        for (x_chain, y_leg), sz_val in elec_by_layer[layer].items():
            x_phys, y_phys = index_to_coord(x_chain, y_leg)
            color = pos_color if sz_val >= 0 else neg_color
            size = base_marker_size * abs(sz_val) / global_max_sz
            ax.scatter(x_phys, y_phys, s=size, c=[color],
                       edgecolors='k', linewidths=0.5, zorder=3)

        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.set_title(f'Layer {layer} — itinerant electrons', fontsize=12, fontweight='bold')
else:
    for row in range(2):
        axes[row, 1].text(0.5, 0.5, 'No data', transform=axes[row, 1].transAxes, ha='center')

# ── Bubble legend ──
legend_ax = fig.add_axes([0.15, -0.02, 0.7, 0.05])
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

fig.suptitle(rf'Local $\langle S^z \rangle$ — Ly={Ly}, $J_\perp$={jp}, D={D} (with pinning field)'
             '\n' r't=1, t$^\prime$=0.3, $J_K$=−4, U=14, $L_x$=20, PinField=−0.5',
             fontsize=13, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out_dir, exist_ok=True)
out_base = f'local_sz_2d_Ly{Ly}_Jperp{jp}_D{D}_pinned'
plt.savefig(os.path.join(out_dir, out_base + '.png'), dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(out_dir, out_base + '.pdf'), bbox_inches='tight')
print(f"Saved to figures/{out_base}.{{png,pdf}}")
