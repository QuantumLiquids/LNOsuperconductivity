"""
Two-panel PBC <Sz> bubble map for itinerant electrons on four-chain Kondo ladder.
  (a) U=14, D=12000, PBC  — stripe regime
  (b) U=2,  D=14000, PBC  — columnar regime
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_this_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
data_dir = _this_dir.parent.parent / "data"

params = [
    {"U": 14, "D": 12000, "label": "(a)",
     "file": "sz_localt20.3Jk-4U14Ly4Lx20D12000_PBC.json"},
    {"U": 2,  "D": 14000, "label": "(b)",
     "file": "sz_localt20.3Jk-4U2Ly4Lx20D14000_PBC.json"},
]

Ly, Lx = 4, 20


def index_to_coord(idx, Ly):
    xchain = idx // Ly
    yleg = idx % Ly
    base = xchain // 2
    x_phys = base + yleg
    if xchain % 2 == 0:
        y_phys = base - yleg
    else:
        y_phys = base + 1 - yleg
    return x_phys, y_phys


def draw_lattice(ax, Ly, Lx):
    for x in range(Lx - 1):
        for y in range(Ly):
            idx1 = y + Ly * x
            idx2 = y + Ly * (x + 1)
            x1, y1 = index_to_coord(idx1, Ly)
            x2, y2 = index_to_coord(idx2, Ly)
            ax.plot([x1, x2], [y1, y2], 'k-', lw=0.8, zorder=1)
        delta = 1 if (x % 2 == 0) else -1
        for y in range(Ly):
            target = y + delta
            if 0 <= target < Ly:
                idx1 = y + Ly * x
                idx2 = target + Ly * (x + 1)
                x1, y1 = index_to_coord(idx1, Ly)
                x2, y2 = index_to_coord(idx2, Ly)
                ax.plot([x1, x2], [y1, y2], 'k--', lw=0.6, zorder=1)
            else:
                wrapped = target % Ly
                idx1 = y + Ly * x
                idx2 = wrapped + Ly * (x + 1)
                x1, y1 = index_to_coord(idx1, Ly)
                x2, y2 = index_to_coord(idx2, Ly)
                ax.plot([x1, x2], [y1, y2], 'b--', lw=0.6, zorder=1, alpha=0.5)


# Colors: blue=spin up, red=spin down
up_color = np.array([142, 139, 254]) / 256
dn_color = np.array([232, 132, 130]) / 256

# Load data
all_data = []
global_max = 0
for p in params:
    with open(data_dir / p["file"]) as f:
        raw = json.load(f)
    # raw format: [[[raw_site], value], ...]
    sz = {entry[0][0] // 2: entry[1] for entry in raw}
    all_data.append(sz)
    global_max = max(global_max, max(abs(v) for v in sz.values()))

base_size = 400

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for panel_idx, (ax, p) in enumerate(zip(axes, params)):
    sz = all_data[panel_idx]
    draw_lattice(ax, Ly, Lx)

    for site, val in sz.items():
        x, y = index_to_coord(site, Ly)
        sz_size = base_size * abs(val) / global_max
        c = up_color if val >= 0 else dn_color
        ax.scatter(x, y, s=max(sz_size, 5), c=[c], edgecolors='k', linewidths=0.5, zorder=3)

    ax.set_aspect('equal')
    ax.axis('off')

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w = xlim[1] - xlim[0]
    h = ylim[1] - ylim[0]
    ax.text(xlim[0] + 0.02 * w, ylim[1] - 0.02 * h, p["label"],
            fontsize=18, fontweight='bold', va='top')
    param_str = (r"$t' = 0.3t$" + "\n" +
                 r"$J_H = 4t$" + "\n" +
                 f"$U = {p['U']}t$" + "\n" + "PBC")
    ax.text(xlim[0] + 0.12 * w, ylim[1] - 0.02 * h, param_str,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', alpha=0.9))

# Legend
legend_max = float(f"{global_max:.2g}")
legend_vals = [legend_max, 0.5 * legend_max, 0.1 * legend_max]
handles = []
for lv in legend_vals:
    h1 = axes[0].scatter([], [], s=base_size * lv / global_max, c=[up_color],
                          edgecolors='k', linewidths=0.5, label=f'+{lv:.2g}')
    h2 = axes[0].scatter([], [], s=base_size * lv / global_max, c=[dn_color],
                          edgecolors='k', linewidths=0.5, label=f'-{lv:.2g}')
    handles.extend([h1, h2])

fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=11,
           title=r'$\langle S^z_i \rangle$ (itinerant)',
           title_fontsize=13, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])

fig_dir = _this_dir / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "kondo_ladder_sz_PBC_two_panel.pdf",
            bbox_inches='tight', transparent=True)
plt.savefig(fig_dir / "kondo_ladder_sz_PBC_two_panel.png",
            bbox_inches='tight', dpi=200)
print("Saved to figures/kondo_ladder_sz_PBC_two_panel.{pdf,png}")
