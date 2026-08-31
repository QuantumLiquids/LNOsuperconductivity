"""
Two-panel PBC spin correlation bubble map on four-chain Kondo ladder.
  (a) (pi/2, pi/2) stripe: U=14, Jk=-4, t2=0.3, Ly=4, Lx=20, D=12000, PBC
  (b) (pi, 0) columnar:    U=2,  Jk=-4, t2=0.3, Ly=4, Lx=20, D=10000, PBC
For Referee A4 response (robustness under PBC).
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_this_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
data_dir = _this_dir.parent.parent / "data"

# Two parameter sets
params = [
    {"U": 14, "D": 12000, "label": r"(a)", "title_q": r"$(\pi/2,\pi/2)$",
     "postfix": "t20.3Jk-4U14Ly4Lx20D12000_PBC.json"},
    {"U": 2,  "D": 10000, "label": r"(b)", "title_q": r"$(\pi,0)$",
     "postfix": "t20.3Jk-4U2Ly4Lx20D10000_PBC.json"},
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


def load_spin_corr(postfix):
    with open(data_dir / f"szsz{postfix}") as f:
        szsz = json.load(f)
    with open(data_dir / f"spsm{postfix}") as f:
        spsm = json.load(f)
    corr = {}
    for entry in szsz:
        (ref, tgt), val = entry
        corr[tgt] = val
    for entry in spsm:
        (ref, tgt), val = entry
        corr[tgt] = corr.get(tgt, 0) + val
    ref_raw = szsz[0][0][0]
    return {raw // 2: val for raw, val in corr.items()}, ref_raw // 2


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


# Colors
pos_color = np.array([142, 139, 254]) / 256
neg_color = np.array([232, 132, 130]) / 256

# Preload data and find global max
all_data = []
global_max = 0
for p in params:
    site_corr, ref_idx = load_spin_corr(p["postfix"])
    all_data.append((site_corr, ref_idx))
    vals = np.array(list(site_corr.values()))
    global_max = max(global_max, np.max(np.abs(vals)))

base_size = 300

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for panel_idx, (ax, p) in enumerate(zip(axes, params)):
    site_corr, ref_idx = all_data[panel_idx]

    draw_lattice(ax, Ly, Lx)

    # Reference site
    x_ref, y_ref = index_to_coord(ref_idx, Ly)
    ax.plot(x_ref, y_ref, 'k*', markersize=14, zorder=5)

    # Bubble plot
    for site, val in site_corr.items():
        if site == ref_idx:
            continue
        x, y = index_to_coord(site, Ly)
        sz = base_size * abs(val) / global_max
        c = pos_color if val >= 0 else neg_color
        ax.scatter(x, y, s=sz, c=[c], edgecolors='k', linewidths=0.5, zorder=3)

    ax.set_aspect('equal')
    ax.axis('off')

    # Panel label and parameters
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    w = xlim[1] - xlim[0]
    h = ylim[1] - ylim[0]
    ax.text(xlim[0] + 0.02 * w, ylim[1] - 0.02 * h, p["label"],
            fontsize=18, fontweight='bold', va='top')
    param_str = (r"$t' = 0.3t$" + "\n" +
                 r"$J_H = 4t$" + "\n" +
                 f"$U = {p['U']}t$" + "\n" +
                 "PBC")
    ax.text(xlim[0] + 0.12 * w, ylim[1] - 0.02 * h, param_str,
            fontsize=12, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', alpha=0.9))

# Shared bubble legend at bottom
legend_max = float(f"{global_max:.2g}")
legend_vals = [legend_max, 0.5 * legend_max, 0.1 * legend_max]
handles = []
for lv in legend_vals:
    h1 = axes[0].scatter([], [], s=base_size * lv / global_max, c=[pos_color],
                          edgecolors='k', linewidths=0.5, label=f'+{lv:.3g}')
    h2 = axes[0].scatter([], [], s=base_size * lv / global_max, c=[neg_color],
                          edgecolors='k', linewidths=0.5, label=f'-{lv:.3g}')
    handles.extend([h1, h2])

fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=11,
           title=r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$',
           title_fontsize=13, frameon=True, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])

# Save
fig_dir = _this_dir / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "kondo_ladder_spin_corr_PBC_two_panel.pdf",
            bbox_inches='tight', transparent=True)
plt.savefig(fig_dir / "kondo_ladder_spin_corr_PBC_two_panel.png",
            bbox_inches='tight', dpi=200)
print("Saved to figures/kondo_ladder_spin_corr_PBC_two_panel.{pdf,png}")
