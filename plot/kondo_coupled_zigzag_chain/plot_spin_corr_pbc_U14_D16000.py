"""
Spin correlation bubble map for U=14, D=16000, PBC (cylindrical BC).
Plots SzSz, S+S-, and total S·S = SzSz + 0.5*(S+S- + S-S+) separately.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_this_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
data_dir = _this_dir.parent.parent / "data"

postfix = "t20.3Jk-4U14Ly4Lx20D16000_PBC.json"
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


up_color = np.array([142, 139, 254]) / 256
dn_color = np.array([232, 132, 130]) / 256

# Load data
with open(data_dir / f"szsz{postfix}") as f:
    szsz_raw = json.load(f)
with open(data_dir / f"spsm{postfix}") as f:
    spsm_raw = json.load(f)
with open(data_dir / f"smsp{postfix}") as f:
    smsp_raw = json.load(f)

szsz = {e[0][1] // 2: e[1] for e in szsz_raw}
spsm = {e[0][1] // 2: e[1] for e in spsm_raw}
smsp = {e[0][1] // 2: e[1] for e in smsp_raw}
total = {k: szsz[k] + 0.5 * (spsm.get(k, 0) + smsp.get(k, 0)) for k in szsz}

ref_idx = szsz_raw[0][0][0] // 2

panels = [
    ("SzSz", szsz),
    ("S+S-", spsm),
    ("S·S (total)", total),
]

fig, axes = plt.subplots(1, 3, figsize=(24, 6))

for ax, (title, corr) in zip(axes, panels):
    draw_lattice(ax, Ly, Lx)
    corr_max = max(abs(v) for v in corr.values())
    base_size = 300

    x_ref, y_ref = index_to_coord(ref_idx, Ly)
    ax.plot(x_ref, y_ref, 'k*', markersize=14, zorder=5)

    for site, val in corr.items():
        x, y = index_to_coord(site, Ly)
        sz_size = base_size * abs(val) / corr_max
        c = up_color if val >= 0 else dn_color
        ax.scatter(x, y, s=max(sz_size, 3), c=[c], edgecolors='k',
                   linewidths=0.5, zorder=3)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"{title}\nU=14, D=16000, PBC", fontsize=14, fontweight='bold')

    # Legend
    legend_max = float(f"{corr_max:.2g}")
    handles = []
    for lv in [legend_max, 0.5 * legend_max, 0.1 * legend_max]:
        h1 = ax.scatter([], [], s=base_size * lv / corr_max, c=[up_color],
                        edgecolors='k', linewidths=0.5, label=f'+{lv:.2g}')
        h2 = ax.scatter([], [], s=base_size * lv / corr_max, c=[dn_color],
                        edgecolors='k', linewidths=0.5, label=f'-{lv:.2g}')
        handles.extend([h1, h2])
    ax.legend(handles=handles, loc='lower right', fontsize=8, ncol=2,
              title=title, title_fontsize=10, frameon=True)

plt.tight_layout()

fig_dir = _this_dir / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "spin_corr_pbc_U14_D16000_3panel.pdf",
            bbox_inches='tight', transparent=True)
plt.savefig(fig_dir / "spin_corr_pbc_U14_D16000_3panel.png",
            bbox_inches='tight', dpi=200)
print("Saved 3-panel spin correlation figure.")
