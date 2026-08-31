"""
Two-panel plot for 4-hole-doped (Nh4) OBC Kondo ladder:
  (a) <Sz> local magnetization (itinerant electrons)
  (b) <SzSz> spin-spin correlation
Parameters: t'=0.3, Jk=-4, U=14, Ly=4, Lx=20, D=12001, OBC, Nh=4
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_this_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
data_dir = _this_dir.parent.parent / "data"
postfix = "t20.3Jk-4U14Ly4Lx20D12001_OBC_Nh4.json"

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


# Colors
up_color = np.array([142, 139, 254]) / 256
dn_color = np.array([232, 132, 130]) / 256

# Load data
with open(data_dir / f"sz_local{postfix}") as f:
    sz_raw = json.load(f)
with open(data_dir / f"szsz{postfix}") as f:
    szsz_raw = json.load(f)
with open(data_dir / f"spsm{postfix}") as f:
    spsm_raw = json.load(f)

sz = {e[0][0] // 2: e[1] for e in sz_raw}
szsz = {e[0][1] // 2: e[1] for e in szsz_raw}
spsm = {e[0][1] // 2: e[1] for e in spsm_raw}
# Total spin correlation: SzSz only (S+S- is negligible for this strongly polarized state)
corr = {k: szsz[k] for k in szsz}

ref_idx = szsz_raw[0][0][0] // 2

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel (a): <Sz>
ax = axes[0]
draw_lattice(ax, Ly, Lx)
sz_max = max(abs(v) for v in sz.values())
base_size_sz = 400
for site, val in sz.items():
    x, y = index_to_coord(site, Ly)
    sz_size = base_size_sz * abs(val) / sz_max
    c = up_color if val >= 0 else dn_color
    ax.scatter(x, y, s=max(sz_size, 5), c=[c], edgecolors='k', linewidths=0.5, zorder=3)
ax.set_aspect('equal')
ax.axis('off')
xlim, ylim = ax.get_xlim(), ax.get_ylim()
w, h = xlim[1] - xlim[0], ylim[1] - ylim[0]
ax.text(xlim[0] + 0.02 * w, ylim[1] - 0.02 * h, "(a)",
        fontsize=18, fontweight='bold', va='top')
ax.text(xlim[0] + 0.10 * w, ylim[1] - 0.02 * h,
        r"$\langle S^z_i \rangle$" + "\n" + r"$N_h = 4$" + "\n" +
        r"$t' = 0.3t$" + "\n" + r"$J_H = 4t,\ U = 14t$",
        fontsize=11, fontweight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', alpha=0.9))

# Panel (b): SzSz correlation
ax = axes[1]
draw_lattice(ax, Ly, Lx)
corr_max = max(abs(v) for v in corr.values())
base_size_corr = 300
x_ref, y_ref = index_to_coord(ref_idx, Ly)
ax.plot(x_ref, y_ref, 'k*', markersize=14, zorder=5)
for site, val in corr.items():
    x, y = index_to_coord(site, Ly)
    sz_size = base_size_corr * abs(val) / corr_max
    c = up_color if val >= 0 else dn_color
    ax.scatter(x, y, s=max(sz_size, 3), c=[c], edgecolors='k', linewidths=0.5, zorder=3)
ax.set_aspect('equal')
ax.axis('off')
xlim, ylim = ax.get_xlim(), ax.get_ylim()
w, h = xlim[1] - xlim[0], ylim[1] - ylim[0]
ax.text(xlim[0] + 0.02 * w, ylim[1] - 0.02 * h, "(b)",
        fontsize=18, fontweight='bold', va='top')
ax.text(xlim[0] + 0.10 * w, ylim[1] - 0.02 * h,
        r"$\langle S^z_i S^z_j \rangle$" + "\n" + r"$N_h = 4$" + "\n" +
        r"$t' = 0.3t$" + "\n" + r"$J_H = 4t,\ U = 14t$",
        fontsize=11, fontweight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='k', alpha=0.9))

# Legends
for ax_i, (label, max_val, bsz) in zip(axes,
        [(r'$\langle S^z_i \rangle$', sz_max, base_size_sz),
         (r'$\langle S^z_i S^z_j \rangle$', corr_max, base_size_corr)]):
    legend_max = float(f"{max_val:.2g}")
    handles = []
    for lv in [legend_max, 0.5 * legend_max, 0.1 * legend_max]:
        h1 = ax_i.scatter([], [], s=bsz * lv / max_val, c=[up_color],
                           edgecolors='k', linewidths=0.5, label=f'+{lv:.2g}')
        h2 = ax_i.scatter([], [], s=bsz * lv / max_val, c=[dn_color],
                           edgecolors='k', linewidths=0.5, label=f'-{lv:.2g}')
        handles.extend([h1, h2])
    ax_i.legend(handles=handles, loc='lower right', fontsize=9, ncol=2,
                title=label, title_fontsize=11, frameon=True)

plt.tight_layout()

fig_dir = _this_dir / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "kondo_ladder_Nh4_spin_two_panel.pdf",
            bbox_inches='tight', transparent=True)
plt.savefig(fig_dir / "kondo_ladder_Nh4_spin_two_panel.png",
            bbox_inches='tight', dpi=200)
print("Saved Nh4 figure.")
