"""
Plot spin correlation bubble map for PBC data on four-chain Kondo ladder.
This is the (pi,0) state: U=2, Jk=-4, t2=0.3, Ly=4, Lx=20, D=10000, PBC.
For Referee A4 response (robustness under PBC).
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Parameters
Ly, Lx = 4, 20
_this_dir = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
data_dir = _this_dir.parent.parent / "data"
postfix = "t20.3Jk-4U2Ly4Lx20D10000_PBC.json"

# Load correlation data
with open(data_dir / f"szsz{postfix}") as f:
    szsz = json.load(f)
with open(data_dir / f"spsm{postfix}") as f:
    spsm = json.load(f)

# Build total spin correlation: <S·S> = <SzSz> + <S+S->
# (smsp ≈ spsm by symmetry, so <S·S> ≈ SzSz + 2*Re(S+S-) but here we use SzSz + S+S-)
corr = {}
for entry in szsz:
    (ref, tgt), val = entry
    corr[tgt] = val
for entry in spsm:
    (ref, tgt), val = entry
    corr[tgt] = corr.get(tgt, 0) + val

ref_raw = szsz[0][0][0]  # raw reference site index (40)
ref_idx = ref_raw // 2    # itinerant electron index (20)

# Convert raw Kondo indices to itinerant electron indices
site_corr = {raw // 2: val for raw, val in corr.items()}


def index_to_coord(idx, Ly):
    """Map itinerant electron index to physical (x, y) on tilted zigzag lattice."""
    xchain = idx // Ly
    yleg = idx % Ly
    base = xchain // 2
    x_phys = base + yleg
    if xchain % 2 == 0:
        y_phys = base - yleg
    else:
        y_phys = base + 1 - yleg
    return x_phys, y_phys


# Build coordinate arrays
sites = sorted(site_corr.keys())
xs, ys, vals = [], [], []
for s in sites:
    x, y = index_to_coord(s, Ly)
    xs.append(x)
    ys.append(y)
    vals.append(site_corr[s])

xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

# Reference site coordinates
x_ref, y_ref = index_to_coord(ref_idx, Ly)

# Colors
pos_color = np.array([142, 139, 254]) / 256
neg_color = np.array([232, 132, 130]) / 256

# Plot
fig, ax = plt.subplots(figsize=(12, 5))

# Draw lattice bonds
for x in range(Lx - 1):
    for y in range(Ly):
        idx1 = y + Ly * x
        idx2 = y + Ly * (x + 1)
        x1, y1 = index_to_coord(idx1, Ly)
        x2, y2 = index_to_coord(idx2, Ly)
        ax.plot([x1, x2], [y1, y2], 'k-', lw=1.0, zorder=1)

    # Diagonal bonds (t' bonds)
    delta = 1 if (x % 2 == 0) else -1
    for y in range(Ly):
        target = y + delta
        if 0 <= target < Ly:
            idx1 = y + Ly * x
            idx2 = target + Ly * (x + 1)
            x1, y1 = index_to_coord(idx1, Ly)
            x2, y2 = index_to_coord(idx2, Ly)
            ax.plot([x1, x2], [y1, y2], 'k--', lw=0.8, zorder=1)
        else:
            # PBC wrap
            wrapped = target % Ly
            idx1 = y + Ly * x
            idx2 = wrapped + Ly * (x + 1)
            x1, y1 = index_to_coord(idx1, Ly)
            x2, y2 = index_to_coord(idx2, Ly)
            ax.plot([x1, x2], [y1, y2], 'b--', lw=0.8, zorder=1)

# Bubble plot
max_abs = np.max(np.abs(vals))
base_size = 300

for i in range(len(xs)):
    sz = base_size * abs(vals[i]) / max_abs
    c = pos_color if vals[i] >= 0 else neg_color
    ax.scatter(xs[i], ys[i], s=sz, c=[c], edgecolors='k', linewidths=0.5, zorder=3)

# Reference site
ax.plot(x_ref, y_ref, 'k*', markersize=14, zorder=4)

# Annotation
ax.text(0.02, 0.95, r"$t' = 0.3t$" + "\n" + r"$J_H = 4t$" + "\n" + r"$U = 2t$" + "\nPBC",
        transform=ax.transAxes, fontsize=13, fontweight='bold',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='k'))

# Legend bubbles
legend_vals = [max_abs, 0.5 * max_abs, 0.1 * max_abs]
legend_labels = [f'{max_abs:.3f}', f'{0.5*max_abs:.3f}', f'{0.1*max_abs:.3f}']
for i, (lv, ll) in enumerate(zip(legend_vals, legend_labels)):
    lx_pos = 0.75 + i * 0.08
    ax.scatter([], [], s=base_size * lv / max_abs, c=[pos_color], edgecolors='k',
               linewidths=0.5, label=f'+{ll}')
    ax.scatter([], [], s=base_size * lv / max_abs, c=[neg_color], edgecolors='k',
               linewidths=0.5, label=f'-{ll}')

ax.legend(loc='lower right', fontsize=9, ncol=2, title=r'$\langle \mathbf{S}_i \cdot \mathbf{S}_j \rangle$')

ax.set_aspect('equal')
ax.axis('off')
ax.set_title(r'Spin correlation (PBC, $L_y=4$, $L_x=20$, $(\pi,0)$ regime)', fontsize=14)

fig_dir = _this_dir / "figures"
fig_dir.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(fig_dir / "kondo_ladder_spin_corr_pbc_pi0_u2_ly4_lx20.pdf",
            bbox_inches='tight', transparent=True)
plt.savefig(fig_dir / "kondo_ladder_spin_corr_pbc_pi0_u2_ly4_lx20.png",
            bbox_inches='tight', dpi=200)
plt.show()
print("Saved to figures/")
