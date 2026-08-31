"""
Plot localized spin-spin correlations <Sz_ref Sz_j> for the two-layer zigzag
Kondo lattice at different Jperp values.

Compares with the single-layer Jperp=0 baseline to check whether the
(pi/2, pi/2) stripe pattern is preserved by finite interlayer coupling.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Parameters ──
Ly = 2
Lx = 20
ref_x = Lx // 4  # = 5
ref_y = 0

data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'kondo_two_layer_zigzag')
baseline_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

# ── Site index → (x, y) for two-layer code ──
def twolayer_loc_site_to_xy(site, layer=0):
    """Inverse of loc_site(x, y, layer) = 4*(y + Ly*x) + 2*layer + 1"""
    base = (site - 2 * layer - 1) // 4
    y = base % Ly
    x = base // Ly
    return x, y

def singlelayer_site_to_xy(site):
    """Inverse of 2*(y + Ly*x) + offset; even sites are electrons."""
    base = site // 2
    y = base % Ly
    x = base // Ly
    return x, y

def load_corr(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data  # list of [[s1, s2], value]


# ── Load two-layer data ──
jperp_values = [0.1, 0.3, 0.5]
D_target = 6000  # highest D available for all runs

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for jp in jperp_values:
    # Layer-0 localized SzSz
    # Try both naming conventions (with/without Geometry prefix)
    for prefix in ["", "OBC"]:
        fname = f"l0szsz{prefix}Jperp{jp}Jk-4t20.3U14Ly2Lx20D{D_target}.json"
        fpath = os.path.join(data_dir, f"Jperp{jp}", fname)
        if os.path.exists(fpath):
            break
    if not os.path.exists(fpath):
        print(f"Missing: {fpath}")
        continue

    corr = load_corr(fpath)

    # Separate by target chain (y=0 vs y=1)
    dx_y0, val_y0 = [], []
    dx_y1, val_y1 = [], []

    for (s1, s2), val in corr:
        x2, y2 = twolayer_loc_site_to_xy(s2, layer=0)
        dx = x2 - ref_x
        if y2 == 0:
            dx_y0.append(dx)
            val_y0.append(val)
        else:
            dx_y1.append(dx)
            val_y1.append(val)

    axes[0].plot(dx_y0, val_y0, 'o-', label=f'$J_\\perp={jp}$, y=0', markersize=4)
    axes[1].plot(dx_y1, val_y1, 's-', label=f'$J_\\perp={jp}$, y=1', markersize=4)

# ── Load single-layer baseline (itinerant szsz, Jperp=0) ──
baseline_file = os.path.join(baseline_dir, "szszt20.3Jk-4U14Lx20D18000.json")
if os.path.exists(baseline_file):
    corr = load_corr(baseline_file)
    dx_y0, val_y0 = [], []
    dx_y1, val_y1 = [], []
    for (s1, s2), val in corr:
        x2, y2 = singlelayer_site_to_xy(s2)
        dx = x2 - ref_x
        if y2 == 0:
            dx_y0.append(dx)
            val_y0.append(val)
        else:
            dx_y1.append(dx)
            val_y1.append(val)
    axes[0].plot(dx_y0, val_y0, 'k^--', label='$J_\\perp=0$ (1-layer, itin.)', markersize=5, alpha=0.7)
    axes[1].plot(dx_y1, val_y1, 'k^--', label='$J_\\perp=0$ (1-layer, itin.)', markersize=5, alpha=0.7)

for ax, title in zip(axes, ['Same chain (y=0)', 'Adjacent chain (y=1)']):
    ax.set_ylabel(r'$\langle S^z_{\rm ref} S^z_j \rangle$')
    ax.legend(fontsize=8)
    ax.set_title(title)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.grid(True, alpha=0.3)

axes[1].set_xlabel(r'$\Delta x$ (sites along chain)')
fig.suptitle(f'Layer-0 localized spin correlation, D={D_target}\n'
             r't=1, t$^\prime$=0.3, $J_K$=−4, U=14, $L_x$=20, $L_y$=2',
             fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'spin_corr_jperp_comparison.pdf'))
plt.savefig(os.path.join(os.path.dirname(__file__), 'figures', 'spin_corr_jperp_comparison.png'), dpi=150)
print("Saved to plot/kondo_two_layer_zigzag/figures/spin_corr_jperp_comparison.{pdf,png}")
plt.show()
