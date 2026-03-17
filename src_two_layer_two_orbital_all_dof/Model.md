# Two band, two orbital model, all DOF

$$
\begin{aligned}
H & =-t_{\|} \sum_{l=1,2} \sum_{\langle i, j\rangle, \sigma} d_{l, i, \sigma}^{\dagger} d_{l, j, \sigma}+h . c .+\epsilon_d \sum_{l=1,2} \sum_i d_{l, i, \sigma}^{\dagger} d_{l, i, \sigma}-t_{\perp} \sum_i f_{1, i, \sigma}^{\dagger} f_{2, i, \sigma}+h . c .+\epsilon_f \sum_{l=1,2} \sum_i f_{l, i, \sigma}^{\dagger} f_{l, i, \sigma} \\
& +\sum_{l, i} U\left(n_{f, l, i, \uparrow} n_{f, l, i, \downarrow}+n_{d, l, i, \uparrow} n_{d, l, i, \downarrow}\right)-2 J_H \mathbf{S}_{d, l, i} \cdot \mathbf{S}_{f, l, i}
\end{aligned}
$$

## Parameter Notes

- `NumElectronsDx2Y2`: optional total number of electrons in the `d_{x^2-y^2}` orbital sector across both layers.
- `NumElectronsDz2`: optional total number of electrons in the `d_{z^2}` orbital sector across both layers.
- Each orbital sector contains `2 * Lx * Ly` spinful sites in the 1D DMRG mapping, so its maximum electron count is `4 * Lx * Ly`.
- If they are omitted, the code uses the manuscript-motivated default filling:
  `NumElectronsDx2Y2 = Lx * Ly` and `NumElectronsDz2 = 2 * Lx * Ly`.
- `InterOrbitalHybridization`: optional on-site hybridization hopping between `d_{x^2-y^2}` and `d_{z^2}` on the same physical Ni site.
- `Dx2Y2InterlayerHopping`: optional extra interlayer hopping in the `d_{x^2-y^2}` orbital.
- If the two optional hopping parameters are omitted, both default to `0.0`.
- The default direct-product initial state is deterministic:
  `d_{z^2}` starts with one electron on every site and total `S^z = 0`; for each x-slice, the top layer uses alternating spins by y and the bottom layer uses the opposite pattern. The `d_{x^2-y^2}` electrons are distributed as evenly as possible while keeping total `S^z = 0`, and every singly occupied `d_{x^2-y^2}` site is initialized parallel to the local `d_{z^2}` spin.

## Example

For `Lx = 6`, `Ly = 2`, the manuscript-motivated filling is

- `NumElectronsDx2Y2 = 12`
- `NumElectronsDz2 = 24`

If the two keys are omitted from the input JSON, these are exactly the values that will be used.
