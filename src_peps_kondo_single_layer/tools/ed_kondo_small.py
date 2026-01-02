#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Exact diagonalization benchmarks for the single-layer Kondo lattice (OBC square).

Model (consistent with src_peps_kondo_single_layer code conventions):
  H = -t * sum_<ij>,σ ( c†_{iσ} c_{jσ} + h.c. )
      + U * sum_i n_{i↑} n_{i↓}
      + JK * sum_i s_i · S_i
      - mu * sum_i (n_{i↑}+n_{i↓})

Notes:
  - Local Hilbert space is 8D = (itinerant electron 4D) ⊗ (local spin 2D).
  - We ED only tiny clusters (2x2 is practical). For 4x4 we provide free-fermion
    energies via one-body filling (no ED).
  - We work in fixed (Ne, Sz2_total) sector, where Sz2_total = 2*Sz_total is integer.
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def site_index(x: int, y: int, Lx: int) -> int:
    return y * Lx + x


def orbital_index(site: int, spin: int) -> int:
    # spin: 0=up, 1=down
    return 2 * site + spin


def bitcount(x: int) -> int:
    return int(x.bit_count())


def apply_annihilate(occ: int, p: int) -> Optional[Tuple[int, int]]:
    """Apply c_p on |occ>, returns (new_occ, phase) or None."""
    if (occ >> p) & 1 == 0:
        return None
    mask = (1 << p) - 1
    parity = bitcount(occ & mask) & 1
    phase = -1 if parity else 1
    return occ ^ (1 << p), phase


def apply_create(occ: int, p: int) -> Optional[Tuple[int, int]]:
    """Apply c†_p on |occ>, returns (new_occ, phase) or None."""
    if (occ >> p) & 1 == 1:
        return None
    mask = (1 << p) - 1
    parity = bitcount(occ & mask) & 1
    phase = -1 if parity else 1
    return occ | (1 << p), phase


def apply_hop(occ: int, p_to: int, p_from: int) -> Optional[Tuple[int, int]]:
    """Apply c†_{to} c_{from}."""
    res1 = apply_annihilate(occ, p_from)
    if res1 is None:
        return None
    occ1, ph1 = res1
    res2 = apply_create(occ1, p_to)
    if res2 is None:
        return None
    occ2, ph2 = res2
    return occ2, ph1 * ph2


def local_spin_sz2(local_up_bits: int, site: int) -> int:
    # +1 for Up, -1 for Down
    return 1 if ((local_up_bits >> site) & 1) else -1


def flip_local_spin(local_up_bits: int, site: int) -> int:
    return local_up_bits ^ (1 << site)


def electron_n_up_dn(occ: int, site: int) -> Tuple[int, int]:
    up = (occ >> orbital_index(site, 0)) & 1
    dn = (occ >> orbital_index(site, 1)) & 1
    return int(up), int(dn)


def electron_ne(occ: int) -> int:
    return bitcount(occ)


def electron_sz2(occ: int, nsites: int) -> int:
    nup = 0
    ndn = 0
    for s in range(nsites):
        u, d = electron_n_up_dn(occ, s)
        nup += u
        ndn += d
    return nup - ndn


def total_sz2(occ: int, local_up_bits: int, nsites: int) -> int:
    sz2e = electron_sz2(occ, nsites)
    nloc_up = bitcount(local_up_bits)
    sz2l = 2 * nloc_up - nsites
    return sz2e + sz2l


def generate_electron_occs(nsites: int, Ne: int) -> List[int]:
    norb = 2 * nsites
    occs = []
    for comb in itertools.combinations(range(norb), Ne):
        occ = 0
        for p in comb:
            occ |= 1 << p
        occs.append(occ)
    return occs


@dataclass(frozen=True)
class Sector:
    Lx: int
    Ly: int
    Ne: int
    Sz2_total: int

    @property
    def nsites(self) -> int:
        return self.Lx * self.Ly


@dataclass(frozen=True)
class Params:
    t: float
    U: float
    JK: float
    mu: float


def bonds_obc_square(Lx: int, Ly: int) -> List[Tuple[int, int]]:
    bonds = []
    for y in range(Ly):
        for x in range(Lx):
            s = site_index(x, y, Lx)
            if x + 1 < Lx:
                bonds.append((s, site_index(x + 1, y, Lx)))
            if y + 1 < Ly:
                bonds.append((s, site_index(x, y + 1, Lx)))
    return bonds


def build_basis(sector: Sector) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    nsites = sector.nsites
    basis: List[Tuple[int, int]] = []

    e_occs = generate_electron_occs(nsites, sector.Ne)
    for occ in e_occs:
        for local_up_bits in range(1 << nsites):
            if total_sz2(occ, local_up_bits, nsites) != sector.Sz2_total:
                continue
            basis.append((occ, local_up_bits))

    index: Dict[Tuple[int, int], int] = {st: i for i, st in enumerate(basis)}
    return basis, index


def build_hamiltonian_dense(*_args, **_kwargs):
    # Intentionally disabled: keep this script dependency-free (no numpy/scipy).
    # Use the Lanczos path (`build_hamiltonian_rows` + `lanczos_smallest_eig`) instead.
    raise RuntimeError("Dense build disabled (no numpy in this repo environment). Use Lanczos sparse ED.")


def ground_energy_ed(Lx: int, Ly: int, Ne: int, Sz2_total: int, t: float, U: float, JK: float, mu: float) -> Tuple[float, int]:
    sector = Sector(Lx=Lx, Ly=Ly, Ne=Ne, Sz2_total=Sz2_total)
    params = Params(t=t, U=U, JK=JK, mu=mu)
    basis, index = build_basis(sector)
    if len(basis) == 0:
        raise ValueError(f"Empty sector: L={Lx}x{Ly}, Ne={Ne}, Sz2_total={Sz2_total}")
    rows = build_hamiltonian_rows(sector, params, basis, index)
    e0 = lanczos_smallest_eig(rows, max_iter=min(200, len(basis)), tol=1e-12, seed=0)
    return e0, len(basis)


def ground_state_ed_vector(
    Lx: int, Ly: int, Ne: int, Sz2_total: int, t: float, U: float, JK: float, mu: float, *,
    max_iter: int = 300, tol: float = 1e-12, seed: int = 0,
) -> Tuple[float, List[float], List[Tuple[int, int]]]:
    """Return (E0, vec, basis) for tiny clusters using Lanczos (no numpy).

    vec is normalized in the ED basis ordering returned by build_basis():
      basis[i] = (electron_occ_bits, local_up_bits)
    """
    sector = Sector(Lx=Lx, Ly=Ly, Ne=Ne, Sz2_total=Sz2_total)
    params = Params(t=t, U=U, JK=JK, mu=mu)
    basis, index = build_basis(sector)
    if len(basis) == 0:
        raise ValueError(f"Empty sector: L={Lx}x{Ly}, Ne={Ne}, Sz2_total={Sz2_total}")
    rows = build_hamiltonian_rows(sector, params, basis, index)
    e0, vec = lanczos_smallest_eigvec(rows, max_iter=min(max_iter, len(basis)), tol=tol, seed=seed)
    return e0, vec, basis


def free_fermion_one_body_levels_obc(Lx: int, Ly: int, t: float) -> List[float]:
    eps = []
    for mx in range(1, Lx + 1):
        kx = math.pi * mx / (Lx + 1)
        for my in range(1, Ly + 1):
            ky = math.pi * my / (Ly + 1)
            eps.append(-2.0 * t * (math.cos(kx) + math.cos(ky)))
    eps.sort()
    return eps


def free_fermion_ground_energy(Lx: int, Ly: int, t: float, Ne: int) -> float:
    # Spinful, noninteracting: fill lowest one-body levels with two spins each.
    eps = free_fermion_one_body_levels_obc(Lx, Ly, t)
    if Ne < 0 or Ne > 2 * len(eps):
        raise ValueError("Ne out of range for free fermion.")
    # Fill with degeneracy 2: each spatial level can host up+down.
    e = 0.0
    remaining = Ne
    for en in eps:
        if remaining <= 0:
            break
        take = 2 if remaining >= 2 else 1
        e += take * en
        remaining -= take
    return e


def build_hamiltonian_rows(
    sector: Sector,
    params: Params,
    basis: List[Tuple[int, int]],
    index: Dict[Tuple[int, int], int],
) -> List[List[Tuple[int, float]]]:
    """Return sparse rows: rows[i] = list of (j, Hij)."""
    nsites = sector.nsites
    dim = len(basis)
    rows: List[List[Tuple[int, float]]] = [[] for _ in range(dim)]
    bonds = bonds_obc_square(sector.Lx, sector.Ly)

    for i, (occ, local_up_bits) in enumerate(basis):
        acc: Dict[int, float] = {}

        # Hopping
        if params.t != 0.0:
            for (a, b) in bonds:
                for spin in (0, 1):
                    pa = orbital_index(a, spin)
                    pb = orbital_index(b, spin)
                    res = apply_hop(occ, pa, pb)
                    if res is not None:
                        occ2, ph = res
                        j = index.get((occ2, local_up_bits))
                        if j is not None:
                            acc[j] = acc.get(j, 0.0) + (-params.t * ph)
                    res = apply_hop(occ, pb, pa)
                    if res is not None:
                        occ2, ph = res
                        j = index.get((occ2, local_up_bits))
                        if j is not None:
                            acc[j] = acc.get(j, 0.0) + (-params.t * ph)

        # Onsite diagonal (U, mu, JK*s_z*S_z)
        diag = 0.0
        for s in range(nsites):
            nu, nd = electron_n_up_dn(occ, s)
            if params.U != 0.0:
                diag += params.U * (nu * nd)
            if params.mu != 0.0:
                diag += -params.mu * (nu + nd)
            if params.JK != 0.0:
                sz_e = 0.5 * (nu - nd)
                Sz_l = 0.5 * (1 if ((local_up_bits >> s) & 1) else -1)
                diag += params.JK * (sz_e * Sz_l)

        if diag != 0.0:
            acc[i] = acc.get(i, 0.0) + diag

        # Onsite off-diagonal JK * 1/2 (s_+ S_- + s_- S_+)
        if params.JK != 0.0:
            for s in range(nsites):
                p_up = orbital_index(s, 0)
                p_dn = orbital_index(s, 1)

                # s_+ S_- : local Up -> Down, electron dn -> up
                if ((local_up_bits >> s) & 1) == 1:
                    res1 = apply_annihilate(occ, p_dn)
                    if res1 is not None:
                        occ1, ph1 = res1
                        res2 = apply_create(occ1, p_up)
                        if res2 is not None:
                            occ2, ph2 = res2
                            local2 = flip_local_spin(local_up_bits, s)
                            j = index.get((occ2, local2))
                            if j is not None:
                                acc[j] = acc.get(j, 0.0) + (params.JK * 0.5 * (ph1 * ph2))

                # s_- S_+ : local Down -> Up, electron up -> dn
                if ((local_up_bits >> s) & 1) == 0:
                    res1 = apply_annihilate(occ, p_up)
                    if res1 is not None:
                        occ1, ph1 = res1
                        res2 = apply_create(occ1, p_dn)
                        if res2 is not None:
                            occ2, ph2 = res2
                            local2 = flip_local_spin(local_up_bits, s)
                            j = index.get((occ2, local2))
                            if j is not None:
                                acc[j] = acc.get(j, 0.0) + (params.JK * 0.5 * (ph1 * ph2))

        # finalize row
        rows[i] = [(j, v) for (j, v) in acc.items() if v != 0.0]

    # Symmetry sanity (optional, cheap for tiny)
    # We won't enforce symmetry explicitly; the construction is Hermitian by design.
    return rows


def dot(x: List[float], y: List[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def norm2(x: List[float]) -> float:
    return dot(x, x)


def axpy(y: List[float], a: float, x: List[float]) -> None:
    for i in range(len(y)):
        y[i] += a * x[i]


def scal(x: List[float], a: float) -> None:
    for i in range(len(x)):
        x[i] *= a


def matvec(rows: List[List[Tuple[int, float]]], x: List[float]) -> List[float]:
    y = [0.0] * len(x)
    for i, row in enumerate(rows):
        s = 0.0
        for (j, v) in row:
            s += v * x[j]
        y[i] = s
    return y


def tridiag_sturm_count(diag: List[float], off: List[float], x: float) -> int:
    """Count eigenvalues <= x for symmetric tridiagonal (diag, off)."""
    n = len(diag)
    count = 0
    p = diag[0] - x
    if p < 0.0:
        count += 1
    for i in range(1, n):
        # avoid division by 0 (standard Sturm trick)
        if abs(p) < 1e-18:
            p = -1e-18 if p < 0.0 else 1e-18
        p = (diag[i] - x) - (off[i - 1] * off[i - 1]) / p
        if p < 0.0:
            count += 1
    return count


def tridiag_smallest_eig_bisect(diag: List[float], off: List[float], tol: float = 1e-14) -> float:
    n = len(diag)
    if n == 1:
        return diag[0]

    lo = float("inf")
    hi = float("-inf")
    for i in range(n):
        r = 0.0
        if i > 0:
            r += abs(off[i - 1])
        if i < n - 1:
            r += abs(off[i])
        lo = min(lo, diag[i] - r)
        hi = max(hi, diag[i] + r)
    lo -= 1.0
    hi += 1.0

    # Ensure lo has 0 eigenvalues below, hi has >=1 below.
    while tridiag_sturm_count(diag, off, lo) != 0:
        lo -= (hi - lo) * 2.0
    while tridiag_sturm_count(diag, off, hi) < 1:
        hi += (hi - lo) * 2.0

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tridiag_sturm_count(diag, off, mid) >= 1:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) < tol * max(1.0, abs(hi), abs(lo)):
            break
    return hi


def tridiag_eigvec_for_eig(diag: List[float], off: List[float], lam: float) -> List[float]:
    """Compute an eigenvector of symmetric tridiagonal for eigenvalue lam.

    Uses a stable-ish forward recurrence. Good enough for small k (<= few 100).
    """
    n = len(diag)
    if n == 1:
        return [1.0]

    y = [0.0] * n
    y[0] = 1.0
    # choose y[1] from first equation: (d0-lam)y0 + b0 y1 = 0
    if off[0] == 0.0:
        y[1] = 0.0
    else:
        y[1] = - (diag[0] - lam) * y[0] / off[0]

    # recurrence
    for i in range(1, n - 1):
        bi = off[i]
        if bi == 0.0:
            y[i + 1] = 0.0
            continue
        y[i + 1] = - (off[i - 1] * y[i - 1] + (diag[i] - lam) * y[i]) / bi

        # rescale occasionally to avoid overflow/underflow
        if (i % 16) == 0:
            m = max(abs(v) for v in y[: i + 2])
            if m != 0.0 and (m > 1e100 or m < 1e-100):
                s = 1.0 / m
                for j in range(i + 2):
                    y[j] *= s

    # normalize
    nrm = math.sqrt(sum(v * v for v in y))
    if nrm == 0.0:
        y[0] = 1.0
        nrm = 1.0
    return [v / nrm for v in y]


def lanczos_smallest_eig(rows: List[List[Tuple[int, float]]], max_iter: int, tol: float, seed: int) -> float:
    n = len(rows)
    rng = random.Random(seed)
    v = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    nv = math.sqrt(norm2(v))
    if nv == 0.0:
        v[0] = 1.0
        nv = 1.0
    scal(v, 1.0 / nv)

    vs: List[List[float]] = []
    alpha: List[float] = []
    beta: List[float] = []

    last_e0: Optional[float] = None

    v_prev = [0.0] * n
    b_prev = 0.0

    for k in range(max_iter):
        w = matvec(rows, v)
        if k > 0:
            axpy(w, -b_prev, v_prev)

        a = dot(v, w)
        axpy(w, -a, v)

        # full re-orthogonalization against previous Krylov vectors (small n, keeps it clean)
        for q in vs:
            c = dot(q, w)
            if c != 0.0:
                axpy(w, -c, q)

        b = math.sqrt(norm2(w))

        vs.append(v)
        alpha.append(a)
        if k > 0:
            beta.append(b_prev)

        # compute Ritz estimate every few steps once we have a tridiagonal
        if k >= 4 and (k % 2 == 0):
            # For T_k: diag=alpha[0..k], off=beta[0..k-1]
            diag = alpha[: k + 1]
            off = beta[:k]  # length k
            e0 = tridiag_smallest_eig_bisect(diag, off, tol=1e-14)
            if last_e0 is not None and abs(e0 - last_e0) < tol:
                return e0
            last_e0 = e0

        if b == 0.0:
            # Krylov space closed
            diag = alpha[:]
            off = beta[:]  # length len(alpha)-1
            return tridiag_smallest_eig_bisect(diag, off, tol=1e-14)

        v_prev = v
        b_prev = b
        v = w
        scal(v, 1.0 / b)

    # final estimate
    diag = alpha[:]
    off = beta[:]
    return tridiag_smallest_eig_bisect(diag, off, tol=1e-14)


def lanczos_smallest_eigvec(rows: List[List[Tuple[int, float]]], max_iter: int, tol: float, seed: int) -> Tuple[float, List[float]]:
    """Return (smallest_eig, eigenvector) using Lanczos + tridiagonal eigenvector backtransform."""
    n = len(rows)
    rng = random.Random(seed)
    v = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    nv = math.sqrt(norm2(v))
    if nv == 0.0:
        v = [0.0] * n
        v[0] = 1.0
        nv = 1.0
    scal(v, 1.0 / nv)

    vs: List[List[float]] = []
    alpha: List[float] = []
    beta: List[float] = []

    last_e0: Optional[float] = None

    v_prev = [0.0] * n
    b_prev = 0.0

    for k in range(max_iter):
        w = matvec(rows, v)
        if k > 0:
            axpy(w, -b_prev, v_prev)

        a = dot(v, w)
        axpy(w, -a, v)

        # full re-orthogonalization
        for q in vs:
            c = dot(q, w)
            if c != 0.0:
                axpy(w, -c, q)

        b = math.sqrt(norm2(w))

        vs.append(v)
        alpha.append(a)
        if k > 0:
            beta.append(b_prev)

        # Ritz estimate and convergence check
        if k >= 6 and (k % 2 == 0):
            diag = alpha[: k + 1]
            off = beta[:k]
            e0 = tridiag_smallest_eig_bisect(diag, off, tol=1e-14)
            if last_e0 is not None and abs(e0 - last_e0) < tol:
                y = tridiag_eigvec_for_eig(diag, off, e0)
                x = [0.0] * n
                for i in range(len(y)):
                    axpy(x, y[i], vs[i])
                # normalize x
                nx = math.sqrt(norm2(x))
                if nx != 0.0:
                    scal(x, 1.0 / nx)
                return e0, x
            last_e0 = e0

        if b == 0.0:
            diag = alpha[:]
            off = beta[:]
            e0 = tridiag_smallest_eig_bisect(diag, off, tol=1e-14)
            y = tridiag_eigvec_for_eig(diag, off, e0)
            x = [0.0] * n
            for i in range(len(y)):
                axpy(x, y[i], vs[i])
            nx = math.sqrt(norm2(x))
            if nx != 0.0:
                scal(x, 1.0 / nx)
            return e0, x

        v_prev = v
        b_prev = b
        v = w
        scal(v, 1.0 / b)

    diag = alpha[:]
    off = beta[:]
    e0 = tridiag_smallest_eig_bisect(diag, off, tol=1e-14)
    y = tridiag_eigvec_for_eig(diag, off, e0)
    x = [0.0] * n
    for i in range(len(y)):
        axpy(x, y[i], vs[i])
    nx = math.sqrt(norm2(x))
    if nx != 0.0:
        scal(x, 1.0 / nx)
    return e0, x


def combined_local_state_from_bits(occ: int, local_up_bits: int, site: int) -> int:
    """Map (electron occ, local spin) on a site to the 0..7 combined label used in PEPS code.

    electron: 0=D,1=U,2=d,3=0; spin: 0=Up,1=Dn; combined = 2*electron + spin
    """
    nu, nd = electron_n_up_dn(occ, site)
    if nu == 1 and nd == 1:
        e = 0
    elif nu == 1 and nd == 0:
        e = 1
    elif nu == 0 and nd == 1:
        e = 2
    else:
        e = 3
    s = 0 if ((local_up_bits >> site) & 1) else 1
    return 2 * e + s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="Compute one ED case (2x2 recommended).")
    ap.add_argument("--Lx", type=int, default=2)
    ap.add_argument("--Ly", type=int, default=2)
    ap.add_argument("--Ne", type=int, default=4)
    ap.add_argument("--Sz2_total", type=int, default=0)
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--U", type=float, default=0.0)
    ap.add_argument("--JK", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--dump_psi0", action="store_true",
                    help="Dump ED ground-state components with |amp|<eps (configuration basis).")
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--max_lines", type=int, default=80)
    ap.add_argument("--print_default_tables", action="store_true")
    args = ap.parse_args()

    if args.one:
        E, dim = ground_energy_ed(
            Lx=int(args.Lx),
            Ly=int(args.Ly),
            Ne=int(args.Ne),
            Sz2_total=int(args.Sz2_total),
            t=float(args.t),
            U=float(args.U),
            JK=float(args.JK),
            mu=float(args.mu),
        )
        print("# ED one-shot (OBC square)")
        print(f"L={args.Lx}x{args.Ly}, Ne={args.Ne}, Sz2_total={args.Sz2_total}")
        print(f"t={args.t}, U={args.U}, JK={args.JK}, mu={args.mu}")
        print(f"E0={E:.12f}   (dim={dim})")
        return

    if args.dump_psi0:
        E0, vec, basis = ground_state_ed_vector(
            Lx=int(args.Lx),
            Ly=int(args.Ly),
            Ne=int(args.Ne),
            Sz2_total=int(args.Sz2_total),
            t=float(args.t),
            U=float(args.U),
            JK=float(args.JK),
            mu=float(args.mu),
            max_iter=400,
            tol=1e-13,
            seed=0,
        )
        Lx = int(args.Lx)
        Ly = int(args.Ly)
        nsites = Lx * Ly
        eps = float(args.eps)
        max_lines = int(args.max_lines)

        small = []
        for i, amp in enumerate(vec):
            if abs(amp) < eps:
                occ, lup = basis[i]
                cfg = [combined_local_state_from_bits(occ, lup, s) for s in range(nsites)]
                small.append((i, amp, cfg))

        print("# ED ground-state near-zero components in configuration basis")
        print(f"# L={Lx}x{Ly}, Ne={args.Ne}, Sz2_total={args.Sz2_total}  t={args.t} U={args.U} JK={args.JK} mu={args.mu}")
        print(f"# E0={E0:.12f}")
        print(f"# eps={eps}  count(|amp|<eps)={len(small)} / dim={len(vec)}")
        print("# Format: basis_idx, amp, cfg(site0..siteN-1) where cfg is combined local label 0..7 (2*electron + spin)")
        for k, (i, amp, cfg) in enumerate(small[:max_lines]):
            print(f"{i},{amp:.3e},{cfg}")
        if len(small) > max_lines:
            print(f"# ... truncated ({len(small)-max_lines} more)")
        return

    if not args.print_default_tables:
        ap.print_help()
        return

    print("# ED benchmarks for PEPS Kondo (OBC square)")
    print("# Convention: H = -t hop + U doublon + JK s·S - mu n")
    print()

    # 2x2 ED sectors we care about for regression tests.
    Lx, Ly = 2, 2
    print("## 2x2 OBC (ED, fixed Sz2_total=0)")
    print("t=1, mu=0")
    print()
    print("Case A: free fermion + free local spins (U=0, JK=0)")
    for Ne in range(0, 2 * (Lx * Ly) + 1):
        try:
            E, dim = ground_energy_ed(Lx, Ly, Ne, 0, t=1.0, U=0.0, JK=0.0, mu=0.0)
            print(f"Ne={Ne:2d}, Sz2_total=0: E0={E:.12f}   (dim={dim})")
        except ValueError:
            print(f"Ne={Ne:2d}, Sz2_total=0: N/A (empty sector)")
    print()

    print("Case B: Hubbard only (U=4, JK=0)")
    for Ne in (2, 4, 6):
        E, dim = ground_energy_ed(Lx, Ly, Ne, 0, t=1.0, U=4.0, JK=0.0, mu=0.0)
        print(f"Ne={Ne:2d}, Sz2_total=0: E0={E:.12f}   (dim={dim})")
    print()

    print("Case C: Kondo only + hopping (U=0, JK=-1, FM; note JK<0 is FM)")
    for Ne in (2, 4, 6):
        E, dim = ground_energy_ed(Lx, Ly, Ne, 0, t=1.0, U=0.0, JK=-1.0, mu=0.0)
        print(f"Ne={Ne:2d}, Sz2_total=0: E0={E:.12f}   (dim={dim})")
    print()

    print("Case D: Kondo only + hopping (U=0, JK=+1, AF)")
    for Ne in (2, 4, 6):
        E, dim = ground_energy_ed(Lx, Ly, Ne, 0, t=1.0, U=0.0, JK=+1.0, mu=0.0)
        print(f"Ne={Ne:2d}, Sz2_total=0: E0={E:.12f}   (dim={dim})")
    print()

    # 4x4 free-fermion via filling (no ED).
    print("## 4x4 OBC free fermion (analytic filling, U=JK=0)")
    Lx, Ly = 4, 4
    for Ne in (0, 8, 16, 24, 32):
        e = free_fermion_ground_energy(Lx, Ly, t=1.0, Ne=Ne)
        print(f"Ne={Ne:2d}: E0={e:.12f}")

    print()
    print("## Atomic-limit sanity (t=0): closed-form energies")
    print("# JK-only (U=0, mu=0): only singly-occupied sites contribute.")
    print("#  Ne<=N:  Nsingly=Ne;  Ne>N: Nsingly=2N-Ne")
    print("#  AF JK>0: E0 = (-3/4)*JK*Nsingly;  FM JK<0: E0 = (1/4)*JK*Nsingly")
    for (Lx, Ly) in ((2, 2), (4, 4)):
        N = Lx * Ly
        for Ne in (0, N // 2, N, N + N // 2, 2 * N):
            Nsingly = Ne if Ne <= N else (2 * N - Ne)
            e_af = (-0.75) * 1.0 * Nsingly
            e_fm = (0.25) * (-1.0) * Nsingly
            print(f"L={Lx}x{Ly}, Ne={Ne:2d}: JK=+1 => E0={e_af:.12f};  JK=-1 => E0={e_fm:.12f}")


if __name__ == "__main__":
    main()


