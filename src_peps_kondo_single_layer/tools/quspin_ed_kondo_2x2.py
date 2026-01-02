#!/usr/bin/env python3
"""
Independent ED cross-check using QuSpin (requires numpy+scipy+quspin).

Goal:
  Cross-validate the 2x2 OBC Kondo-lattice ground-state energy against
  `ed_kondo_small.py` for the sector:
    - total itinerant electron number Ne fixed
    - total Sz_total (electrons + local spins) fixed

This script intentionally uses a *different implementation path*:
  build electron and local-spin bases in QuSpin, construct sparse Hamiltonian
  with scipy.sparse.kron, then compute the lowest eigenvalue with eigsh.

Environment notes (important):
  - QuSpin typically supports Python 3.9-3.12 (wheels). If you're on 3.14,
    don't bother. Use conda to create a supported Python.
  - Example setup:
      conda create -n quspin310 -c conda-forge python=3.10 numpy scipy quspin -y
      conda activate quspin310
      python src_peps_kondo_single_layer/tools/quspin_ed_kondo_2x2.py --case full

Model conventions:
  - Lattice: 2x2 OBC, sites indexed row-major: i = y*Lx + x.
  - Electron Hubbard part:
      H_t  = -t * sum_<ij>,sigma (c^†_{i,s} c_{j,s} + h.c.)
      H_U  =  U * sum_i n_{i,up} n_{i,dn}
      H_mu = -mu * sum_i (n_{i,up} + n_{i,dn})
  - Kondo coupling (on-site):
      H_K = Jk * sum_i (S_i · s_i)
          = Jk * sum_i (S^z_i s^z_i + 0.5*(S^+_i s^-_i + S^-_i s^+_i))
    where s^z_i = (n_up - n_dn)/2,
          s^+_i = c^†_{up} c_{dn}, s^-_i = c^†_{dn} c_{up},
          S operators are spin-1/2 local moments.

If your `ed_kondo_small.py` uses a different sign convention, you'll see a mismatch.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass


def _die(msg: str) -> "NoReturn":
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


@dataclass(frozen=True)
class Params:
    t: float
    U: float
    Jk: float
    mu: float
    Ne: int
    Sz_tot: int  # total Sz (electrons + local spins), integer because 4 spins -> Sz in [-2..2]


def bonds_2x2_obc():
    # Sites: (0,0)->0, (1,0)->1, (0,1)->2, (1,1)->3
    # NN bonds: horiz (0-1, 2-3), vert (0-2, 1-3)
    return [(0, 1), (2, 3), (0, 2), (1, 3)]


def _try_import_quspin():
    try:
        import numpy as np  # noqa: F401
        import scipy.sparse as sp  # noqa: F401
        import scipy.sparse.linalg as spla  # noqa: F401
        from quspin.basis import spin_basis_general, spin_basis_1d, spinful_fermion_basis_1d  # noqa: F401
        from quspin.operators import hamiltonian  # noqa: F401
    except Exception as e:
        _die(
            "Missing dependencies. Need numpy+scipy+quspin.\n"
            f"Import error: {e}\n\n"
            "Recommended:\n"
            "  conda create -n quspin310 -c conda-forge python=3.10 numpy scipy quspin -y\n"
            "  conda activate quspin310\n"
        )


def build_block_energy_quspin(params: Params, Nup: int, Ndn: int) -> float:
    raise RuntimeError(
        "Outdated: block construction is not used anymore. "
        "We now build the full 4096-dim Hilbert space and project to (Ne,Sz_tot)."
    )


def ground_energy_quspin(params: Params) -> float:
    _try_import_quspin()
    import numpy as np
    import scipy.sparse as sp
    from quspin.basis import spin_basis_1d, spinful_fermion_basis_1d
    from quspin.operators import hamiltonian

    N = 4
    bonds = bonds_2x2_obc()

    Ne = params.Ne
    if Ne < 0 or Ne > 2 * N:
        _die("Ne out of range for 2x2.")

    # Full bases (no fixed Nup/Ndn for electrons, no fixed magnetization for local spins).
    # We project the total Hamiltonian down to the target (Ne, Sz_tot) subspace.
    e_basis = spinful_fermion_basis_1d(N)  # full 4^N electron Fock space
    s_basis = spin_basis_1d(N)  # full 2^N local-spin space

    Ie = sp.identity(e_basis.Ns, format="csc", dtype=np.float64)
    Is = sp.identity(s_basis.Ns, format="csc", dtype=np.float64)

    # --- Electron Hubbard part ---
    # QuSpin 0.3.7 quirk: "-+|" is exactly the negative of "+-|", so adding both cancels.
    # To build hermitian hopping, use ONE opstr with couplings (i->j) and (j->i).
    t = float(params.t)
    U = float(params.U)
    mu = float(params.mu)

    hop_bidir = []
    for (i, j) in bonds:
        hop_bidir.append([-t, i, j])
        hop_bidir.append([-t, j, i])

    static_e = []
    static_e += [["+-|", hop_bidir]]  # up hopping both directions
    static_e += [["|+-", hop_bidir]]  # down hopping both directions
    static_e += [["n|n", [[U, i, i] for i in range(N)]]]
    if abs(mu) > 0.0:
        static_e += [["n|", [[-mu, i] for i in range(N)]]]
        static_e += [["|n", [[-mu, i] for i in range(N)]]]

    He = hamiltonian(
        static_e,
        [],
        basis=e_basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    ).tocsc()

    # --- Local-spin operators ---
    def opS(op: str, i: int):
        return hamiltonian(
            [[op, [[1.0, i]]]],
            [],
            basis=s_basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        ).tocsc()

    Sz_ops = [opS("z", i) for i in range(N)]
    Sp_ops = [opS("+", i) for i in range(N)]
    Sm_ops = [opS("-", i) for i in range(N)]

    # --- Electron spin-density operators ---
    def opE(opstr: str, args):
        return hamiltonian(
            [[opstr, args]],
            [],
            basis=e_basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        ).tocsc()

    nup_ops = [opE("n|", [[1.0, i]]) for i in range(N)]
    ndn_ops = [opE("|n", [[1.0, i]]) for i in range(N)]
    sz_e_ops = [0.5 * (nup_ops[i] - ndn_ops[i]) for i in range(N)]

    # Electron spin-flip operators.
    #
    # In QuSpin 0.3.7 the raw "-|+" operator carries a minus sign relative to the
    # usual s^- = (s^+)† convention on the one-site basis. We fix it here so that
    #   s^- == (s^+)†
    # holds, and the Kondo exchange uses the standard
    #   S·s = Sz*sz + 0.5(S^+ s^- + S^- s^+).
    sp_e_ops = [opE("+|-", [[1.0, i, i]]) for i in range(N)]        # s^+
    sm_e_ops = [-opE("-|+", [[1.0, i, i]]) for i in range(N)]       # s^- (sign-fixed)

    # --- Total Hamiltonian in full Hilbert space ---
    H = sp.kron(He, Is, format="csc")

    Jk = float(params.Jk)
    if abs(Jk) > 0.0:
        for i in range(N):
            # QuSpin `spin_basis_1d` uses Pauli matrices:
            #   Sz eigenvalues are ±1 (not ±1/2), and S± matrix elements are 2.
            # Physical spin-1/2 operators are S_phys = sigma/2.
            # Therefore implement:
            #   Jk * (S_phys · s_phys)
            # = Jk * ( (Sz/2)*sz + 1/2*( (Sp/2)*s^- + (Sm/2)*s^+ ) )
            H = H + (0.5 * Jk) * sp.kron(sz_e_ops[i], Sz_ops[i], format="csc")
            H = H + (0.25 * Jk) * sp.kron(sm_e_ops[i], Sp_ops[i], format="csc")
            H = H + (0.25 * Jk) * sp.kron(sp_e_ops[i], Sm_ops[i], format="csc")

    # --- Project to the target (Ne, Sz_tot) sector ---
    def popcnt(x: int) -> int:
        # Python 3.7 compatibility (your `spin` env).
        v = int(x)
        c = 0
        while v:
            v &= v - 1
            c += 1
        return c

    mask = (1 << N) - 1
    target_Sz2 = 2 * int(params.Sz_tot)  # 2*Sz

    sel = []
    for e_idx, est in enumerate(e_basis.states):
        # QuSpin 0.3.7 convention for `spinful_fermion_basis_1d`:
        #   - lower N bits: spin-down occupations
        #   - upper N bits: spin-up occupations
        ndn = popcnt(est & mask)
        nup = popcnt((est >> N) & mask)
        if nup + ndn != Ne:
            continue
        sz2_e = nup - ndn  # 2*Sz_e
        for s_idx, sst in enumerate(s_basis.states):
            nup_loc = popcnt(sst)
            sz2_loc = 2 * (nup_loc - (N // 2))  # 2*Sz_loc
            if sz2_e + sz2_loc != target_Sz2:
                continue
            sel.append(e_idx * s_basis.Ns + s_idx)

    if not sel:
        return float("inf")

    Hs = H[sel, :][:, sel].toarray()
    w = np.linalg.eigvalsh(Hs)
    return float(w[0])


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=str, default="full", choices=["full", "free", "Uonly", "Jkonly"])
    p.add_argument("--t", type=float, default=None)
    p.add_argument("--U", type=float, default=None)
    p.add_argument("--Jk", type=float, default=None)
    p.add_argument("--mu", type=float, default=None)
    p.add_argument("--Ne", type=int, default=4)
    p.add_argument("--Sz", type=int, default=0)
    return p.parse_args(argv)


def main(argv):
    a = parse_args(argv)
    # Defaults aligned with our "full parameter" 2x2 test.
    t, U, Jk, mu = 1.0, 4.0, -1.0, 0.0
    if a.case == "free":
        t, U, Jk, mu = 1.0, 0.0, 0.0, 0.0
    elif a.case == "Uonly":
        t, U, Jk, mu = 1.0, 4.0, 0.0, 0.0
    elif a.case == "Jkonly":
        t, U, Jk, mu = 0.0, 0.0, -1.0, 0.0

    if a.t is not None:
        t = a.t
    if a.U is not None:
        U = a.U
    if a.Jk is not None:
        Jk = a.Jk
    if a.mu is not None:
        mu = a.mu

    params = Params(t=t, U=U, Jk=Jk, mu=mu, Ne=a.Ne, Sz_tot=a.Sz)

    E0 = ground_energy_quspin(params)
    if not math.isfinite(E0):
        _die("Failed to find a finite ground energy. Check sector constraints.")

    print(f"[QuSpin-ED] 2x2 OBC: t={t} U={U} Jk={Jk} mu={mu} Ne={a.Ne} Sz_tot={a.Sz}")
    print(f"[QuSpin-ED] E0 = {E0:.15f}")


if __name__ == "__main__":
    main(sys.argv[1:])


