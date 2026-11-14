#!/usr/bin/env python3
"""
verify_master_form.py
=====================
Checks behind Sec. 2.3-2.4 of the revised manuscript (the one-colour ansatz,
the closed-form eigenvalue, and the "master form" of the free energy).

On the one-colour axis
    x_0 = [1+(q-1)phi]/q,   x_1 = ... = x_{q-1} = (1-phi)/q
the variational free energy can be written exactly as

    beta f(phi) = [beta h (q-1)/(2q)] phi (phi - phi_t) - beta h/(2q) - ln g(B(phi)),

    phi_t = (q-2)/(q-1),  B = u s + c/s,  s = exp(beta h phi/2),
    u = e^{beta J},  c = u+q-2,  d = (u-1)(u+q-1),
    g = larger root of g^2 - B g + d = 0.

This script verifies, in order:
  1. the master form against direct diagonalisation of the q x q transfer
     matrix (it is an identity, not an approximation);
  2. the paramagnetic-curvature formula f''(0);
  3. the reflection symmetry f(phi) = f(phi_t - phi) at h_t, and its failure
     away from h_t;
  4. uniqueness: solving {f(0)=f(phi*), f'(phi*)=0} for (phi*, h) from a wide
     grid of starting points returns only (phi_t, h_t);
  5. the closed-form barrier beta*Delta f at h_t;
  6. the full Hessian of beta f(x) on the simplex at both coexisting states
     (stability in every direction, not only along the ansatz);
  7. a random-start minimisation of beta f(x) over the whole simplex at h_t
     (no state lower than the one-colour minima);
  8. the herding-overlap jump Delta Omega = (q-2)^2 / [q(q-1)].

Run from the repository root:
    python scripts/verify_master_form.py            # prints a report
    python scripts/verify_master_form.py --json out.json
"""
import argparse
import json

import numpy as np
from numpy.linalg import eigvalsh
from scipy.optimize import fsolve, minimize
import mpmath as mp

QS = [3, 4, 5, 8]
BJS = [0.3, 1.0, 2.0]


# ---------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------
def bh_t(q, bJ):
    return 2.0 * (q - 1) / (q - 2) * np.log1p((q - 2) * np.exp(-bJ))


def bh_sp(q, bJ):
    return q * q / (2.0 * np.exp(bJ) + q - 2)


def phi_t(q):
    return (q - 2.0) / (q - 1.0)


def barrier_closed(q, bJ):
    u = np.exp(bJ)
    w = 1.0 + (q - 2) * np.exp(-bJ)
    d = (u - 1) * (u + q - 1)
    return (-(q - 2) / (4.0 * q) * np.log(w)
            - np.log((u * np.sqrt(w) + np.sqrt(u * u * w - d)) / (u + q - 1)))


# ---------------------------------------------------------------------------
# free energies
# ---------------------------------------------------------------------------
def x_of_phi(phi, q):
    x = np.full(q, (1.0 - phi) / q)
    x[0] = (1.0 + (q - 1) * phi) / q
    return x


def bf_matrix(x, bJ, bh):
    """beta f(x) by direct diagonalisation of the q x q transfer matrix."""
    q = len(x)
    T = np.exp(bJ * np.eye(q) + 0.5 * bh * (x[:, None] + x[None, :]))
    return 0.5 * bh * x @ x - np.log(eigvalsh(T)[-1])


def bf_master(phi, q, bJ, bh):
    """beta f(phi) from the master form."""
    u = np.exp(bJ)
    c = u + q - 2
    d = (u - 1) * (u + q - 1)
    s = np.exp(0.5 * bh * phi)
    B = u * s + c / s
    g = 0.5 * (B + np.sqrt(B * B - 4 * d))
    return (bh * (q - 1) / (2 * q) * phi * (phi - phi_t(q))
            - bh / (2 * q) - np.log(g))


def bf_master_mp(phi, q, bJ, bh):
    phi, bJ, bh = mp.mpf(phi), mp.mpf(bJ), mp.mpf(bh)
    u = mp.e ** bJ
    c = u + q - 2
    d = (u - 1) * (u + q - 1)
    s = mp.e ** (bh * phi / 2)
    B = u * s + c / s
    g = (B + mp.sqrt(B * B - 4 * d)) / 2
    pt = mp.mpf(q - 2) / (q - 1)
    return bh * (q - 1) / (2 * q) * phi * (phi - pt) - bh / (2 * q) - mp.log(g)


def dbf_master(phi, q, bJ, bh, eps=1e-6):
    return (bf_master(phi + eps, q, bJ, bh)
            - bf_master(phi - eps, q, bJ, bh)) / (2 * eps)


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_identity():
    worst = 0.0
    for q in QS:
        for bJ in [0.0, 0.3, 0.7, 1.0, 1.6, 2.0]:
            for bh in [0.5, 1.3, 3.0, bh_t(q, max(bJ, 1e-9))]:
                for phi in np.linspace(-1.0 / (q - 1), 1.0, 23):
                    a = bf_matrix(x_of_phi(phi, q), bJ, bh)
                    b = bf_master(phi, q, bJ, bh)
                    worst = max(worst, abs(a - b))
    return worst


def check_curvature():
    worst = 0.0
    mp.mp.dps = 40
    for q in QS:
        for bJ in BJS:
            for bh in [0.5, 1.0, 2.0]:
                num = mp.diff(lambda p: bf_master_mp(p, q, bJ, bh), 0, 2)
                u = mp.e ** mp.mpf(bJ)
                bhm = mp.mpf(bh)
                ana = bhm * (q - 1) / mp.mpf(q) ** 3 * (q * q - bhm * (2 * u + q - 2))
                worst = max(worst, float(abs(num - ana)))
    return worst


def check_reflection():
    mp.mp.dps = 45
    out = {}
    for q in QS:
        for bJ in [0.3, 1.0]:
            res = {}
            for fac in [0.90, 0.99, 1.00, 1.01, 1.10]:
                u = mp.e ** mp.mpf(bJ)
                bht = fac * 2 * mp.mpf(q - 1) / (q - 2) * mp.log(1 + (q - 2) / u)
                pt = mp.mpf(q - 2) / (q - 1)
                m = 0
                for k in range(401):
                    p = -mp.mpf("0.3") + (pt + mp.mpf("0.6")) * k / 400
                    m = max(m, abs(bf_master_mp(p, q, bJ, bht)
                                   - bf_master_mp(pt - p, q, bJ, bht)))
                res[f"{fac:.2f}"] = float(m)
            out[f"q={q},bJ={bJ}"] = res
    return out


def check_uniqueness(nphi=25, nh=25):
    """Solve {f(0)=f(phi), f'(phi)=0} for (phi, h) from a wide grid of starts."""
    out = {}
    for q in QS:
        for bJ in BJS:
            ht = bh_t(q, bJ)
            lo = -1.0 / (q - 1) + 1e-3
            sols = []
            n_h0 = 0
            for p0 in np.linspace(lo, 0.98, nphi):
                for h0 in np.linspace(0.2 * ht, 3.0 * ht, nh):
                    def F(v):
                        p, bh = v
                        return [bf_master(p, q, bJ, bh) - bf_master(0.0, q, bJ, bh),
                                dbf_master(p, q, bJ, bh)]
                    v, info, ier, _ = fsolve(F, [p0, h0], full_output=True,
                                             xtol=1e-13)
                    p, bh = v
                    if ier != 1 or bh <= 0 or not (lo <= p <= 1.0):
                        continue
                    if np.max(np.abs(F(v))) > 1e-9:
                        continue
                    if abs(p) < 1e-4:          # trivial branch phi*=0
                        continue
                    if bh < 1e-6 * ht:         # degenerate h=0: f(phi) constant
                        n_h0 += 1
                        continue
                    sols.append((round(float(p), 7), round(float(bh), 7)))
            uniq = sorted(set(sols))
            out[f"q={q},bJ={bJ}"] = dict(
                n_converged=len(sols), distinct=uniq, n_degenerate_h0=n_h0,
                expected=(round(phi_t(q), 7), round(float(ht), 7)))
    return out


def check_barrier():
    out = {}
    for q in QS:
        for bJ in BJS:
            bh = bh_t(q, bJ)
            num = bf_master(phi_t(q) / 2, q, bJ, bh) - bf_master(0.0, q, bJ, bh)
            # also locate the maximum numerically between the minima
            ps = np.linspace(0, phi_t(q), 20001)
            fs = bf_master(ps, q, bJ, bh) - bf_master(0.0, q, bJ, bh)
            out[f"q={q},bJ={bJ}"] = dict(
                closed=float(barrier_closed(q, bJ)), at_midpoint=float(num),
                grid_max=float(fs.max()), argmax=float(ps[fs.argmax()]))
    return out


def _tangent_basis(q):
    M = np.eye(q) - 1.0 / q
    U, _, _ = np.linalg.svd(M)
    return U[:, :q - 1]


def _hess(fun, x0, E, eps=1e-4):
    n = E.shape[1]
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = E[:, i] * eps, E[:, j] * eps
            H[i, j] = (fun(x0 + ei + ej) - fun(x0 + ei - ej)
                       - fun(x0 - ei + ej) + fun(x0 - ei - ej)) / (4 * eps * eps)
    return 0.5 * (H + H.T)


def check_hessian(bJ=1.0):
    out = {}
    for q in QS:
        bh = bh_t(q, bJ)
        E = _tangent_basis(q)
        f = lambda x: bf_matrix(x, bJ, bh)
        ev_ord = eigvalsh(_hess(f, x_of_phi(phi_t(q), q), E))
        ev_dis = eigvalsh(_hess(f, x_of_phi(0.0, q), E))
        out[f"q={q}"] = dict(ordered=[round(v, 4) for v in ev_ord],
                             disordered=[round(v, 4) for v in ev_dis])
    return out


def check_global(bJs=(0.0, 0.3, 1.0, 2.0), nstart=400, seed=1):
    """Random-start minimisation of beta f(x) over the simplex at h_t."""
    rng = np.random.default_rng(seed)
    out = {}
    for q, bJ in [(q, b) for q in QS for b in bJs]:
        bh = bh_t(q, bJ)
        f0 = bf_matrix(x_of_phi(0.0, q), bJ, bh)

        def obj(z):          # softmax keeps x on the simplex
            x = np.exp(z - z.max()); x /= x.sum()
            return bf_matrix(x, bJ, bh)
        best = np.inf
        best_x = None
        for _ in range(nstart):
            z0 = rng.normal(scale=2.0, size=q)
            r = minimize(obj, z0, method="Nelder-Mead",
                         options=dict(xatol=1e-10, fatol=1e-14, maxiter=20000))
            if r.fun < best:
                best = r.fun
                x = np.exp(r.x - r.x.max()); best_x = x / x.sum()
        out[f"q={q},bJ={bJ}"] = dict(f_min_minus_f0=float(best - f0),
                             x_at_min=sorted([round(v, 5) for v in best_x],
                                             reverse=True))
    return out


def jumps():
    return {f"q={q}": dict(phi_t=phi_t(q),
                           dOmega=(q - 2) ** 2 / (q * (q - 1.0)))
            for q in QS}


def main():
    np.seterr(over="ignore", invalid="ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true",
                    help="smaller uniqueness grid and fewer random starts")
    args = ap.parse_args()

    rep = {}
    rep["master_form_vs_matrix_max_dev"] = check_identity()
    print(f"[1] master form vs transfer matrix: max |dev| = "
          f"{rep['master_form_vs_matrix_max_dev']:.1e}")

    rep["curvature_formula_max_dev"] = check_curvature()
    print(f"[2] f''(0) closed form vs 40-digit numerics: max |dev| = "
          f"{rep['curvature_formula_max_dev']:.1e}")

    rep["reflection"] = check_reflection()
    print("[3] reflection residual max_phi |f(phi)-f(phi_t-phi)| (45 digits)")
    for k, v in rep["reflection"].items():
        print("    ", k, {a: f"{b:.1e}" for a, b in v.items()})

    n = 9 if args.quick else 25
    rep["uniqueness"] = check_uniqueness(n, n)
    print("[4] solutions of {f(0)=f(phi*), f'(phi*)=0}, phi*!=0:")
    for k, v in rep["uniqueness"].items():
        print(f"     {k}: {v['n_converged']} converged starts -> "
              f"distinct {v['distinct']}  (expected {v['expected']}; "
              f"{v['n_degenerate_h0']} starts fell onto the degenerate h=0 point)")

    rep["barrier"] = check_barrier()
    print("[5] barrier beta*Delta f at h_t: closed form vs numerics")
    for k, v in rep["barrier"].items():
        print(f"     {k}: closed={v['closed']:.10e}  midpoint={v['at_midpoint']:.10e}"
              f"  grid-max={v['grid_max']:.10e} at phi={v['argmax']:.5f}")

    rep["hessian_bJ1"] = check_hessian()
    print("[6] Hessian eigenvalues of beta f(x) on the simplex at h_t (bJ=1):")
    for k, v in rep["hessian_bJ1"].items():
        print(f"     {k}: ordered {v['ordered']}   disordered {v['disordered']}")

    rep["global"] = check_global(nstart=40 if args.quick else 200)
    print("[7] random-start minimisation over the full simplex at h_t:")
    for k, v in rep["global"].items():
        print(f"     {k}: min f - f(0) = {v['f_min_minus_f0']:.2e}  at x = {v['x_at_min']}")

    rep["jumps"] = jumps()
    print("[8] jumps:", {k: (round(v['phi_t'], 6), round(v['dOmega'], 6))
                         for k, v in rep["jumps"].items()})

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rep, fh, indent=2, default=str)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
