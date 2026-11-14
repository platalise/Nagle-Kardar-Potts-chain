#!/usr/bin/env python3
"""
run_reweighting.py
==================
Susceptibility peak, Binder cumulant and P(phi) from long runs at h_t plus
exact single-histogram reweighting in h.

Why this replaces a direct fine-grid scan near h_t
--------------------------------------------------
At L >= 3200 the mean time between crossings of the free-energy barrier
exceeds the length of a single grid-point run (4e4 MCS).  A direct scan,
however fine its grid, then samples one basin at a time: chi(h) jumps between
the single-phase value (~0.002 L) and values set by the one or two crossings a
run happens to contain.  The maximum over such a scan is not an equilibrium
estimate, and per-run jackknife errors (which assume independent samples)
are far too small.

Here instead every size is run at h_t for long enough to contain many
crossings, in R independent replicas.  Because h enters the Hamiltonian only
through S = sum_a n_a(n_a-1), reweighting from h_t to any h is exact:

    <A>_h = sum_k A_k w_k / sum_k w_k,   w_k = exp[ beta (h - h_t) S_k / (2L) ].

Errors are leave-one-replica-out jackknife, i.e. they include the
replica-to-replica scatter of the crossing statistics.

Usage (repository root):
    python scripts/run_reweighting.py run        # MC, ~15 min on 8 cores
    python scripts/run_reweighting.py analyze    # reweighting -> CSV
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

# make sure the package of THIS checkout is imported (not another installed copy)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

Q, J, BETA = 3, 1.0, 1.0
HT = 2.0 * (Q - 1) / (Q - 2) * np.log1p((Q - 2) * np.exp(-BETA * J))
PHIT = (Q - 2.0) / (Q - 1.0)
OUT = os.path.join("data", "longrun")

# L: (replicas, MCS per replica, sampling interval in MCS)
PLAN = {400: (8, 400_000, 4),
        800: (8, 400_000, 4),
        1600: (8, 600_000, 6),
        3200: (8, 1_200_000, 12),
        6400: (8, 2_500_000, 25)}
BURN = 20_000


def _task(args):
    from nkpotts.mc_corrected import run_record
    L, r, nmeas, freq = args
    seed = 700_000 + 1_000 * r + L
    nmax, S, nb = run_record(L, HT, Q, J, BETA, BURN, nmeas, freq, seed)
    fn = os.path.join(OUT, f"L{L}_r{r}.npz")
    np.savez_compressed(fn, nmax=nmax.astype(np.int16), S=S, nb=nb.astype(np.int16),
                        L=L, h=HT, burn=BURN, nmeas=nmeas, freq=freq, seed=seed)
    return fn


def do_run(nproc, Ls):
    os.makedirs(OUT, exist_ok=True)
    tasks = [(L, r, PLAN[L][1], PLAN[L][2])
             for L in sorted(Ls, reverse=True) for r in range(PLAN[L][0])]
    with Pool(nproc) as p:
        for fn in p.imap_unordered(_task, tasks):
            print("  wrote", fn, flush=True)


# ---------------------------------------------------------------------------
def crossings(phi, lo=PHIT / 4, hi=3 * PHIT / 4):
    """Barrier crossings, counted with a hysteresis band [lo, hi]."""
    state, n = None, 0
    for v in phi:
        s = 0 if v < lo else (1 if v > hi else None)
        if s is not None:
            if state is not None and s != state:
                n += 1
            state = s
    return n


# Features whose weighted sums give every observable (e = a - h b):
#   0:1  1:phi  2:phi^2  3:phi^4  4:a  5:a^2  6:b  7:b^2  8:ab  9:o  10:o*phi  11:o*phi^2
# with a = E_loc/L, b = S/(2L^2), o = [phi > phi_t/2] (ordered basin).
def _features(rep, L):
    phi = rep["phi"]
    a = rep["eloc"] / L
    b = rep["S"].astype(float) / (2.0 * L * L)
    o = (phi > PHIT / 2).astype(float)
    return np.column_stack([np.ones_like(phi), phi, phi ** 2, phi ** 4, a, a * a,
                            b, b * b, a * b, o, o * phi, o * phi ** 2])


def _replica_sums(reps, L, hs, block=64):
    """T[r, i, k] = sum_samples w_i * F_k for replica r and field hs[i], with a
    common (replica-independent) normalisation of the weights at each field."""
    c = BETA * (hs - HT) / (2.0 * L)
    smax = max(r["S"].max() for r in reps)
    smin = min(r["S"].min() for r in reps)
    off = np.maximum(c * smax, c * smin)
    T = np.zeros((len(reps), len(hs), 12))
    for ir, rep in enumerate(reps):
        F = _features(rep, L)
        S = rep["S"].astype(float)
        for j in range(0, len(hs), block):
            W = np.exp(np.outer(c[j:j + block], S) - off[j:j + block, None])
            T[ir, j:j + block] = W @ F
    return T


def _observables(T, hs, L):
    """Observables on the field axis from summed weights T[i, k]."""
    Z = T[:, 0]
    m1, m2, m4 = T[:, 1] / Z, T[:, 2] / Z, T[:, 3] / Z
    ea, ea2, eb, eb2, eab = (T[:, k] / Z for k in (4, 5, 6, 7, 8))
    e1 = ea - hs * eb
    e2 = ea2 - 2 * hs * eab + hs ** 2 * eb2
    p = T[:, 9] / Z
    mu1 = T[:, 10] / T[:, 9]
    mu0 = (T[:, 1] - T[:, 10]) / (Z - T[:, 9])
    v1 = T[:, 11] / T[:, 9] - mu1 ** 2
    v0 = (T[:, 2] - T[:, 11]) / (Z - T[:, 9]) - mu0 ** 2
    chi = BETA * L * (m2 - m1 ** 2)
    u4 = 1.0 - m4 / (3.0 * m2 ** 2)
    cv = BETA ** 2 * L * (e2 - e1 ** 2)
    return np.column_stack([m1, chi, u4, cv, p, mu0, mu1, v0, v1])


def _curves_all(reps, L, hs):
    Tr = _replica_sums(reps, L, hs)
    Ttot = Tr.sum(axis=0)
    full = _observables(Ttot, hs, L)
    jk = np.array([_observables(Ttot - Tr[k], hs, L) for k in range(len(reps))])
    return full, jk


def do_analyze():
    curve_rows, summary, hist_rows, check_rows = [], [], [], []
    meta = {}
    for L in sorted(PLAN):
        files = sorted(glob.glob(os.path.join(OUT, f"L{L}_r*.npz")))
        if not files:
            print(f"  [skip] no runs for L={L}")
            continue
        reps = []
        ncross = []
        for fn in files:
            d = np.load(fn)
            phi = (Q * d["nmax"].astype(float) / L - 1.0) / (Q - 1.0)
            reps.append(dict(phi=phi, S=d["S"], eloc=-J * d["nb"].astype(float)))
            ncross.append(crossings(phi))
        R = len(reps)
        mcs = int(np.load(files[0])["nmeas"]) * R
        # window: L*beta*(h-h_t) in [-120, 40], but |h-h_t| <= 0.1 so that the
        # reweighting factor stays moderate at small L
        hs = HT + np.linspace(max(-120.0 / L, -0.1), 40.0 / L, 761)
        full, jk = _curves_all(reps, L, hs)

        def jkerr(a):
            return np.sqrt((R - 1) * np.mean((a - a.mean(axis=0)) ** 2, axis=0))
        err = jkerr(jk)
        for i, h in enumerate(hs):
            curve_rows.append(dict(L=L, h=h, phi=full[i, 0], phi_err=err[i, 0],
                                   chi=full[i, 1], chi_err=err[i, 1],
                                   U4=full[i, 2], U4_err=err[i, 2],
                                   Cv=full[i, 3], Cv_err=err[i, 3],
                                   p_ord=full[i, 4], p_ord_err=err[i, 4]))
        ip = int(np.argmax(full[:, 1]))
        chimax_jk = jk[:, :, 1].max(axis=1)
        hpk_jk = hs[jk[:, :, 1].argmax(axis=1)]
        iu = int(np.argmin(full[:, 2]))

        def heq(curve):            # field where the ordered weight is 1/2
            pc = curve[:, 4]
            j = int(np.argmax(pc >= 0.5))
            if j == 0:
                return np.nan
            return hs[j - 1] + (0.5 - pc[j - 1]) * (hs[j] - hs[j - 1]) / (pc[j] - pc[j - 1])
        heq_full = heq(full)
        heq_jk = np.array([heq(c) for c in jk])
        p_pk, mu0, mu1, v0, v1 = full[ip, 4:9]
        between = p_pk * (1 - p_pk) * (mu1 - mu0) ** 2
        within = p_pk * v1 + (1 - p_pk) * v0
        u4min_jk = jk[:, :, 2].min(axis=1)
        summary.append(dict(
            L=L, replicas=R, mcs_total=mcs, crossings=int(sum(ncross)),
            crossings_per_replica_min=int(min(ncross)),
            chi_max=full[ip, 1], chi_max_err=float(jkerr(chimax_jk)),
            chi_max_over_L=full[ip, 1] / L,
            chi_max_over_L_err=float(jkerr(chimax_jk)) / L,
            h_peak=hs[ip], h_peak_err=float(jkerr(hpk_jk)),
            h_peak_minus_ht=hs[ip] - HT,
            U4_min=full[iu, 2], U4_min_err=float(jkerr(u4min_jk)),
            h_U4_min=hs[iu],
            h_eq=heq_full, h_eq_err=float(jkerr(heq_jk)),
            h_eq_minus_ht=heq_full - HT,
            p_ord_at_ht=float(full[np.argmin(np.abs(hs - HT)), 4]),
            p_ord_at_ht_err=float(err[np.argmin(np.abs(hs - HT)), 4]),
            p_ord_at_peak=p_pk, mu0_at_peak=mu0, mu1_at_peak=mu1,
            var_between_at_peak=between, var_within_at_peak=within))
        # cross-check against the independent field scan at the scan fields
        scan = pd.read_csv(os.path.join("data", "data_observables.csv"))
        scan = scan[(scan.L == L) & (scan.h >= hs[0]) & (scan.h <= hs[-1])]
        if len(scan):
            fs, jks = _curves_all(reps, L, scan.h.values)
            es = np.sqrt((R - 1) * np.mean((jks - jks.mean(axis=0)) ** 2, axis=0))
            zphi = (fs[:, 0] - scan.M.values) / np.hypot(es[:, 0], scan.M_err.values)
            zchi = (fs[:, 1] - scan.chi.values) / np.hypot(es[:, 1], scan.chi_err.values)
            for hh, a0, b0, a1, b1, z0, z1 in zip(scan.h.values, fs[:, 0], scan.M.values,
                                                 fs[:, 1], scan.chi.values, zphi, zchi):
                check_rows.append(dict(L=L, h=hh, phi_rw=a0, phi_scan=b0,
                                       chi_rw=a1, chi_scan=b1, z_phi=z0, z_chi=z1))
        # P(phi) at h_t, pooled over replicas (no reweighting)
        allphi = np.concatenate([r["phi"] for r in reps])
        dens, edges = np.histogram(allphi, bins=np.linspace(0, 0.85, 52), density=True)
        for c, p in zip(0.5 * (edges[1:] + edges[:-1]), dens):
            hist_rows.append(dict(L=L, phi=c, density=p))
        meta[L] = dict(files=[os.path.basename(f) for f in files],
                       crossings=ncross,
                       frac_ordered_at_ht=float((allphi > PHIT / 2).mean()),
                       frac_between_02_03=float(((allphi > 0.2) & (allphi < 0.3)).mean()))
        s = summary[-1]
        print(f"L={L:5d}  R={R}  MCS={mcs:.1e}  crossings={s['crossings']:5d} "
              f"(min/replica {s['crossings_per_replica_min']})  "
              f"chi_max/L={s['chi_max_over_L']:.4f}+-{s['chi_max_over_L_err']:.4f}  "
              f"h_pk-h_t={s['h_peak_minus_ht']:+.2e}+-{s['h_peak_err']:.1e}  "
              f"U4min={s['U4_min']:.3f}+-{s['U4_min_err']:.3f}  "
              f"p_ord(h_t)={s['p_ord_at_ht']:.3f}+-{s['p_ord_at_ht_err']:.3f}  "
              f"L(h_eq-h_t)={L*s['h_eq_minus_ht']:+.2f}+-{L*s['h_eq_err']:.2f}  "
              f"peak: mu0={mu0:.3f} mu1={mu1:.3f} between={between:.4f} within={within:.4f}")
    pd.DataFrame(curve_rows).to_csv("data/data_reweighted_curves.csv", index=False)
    pd.DataFrame(summary).to_csv("data/data_chi_reweighted.csv", index=False)
    pd.DataFrame(hist_rows).to_csv("data/data_histograms_longrun.csv", index=False)
    ck = pd.DataFrame(check_rows)
    ck.to_csv("data/data_reweighting_vs_scan.csv", index=False)
    for L, g in ck.groupby("L"):
        print(f"  cross-check L={L}: {len(g)} scan fields, median |z_phi|={g.z_phi.abs().median():.2f},"
              f" median |z_chi|={g.z_chi.abs().median():.2f}, max |z_chi|={g.z_chi.abs().max():.1f}")
    with open("data/longrun_meta.json", "w") as fh:
        json.dump(dict(h_t=HT, plan={str(k): v for k, v in PLAN.items()},
                       burn=BURN, per_L=meta), fh, indent=2)
    print("wrote data/data_chi_reweighted.csv, data/data_reweighted_curves.csv, "
          "data/data_histograms_longrun.csv, data/longrun_meta.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--L", type=int, nargs="+", default=sorted(PLAN))
    a = ap.parse_args()
    if a.mode == "run":
        do_run(a.nproc, a.L)
    else:
        do_analyze()


if __name__ == "__main__":
    main()
