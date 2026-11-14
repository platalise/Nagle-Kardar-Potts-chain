#!/usr/bin/env python3
"""
verify_data_vs_claims.py
========================
Recompute every Monte Carlo number quoted in the revised manuscript from the
data files in data/, and compare it with the value printed in the text.

Each entry of CLAIMS is (label, value as printed, tolerance, function that
recomputes it from data).  The script prints one line per claim and exits with
status 1 if any claim fails, so it can be run before every resubmission:

    python scripts/verify_data_vs_claims.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

D = "data"
Q, BJ = 3, 1.0
HT = 2.0 * (Q - 1) / (Q - 2) * np.log1p((Q - 2) * np.exp(-BJ))
PHIT = (Q - 2) / (Q - 1)
SIZES = [400, 800, 1600, 3200, 6400]


def _csv(name):
    return pd.read_csv(os.path.join(D, name))


# ---------------------------------------------------------------- sources
def uniform_peak(L):
    g = _csv("data_observables.csv").query("L == @L")
    return float(g.chi.max())


def rw(L, col):
    s = _csv("data_chi_reweighted.csv").set_index("L")
    return float(s.loc[L, col])


def ratio_rw_uniform(L):
    return rw(L, "chi_max") / uniform_peak(L)


def local_slope(L1, L2):
    return np.log(rw(L2, "chi_max") / rw(L1, "chi_max")) / np.log(L2 / L1)


def global_p():
    y = [rw(L, "chi_max") for L in SIZES]
    return float(np.polyfit(np.log(SIZES), np.log(y), 1)[0])


def fwhm_points(L):
    """Number of uniform-grid points (dh = 7.5e-3) inside the FWHM of the
    reweighted chi(h)."""
    c = _csv("data_reweighted_curves.csv").query("L == @L")
    half = c.chi.max() / 2
    above = c[c.chi >= half]
    width = above.h.max() - above.h.min()
    grid = _csv("data_observables.csv").query("L == @L").h.values
    return int(((grid >= above.h.min()) & (grid <= above.h.max())).sum()), width


def u4min_any(L):
    """Lowest U4 sampled at size L on the uniform grid or on the reweighted
    axis."""
    a = _csv("data_observables.csv").query("L == @L").U4.min()
    return float(min(a, rw(L, "U4_min")))


def crossings(L):
    return int(rw(L, "crossings"))


def barrier_L(L):
    u = np.exp(BJ)
    w = 1 + (Q - 2) * np.exp(-BJ)
    d = (u - 1) * (u + Q - 1)
    bf = -(Q - 2) / (4 * Q) * np.log(w) - np.log((u * np.sqrt(w) + np.sqrt(u * u * w - d)) / (u + Q - 1))
    return bf * L


def frac_ordered_ht(L):
    m = json.load(open(os.path.join(D, "longrun_meta.json")))
    return m["per_L"][str(L)]["frac_ordered_at_ht"]


def hist_files():
    return sorted(_csv("data_histograms_longrun.csv").L.unique().tolist())


def hyst_sizes():
    return sorted(_csv("data_hysteresis.csv").L.unique().tolist())


def obs_sizes():
    return sorted(_csv("data_observables.csv").L.unique().tolist())


# ---------------------------------------------------------------- claims
# (label, value as printed in tex/main.tex or tex/SI.tex, tolerance, recompute)
# tolerance None means exact equality (integers, lists).
def _mcs_per_crossing(L):
    s = _csv("data_chi_reweighted.csv").set_index("L")
    return s.loc[L, "mcs_total"] / s.loc[L, "crossings"]


def _between(L):
    m = json.load(open(os.path.join(D, "longrun_meta.json")))
    return m["per_L"][str(L)]["frac_between_02_03"]


def _hyst_jump(L, branch):
    d = _csv("data_hysteresis.csv")
    g = d[(d.L == L) & (d.branch == branch)]
    if branch == "up":
        return float(g.sort_values("h")[g.sort_values("h").phi > PHIT / 2].h.min())
    g = g.sort_values("h", ascending=False)
    return float(g[g.phi < PHIT / 2].h.max())


def _crossings_min_replica():
    m = json.load(open(os.path.join(D, "longrun_meta.json")))
    return min(min(v["crossings"]) for v in m["per_L"].values())


CLAIMS = [
    # Sec. 4.2, Fig. 4a
    *[(f"chi_max/L, L={L}", v, 5e-5, (lambda L=L: rw(L, "chi_max_over_L")))
      for L, v in zip(SIZES, [0.0381, 0.0361, 0.0370, 0.0418, 0.0486])],
    ("max error of chi_max/L <= 2e-4", 2e-4, 2e-4,
     lambda: max(rw(L, "chi_max_over_L_err") for L in SIZES)),
    ("chi_max/L at 6400 as fraction of 1/16 (78%)", 0.78, 0.005,
     lambda: rw(6400, "chi_max_over_L") / 0.0625),
    # Fig. 4b
    *[(f"ratio rw/scan, L={L}", v, 0.005, (lambda L=L: ratio_rw_uniform(L)))
      for L, v in zip(SIZES, [0.95, 0.96, 1.01, 1.06, 1.40])],
    # local slopes and global p
    *[(f"local slope {a}->{b}", v, 0.005, (lambda a=a, b=b: local_slope(a, b)))
      for (a, b), v in zip(zip(SIZES, SIZES[1:]), [0.92, 1.03, 1.18, 1.22])],
    ("global exponent p", 1.09, 0.005, global_p),
    # peak position
    ("beta(h_peak-h_t), L=400 (1.7e-2)", 1.7e-2, 5e-4, lambda: rw(400, "h_peak_minus_ht")),
    ("beta(h_peak-h_t), L=6400 (-3.3e-4)", -3.3e-4, 5e-6, lambda: rw(6400, "h_peak_minus_ht")),
    ("error of h_peak, L=6400 (1.3e-4)", 1.3e-4, 5e-6, lambda: rw(6400, "h_peak_err")),
    # FWHM and grid coverage
    *[(f"uniform points inside FWHM, L={L}", v, None, (lambda L=L: fwhm_points(L)[0]))
      for L, v in zip(SIZES, [18, 9, 4, 2, 1])],
    ("FWHM L=400 (0.13)", 0.13, 0.005, lambda: fwhm_points(400)[1]),
    ("FWHM L=6400 (0.0072)", 0.0072, 5e-5, lambda: fwhm_points(6400)[1]),
    # crossings
    ("MCS per crossing, L=400 (170)", 170, 10, lambda: _mcs_per_crossing(400)),
    ("MCS per crossing, L=6400 (3.3e4)", 3.3e4, 500, lambda: _mcs_per_crossing(6400)),
    ("crossings at L=6400 (608)", 608, None, lambda: crossings(6400)),
    ("min crossings per replica (60)", 60, None, _crossings_min_replica),
    # variance split at the peak, L=6400
    ("between-basin variance, L=6400", 0.0467, 5e-5, lambda: rw(6400, "var_between_at_peak")),
    ("within-basin variance, L=6400", 0.0020, 5e-5, lambda: rw(6400, "var_within_at_peak")),
    ("disordered basin mean, L=6400", 0.051, 5e-4, lambda: rw(6400, "mu0_at_peak")),
    ("ordered basin mean, L=6400", 0.483, 5e-4, lambda: rw(6400, "mu1_at_peak")),
    # Sec. 4.3
    ("weight 0.2<phi<0.3, L=400 (15%)", 0.15, 0.005, lambda: _between(400)),
    ("weight 0.2<phi<0.3, L=3200 (4%)", 0.04, 0.005, lambda: _between(3200)),
    ("weight 0.2<phi<0.3, L=6400 (0.7%)", 0.007, 0.0005, lambda: _between(6400)),
    *[(f"beta*Delta f*L, L={L}", v, 0.005, (lambda L=L: barrier_L(L)))
      for L, v in zip(SIZES, [0.29, 0.57, 1.14, 2.29, 4.58])],
    *[(f"p_ord(h_t), L={L}", v, 0.0005, (lambda L=L: rw(L, "p_ord_at_ht")))
      for L, v in zip(SIZES, [0.605, 0.568, 0.559, 0.545, 0.562])],
    ("p_ord error, L=3200", 0.011, 0.0005, lambda: rw(3200, "p_ord_at_ht_err")),
    ("p_ord error, L=6400", 0.017, 0.0005, lambda: rw(6400, "p_ord_at_ht_err")),
    *[(f"L beta(h_eq-h_t), L={L}", v, 0.005, (lambda L=L: L * rw(L, "h_eq_minus_ht")))
      for L, v in zip(SIZES, [-6.73, -4.31, -3.65, -2.54, -3.22])],
    ("hysteresis up jump, L=3200", 1.276, 5e-4, lambda: _hyst_jump(3200, "up")),
    ("hysteresis up jump, L=6400", 1.276, 5e-4, lambda: _hyst_jump(6400, "up")),
    ("hysteresis down jump, L=3200", 1.228, 5e-4, lambda: _hyst_jump(3200, "down")),
    ("hysteresis down jump, L=6400", 1.228, 5e-4, lambda: _hyst_jump(6400, "down")),
    *[(f"U4 min (reweighted), L={L}", v, 0.005, (lambda L=L: rw(L, "U4_min")))
      for L, v in zip([800, 1600, 3200, 6400], [-0.10, -0.43, -1.36, -4.48])],
    ("U4 min at L=400 is not negative", True, None, lambda: bool(rw(400, "U4_min") > 0)),
    # sizes used in the figures
    ("sizes in field scan (Figs. 3, 7a)", SIZES, None, obs_sizes),
    ("sizes in long runs (Figs. 4, 5, 7b)", SIZES, None, hist_files),
    ("sizes in hysteresis (Fig. 6)", [3200, 6400], None, hyst_sizes),
]


def report():
    print("=" * 78)
    print("Recomputed Monte Carlo quantities")
    print("=" * 78)
    for L in SIZES:
        n, w = fwhm_points(L)
        print(f"L={L:5d}  chi_max/L(rw)={rw(L,'chi_max_over_L'):.4f}"
              f"+-{rw(L,'chi_max_over_L_err'):.4f}  ratio rw/uni={ratio_rw_uniform(L):.3f}"
              f"  h_pk-h_t={rw(L,'h_peak_minus_ht'):+.2e}+-{rw(L,'h_peak_err'):.1e}"
              f"  FWHM={w:.2e} ({n} uniform pts)  U4min(rw)={rw(L,'U4_min'):.3f}"
              f"  U4min(any)={u4min_any(L):.3f}  crossings={crossings(L)}"
              f"  frac_ord(h_t)={frac_ordered_ht(L):.3f}  bDf*L={barrier_L(L):.2f}")
    print("local slopes:", [round(local_slope(a, b), 3) for a, b in zip(SIZES, SIZES[1:])])
    print("global p (5 sizes):", round(global_p(), 3))
    print("sizes: observables", obs_sizes(), " histograms", hist_files(),
          " hysteresis", hyst_sizes())


def check():
    bad = 0
    print("=" * 78)
    print("Claims in the manuscript vs data")
    print("=" * 78)
    for label, printed, tol, fn in CLAIMS:
        val = fn()
        ok = (val == printed) if tol is None else abs(val - printed) <= tol
        bad += not ok
        print(f"[{'OK ' if ok else 'BAD'}] {label:55s} printed={printed!s:>10}  data={val!s:.10}")
    print(f"{len(CLAIMS) - bad}/{len(CLAIMS)} claims reproduced")
    return bad


if __name__ == "__main__":
    report()
    sys.exit(1 if check() else 0)
