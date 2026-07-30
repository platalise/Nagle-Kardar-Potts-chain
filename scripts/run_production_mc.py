"""
run_production_mc.py
====================
Publication-scale runs for the corrected 1D Potts+herding model.
Generates:
  data_observables.csv         (M, chi, Cv, U4 with jackknife errors, all L)
  fig_observables.pdf          (M, chi, Cv vs h)
  fig_binder.pdf               (U4 crossing -> weak first-order)
  fig_histograms.pdf           (P(phi) at h_t for growing L -> emerging bimodality)
  fig_hysteresis.pdf           (up/down sweeps -> metastability)
  fig_chi_scaling.pdf          (chi_max vs L: first-order gives chi_max ~ L)
  production_summary.json

Usage (see run_fullscale.sh for HPC settings):
  python3 run_production_mc.py --mode grid   --Ls 400 800 1600 3200
  python3 run_production_mc.py --mode hist    --Ls 400 1600 6400
  python3 run_production_mc.py --mode hyst    --Ls 3200
  python3 run_production_mc.py --mode all-fast   # laptop demo
"""
import numpy as np
import pandas as pd
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from nkpotts.mc_corrected import measure_grid, hysteresis, histogram_phi, run_one, jackknife, Q, J, T

H_T_EXACT = 1.2530467500626585
H_SP_EXACT = 1.3982616314726728

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7,
                     "figure.dpi": 120, "savefig.bbox": "tight"})
COLORS = {400: "#1f77b4", 800: "#2ca02c", 1600: "#ff7f0e",
          3200: "#d62728", 6400: "#9467bd", 12800: "#8c564b"}


def _grid_worker(args):
    L, h_axis, burn, nmeas, freq = args
    return measure_grid(L, h_axis, burn, nmeas, freq)


def run_grid(Ls, burn, nmeas, freq, nh_fine=41):
    h_axis = np.concatenate([
        np.linspace(0.4, 1.10, 8, endpoint=False),
        np.linspace(1.10, 1.40, nh_fine),
        np.linspace(1.42, 2.0, 8),
    ])
    tasks = [(L, h_axis, burn, nmeas, freq) for L in Ls]
    with Pool(min(len(Ls), max(1, cpu_count()-1))) as p:
        allrows = p.map(_grid_worker, tasks)
    rows = [r for sub in allrows for r in sub]
    df = pd.DataFrame(rows).sort_values(["L", "h"])
    df.to_csv("data_observables.csv", index=False)
    return df


def plot_observables(df, Ls):
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for L in Ls:
        s = df[df.L == L]
        c = COLORS.get(L, "k")
        ax[0].errorbar(s.h, s.M, yerr=s.M_err, fmt="-o", ms=3, lw=1, color=c, label=f"L={L}", capsize=2)
        ax[1].errorbar(s.h, s.chi, yerr=s.chi_err, fmt="-o", ms=3, lw=1, color=c, capsize=2)
        ax[2].errorbar(s.h, s.Cv, yerr=s.Cv_err, fmt="-o", ms=3, lw=1, color=c, capsize=2)
    for a in ax:
        a.axvline(H_T_EXACT, color="0.5", ls="--", lw=0.8)
    ax[0].set_ylabel(r"$M$"); ax[1].set_ylabel(r"$\chi$"); ax[2].set_ylabel(r"$C_V$")
    for a in ax:
        a.set_xlabel(r"$h$")
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].text(H_T_EXACT+0.02, 0.05, r"$h_t$", fontsize=8, color="0.4")
    fig.tight_layout(); fig.savefig("fig_observables.pdf"); fig.savefig("fig_observables.png", dpi=200)


def plot_binder(df, Ls):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for L in Ls:
        s = df[df.L == L]
        ax.errorbar(s.h, s.U4, yerr=s.U4_err, fmt="-o", ms=3, lw=1,
                    color=COLORS.get(L, "k"), label=f"L={L}", capsize=2)
    ax.axvline(H_T_EXACT, color="0.5", ls="--", lw=0.8, label=r"exact $h_t$")
    ax.set_xlabel(r"$h$"); ax.set_ylabel(r"$U_4$")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("fig_binder.pdf"); fig.savefig("fig_binder.png", dpi=200)


def run_histograms(Ls, burn, nmeas, freq):
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    out = {}
    raw = {}
    binned_rows = []
    for L in Ls:
        m = histogram_phi(L, H_T_EXACT, burn, nmeas, freq, seed=2024+L)
        counts, edges = np.histogram(m, bins=40, density=True)
        centers = 0.5*(edges[:-1]+edges[1:])
        ax.hist(m, bins=40, density=True, histtype="step", lw=1.4,
                color=COLORS.get(L, "k"), label=f"L={L}")
        out[L] = [float(m.mean()), float(m.std())]
        raw[f"phi_L{L}"] = m
        for c, d in zip(centers, counts):
            binned_rows.append(dict(L=L, phi=c, density=d))
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$P(\phi)$ at $h_t$")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("fig_histograms.pdf"); fig.savefig("fig_histograms.png", dpi=200)
    # DATA EXPORT
    np.savez("data_histograms_raw.npz", **raw)      # raw phi samples per L
    pd.DataFrame(binned_rows).to_csv("data_histograms.csv", index=False)  # binned P(phi)
    return out


def run_hyst(Ls, sweeps_per_step, nh=61):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    rows = []
    for L in Ls:
        hu, mu, hd, md = hysteresis(L, 1.10, 1.42, nh, sweeps_per_step, seed=11+L)
        ax.plot(hu, mu, "-o", ms=2, lw=1, color=COLORS.get(L, "k"), label=f"L={L} up")
        ax.plot(hd, md, "--s", ms=2, lw=1, color=COLORS.get(L, "k"), alpha=0.6, label=f"L={L} down")
        for h, m in zip(hu, mu):
            rows.append(dict(L=L, branch="up", h=h, phi=m))
        for h, m in zip(hd, md):
            rows.append(dict(L=L, branch="down", h=h, phi=m))
    ax.axvline(H_T_EXACT, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel(r"$h$"); ax.set_ylabel(r"$\phi$ (single run)")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig("fig_hysteresis.pdf"); fig.savefig("fig_hysteresis.png", dpi=200)
    pd.DataFrame(rows).to_csv("data_hysteresis.csv", index=False)   # DATA EXPORT


def plot_chi_scaling(df, Ls):
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    chimax = np.array([df[df.L == L].chi.max() for L in Ls])
    Ls_a = np.array(Ls, float)
    ax.loglog(Ls_a, chimax, "o", ms=6, color="k")
    # fit slope
    b, a = np.polyfit(np.log(Ls_a), np.log(chimax), 1)
    xx = np.linspace(Ls_a.min(), Ls_a.max(), 50)
    ax.loglog(xx, np.exp(a)*xx**b, "-", color="crimson",
              label=rf"$\chi_{{\max}}\sim L^{{{b:.2f}}}$")
    ax.loglog(xx, chimax[0]*(xx/Ls_a[0]), ":", color="gray", label=r"$\propto L$ (first order)")
    ax.set_xlabel(r"$L$"); ax.set_ylabel(r"$\chi_{\max}$")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig("fig_chi_scaling.pdf"); fig.savefig("fig_chi_scaling.png", dpi=200)
    pd.DataFrame({"L": Ls_a, "chi_max": chimax}).to_csv("data_chi_scaling.csv", index=False)  # DATA EXPORT
    return float(b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all-fast",
                    choices=["grid", "hist", "hyst", "chi", "all-fast", "all-full"])
    ap.add_argument("--Ls", type=int, nargs="+", default=None)
    ap.add_argument("--burn", type=int, default=None)
    ap.add_argument("--nmeas", type=int, default=None)
    ap.add_argument("--freq", type=int, default=10)
    args = ap.parse_args()

    summary = {"h_t_exact": H_T_EXACT, "h_sp_exact": H_SP_EXACT}

    if args.mode == "all-fast":
        Ls = args.Ls or [400, 800, 1600]
        df = run_grid(Ls, burn=args.burn or 3000, nmeas=args.nmeas or 4000, freq=args.freq, nh_fine=25)
        plot_observables(df, Ls); plot_binder(df, Ls)
        summary["chi_scaling_exponent"] = plot_chi_scaling(df, Ls)
        summary["hist_mean_std"] = run_histograms([400, 1600], 3000, 6000, 3)
        run_hyst([1600], sweeps_per_step=40)
    elif args.mode == "all-full":
        Ls = args.Ls or [400, 800, 1600, 3200, 6400]
        df = run_grid(Ls, burn=args.burn or 20000, nmeas=args.nmeas or 40000, freq=args.freq, nh_fine=51)
        plot_observables(df, Ls); plot_binder(df, Ls)
        summary["chi_scaling_exponent"] = plot_chi_scaling(df, Ls)
        summary["hist_mean_std"] = run_histograms([400, 1600, 6400], 20000, 80000, 5)
        run_hyst([6400], sweeps_per_step=200)
    elif args.mode == "grid":
        Ls = args.Ls or [400, 800, 1600, 3200]
        df = run_grid(Ls, args.burn or 20000, args.nmeas or 40000, args.freq)
        plot_observables(df, Ls); plot_binder(df, Ls)
        summary["chi_scaling_exponent"] = plot_chi_scaling(df, Ls)
    elif args.mode == "hist":
        Ls = args.Ls or [400, 1600, 6400]
        summary["hist_mean_std"] = run_histograms(Ls, args.burn or 20000, args.nmeas or 80000, 5)
    elif args.mode == "hyst":
        Ls = args.Ls or [3200]
        run_hyst(Ls, sweeps_per_step=200)

    with open("production_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
