"""
replot_all.py
=============
Regenerate EVERY figure in the paper from the exported data files ONLY.
No Monte Carlo, no torch, no exact solver is run here -- this reads CSV/NPZ and
draws. Edit styling freely (colors, fonts, sizes) and re-run; it is fast.

Required data files (produced by the pipeline; see the mapping in README):
  data_theory_landscape.csv    <- theory_exact.py
  phase_diagram_data.npz       <- theory_exact.py
  data_observables.csv         <- run_production_mc.py   (M, chi, Cv, U4)
  data_chi_scaling.csv         <- run_production_mc.py
  data_histograms.csv          <- run_production_mc.py   (binned P(phi))
  data_hysteresis.csv          <- run_production_mc.py
  data_gnn_binned.csv          <- gnn_corrected.py
  data_powerlaw.csv            <- run_powerlaw.py
  data_powerlaw_perL.csv       <- run_powerlaw.py

Any file that is missing is skipped with a notice, so you can re-plot whatever
subset you have. Run:  python3 replot_all.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7, "savefig.bbox": "tight"})
COLORS = {400: "#1f77b4", 800: "#2ca02c", 1600: "#ff7f0e",
          3200: "#d62728", 6400: "#9467bd", 12800: "#8c564b"}
H_T = 1.2530467500626585

# Search both the current directory and ./data/ so the script works whether it
# is invoked from the repo root (data lives in data/) or from a workdir where
# a fresh run just wrote the CSVs alongside it.
_SEARCH_DIRS = [".", "data"]


def _resolve(fn):
    for d in _SEARCH_DIRS:
        p = os.path.join(d, fn)
        if os.path.exists(p):
            return p
    return None


def have(fn):
    p = _resolve(fn)
    if p is None:
        print(f"  [skip] {fn} not found in {_SEARCH_DIRS}")
        return False
    return True


def _load_csv(fn):
    return pd.read_csv(_resolve(fn))


def _load_npz(fn):
    return np.load(_resolve(fn))


# ---------------------------------------------------------------- theory landscape
def plot_landscape():
    if not have("data_theory_landscape.csv"):
        return
    df = _load_csv("data_theory_landscape.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    cols = [c for c in df.columns if c.startswith("f_h")]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(cols)))
    for c, col in zip(cols, colors):
        ax.plot(df["phi"], df[c], lw=1.6, color=col, label=c.replace("f_h", "h="))
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$\beta[f(\phi)-f(0)]$")
    ax.set_ylim(-0.05, 0.08); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("fig_theory_landscape.pdf")
    print("  wrote fig_theory_landscape.pdf")


# ---------------------------------------------------------------- phase diagram
def plot_phase():
    if not have("phase_diagram_data.npz"):
        return
    d = _load_npz("phase_diagram_data.npz")
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.plot(d["J"], d["sp_up"], "--", color="crimson", lw=1.3, label=r"$h_{sp}^+$")
    ax.plot(d["J"], d["ht"], "-", color="k", lw=1.8, label=r"$h_t(J)$")
    ax.plot(d["J"], d["sp_low"], ":", color="royalblue", lw=1.3, label=r"$h_{sp}^-$")
    ax.fill_between(d["J"], d["sp_low"], d["sp_up"], color="0.9")
    ax.set_xlabel(r"$J/k_BT$"); ax.set_ylabel(r"$h/k_BT$")
    ax.legend(fontsize=8, frameon=False)
    jj = d["jump"][np.isfinite(d["jump"])]
    dev = float(np.max(np.abs(jj - 0.5))) if jj.size else 1e-6
    ax.text(0.97, 0.03, r"$\Delta\phi_t=\frac{1}{2}$ along the $q{=}3$ line" + "\n"
            + rf"(deviation $< {max(dev,1e-6):.0e}$)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig.tight_layout(); fig.savefig("fig_phase_diagram.pdf")
    print("  wrote fig_phase_diagram.pdf")


# ---------------------------------------------------------------- observables
def plot_observables():
    if not have("data_observables.csv"):
        return
    df = _load_csv("data_observables.csv")
    Ls = sorted(df.L.unique())
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for L in Ls:
        s = df[df.L == L]; c = COLORS.get(L, "k")
        ax[0].errorbar(s.h, s.M, yerr=s.M_err, fmt="-o", ms=3, lw=1, color=c, label=f"L={L}", capsize=2)
        ax[1].errorbar(s.h, s.chi, yerr=s.chi_err, fmt="-o", ms=3, lw=1, color=c, capsize=2)
        ax[2].errorbar(s.h, s.Cv, yerr=s.Cv_err, fmt="-o", ms=3, lw=1, color=c, capsize=2)
    for a in ax:
        a.axvline(H_T, color="0.5", ls="--", lw=0.8); a.set_xlabel("h")
    ax[0].set_ylabel("M"); ax[1].set_ylabel(r"$\chi$"); ax[2].set_ylabel(r"$C_V$")
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("fig_observables.pdf")
    print("  wrote fig_observables.pdf")


# ---------------------------------------------------------------- binder
def plot_binder():
    if not have("data_observables.csv"):
        return
    df = _load_csv("data_observables.csv")
    Ls = sorted(df.L.unique())
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for L in Ls:
        s = df[df.L == L]
        yerr = s.U4_err if "U4_err" in s else None
        ax.errorbar(s.h, s.U4, yerr=yerr, fmt="-o", ms=3, lw=1,
                    color=COLORS.get(L, "k"), label=f"L={L}", capsize=2)
    ax.axvline(H_T, color="0.5", ls="--", lw=0.8, label="exact $h_t$")
    ax.set_xlabel("h"); ax.set_ylabel(r"$U_4$"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("fig_binder.pdf")
    print("  wrote fig_binder.pdf")


# ---------------------------------------------------------------- chi scaling
def plot_chi_scaling():
    if not have("data_chi_scaling.csv"):
        return
    df = _load_csv("data_chi_scaling.csv")
    L = df.L.values.astype(float); chimax = df.chi_max.values
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.loglog(L, chimax, "o", ms=6, color="k")
    b, a = np.polyfit(np.log(L), np.log(chimax), 1)
    xx = np.linspace(L.min(), L.max(), 50)
    ax.loglog(xx, np.exp(a)*xx**b, "-", color="crimson", label=rf"$\chi_{{\max}}\sim L^{{{b:.2f}}}$")
    ax.loglog(xx, chimax[0]*(xx/L[0]), ":", color="gray", label=r"$\propto L$")
    ax.set_xlabel("L"); ax.set_ylabel(r"$\chi_{\max}$"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig("fig_chi_scaling.pdf")
    print(f"  wrote fig_chi_scaling.pdf  (fitted exponent p={b:.3f})")


# ---------------------------------------------------------------- histograms
def plot_histograms():
    if not have("data_histograms.csv"):
        return
    df = _load_csv("data_histograms.csv")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for L in sorted(df.L.unique()):
        s = df[df.L == L]
        ax.step(s.phi, s.density, lw=1.4, color=COLORS.get(L, "k"), label=f"L={L}")
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$P(\phi)$ at $h_t$"); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig("fig_histograms.pdf")
    print("  wrote fig_histograms.pdf")


# ---------------------------------------------------------------- hysteresis
def plot_hysteresis():
    if not have("data_hysteresis.csv"):
        return
    df = _load_csv("data_hysteresis.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for L in sorted(df.L.unique()):
        up = df[(df.L == L) & (df.branch == "up")]
        dn = df[(df.L == L) & (df.branch == "down")]
        ax.plot(up.h, up.phi, "-o", ms=2, lw=1, color=COLORS.get(L, "k"), label=f"L={L} up")
        ax.plot(dn.h, dn.phi, "--s", ms=2, lw=1, color=COLORS.get(L, "k"), alpha=0.6, label=f"L={L} down")
    ax.axvline(H_T, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("h"); ax.set_ylabel(r"$\phi$"); ax.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig("fig_hysteresis.pdf")
    print("  wrote fig_hysteresis.pdf")


# ---------------------------------------------------------------- gnn
def plot_gnn():
    if not have("data_gnn_binned.csv"):
        return
    df = _load_csv("data_gnn_binned.csv")
    style = {"GNN": ("#1f77b4", "o"), "MLPg": ("#d62728", "s"), "Ridge-hand": ("#2ca02c", "^")}
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for name, (col, mk) in style.items():
        s = df[df.model == name]
        if s.empty:
            continue
        ax.errorbar(s.true_h_bin, s.pred_mean, yerr=s.pred_std, fmt="-"+mk, ms=4,
                    color=col, capsize=2, label=name)
        ax.fill_between(s.true_h_bin, s.pred_mean-s.pred_std, s.pred_mean+s.pred_std, color=col, alpha=0.12)
    ax.plot([0.4, 2.0], [0.4, 2.0], "k--", lw=1.2, label="ideal")
    ax.axvspan(1.15, 1.40, color="gray", alpha=0.15)
    ax.axvline(H_T, color="0.4", ls=":", lw=1)
    ax.set_xlabel(r"true $h$"); ax.set_ylabel(r"predicted $\hat h$"); ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig("fig_gnn.pdf")
    print("  wrote fig_gnn.pdf")


# ---------------------------------------------------------------- power law
def plot_powerlaw():
    if not have("data_powerlaw.csv"):
        return
    df = _load_csv("data_powerlaw.csv")
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    ax[0].plot(df.alpha, df.h_t, "o-", color="k", ms=5)
    ax[0].axhline(H_T, color="crimson", ls="--", lw=1, label=r"MF plateau")
    ax[0].axvspan(0, 1, color="0.9"); ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel(r"$h_t(\alpha)$")
    ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("(a) transition field")
    ax[1].plot(df.alpha, df.p_chi_exponent, "s-", color="#1f77b4", ms=5)
    ax[1].axhline(1.0, color="crimson", ls="--", lw=1, label="first order")
    ax[1].axhline(0.5, color="gray", ls=":", lw=1, label="MF-continuous")
    ax[1].set_xlabel(r"$\alpha$"); ax[1].set_ylabel(r"$p$ in $\chi_{\max}\sim L^p$")
    ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("(b) order indicator")
    ax[2].plot(df.alpha, df.binder_dip, "^-", color="#2ca02c", ms=5)
    ax[2].axvspan(0, 1, color="0.9"); ax[2].set_xlabel(r"$\alpha$")
    ax[2].set_ylabel(r"$1-U_4^{\min}$"); ax[2].set_title("(c) Binder dip")
    fig.tight_layout(); fig.savefig("fig_powerlaw.pdf")
    print("  wrote fig_powerlaw.pdf")


def plot_infobound():
    if not (have("data_infobound_opt.csv") and have("data_infobound_crb.csv")):
        return
    opt = _load_csv("data_infobound_opt.csv")
    crb = _load_csv("data_infobound_crb.csv")
    r2 = 1.0 - np.mean((opt.true_h - opt.opt_pred)**2)/np.var(opt.true_h)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))
    sc = ax[0].scatter(opt.true_h, opt.opt_pred, s=4, c=opt.Q, cmap="viridis", alpha=0.35, rasterized=True)
    ax[0].plot([0.4, 2.0], [0.4, 2.0], "k--", lw=1.2, label="ideal")
    ax[0].axvspan(1.15, 1.40, color="gray", alpha=0.15)
    ax[0].set_xlabel(r"true $h$"); ax[0].set_ylabel(r"Bayes-optimal $\hat h=E[h|Q]$")
    ax[0].set_title(rf"(a) optimal estimator, $R^2_{{\rm opt}}={r2:.3f}$")
    ax[0].legend(fontsize=8, frameon=False)
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label(r"$Q=\sum_a n_a^2$", fontsize=8)
    ax[1].plot(crb.h_grid, crb.std_opt, "o-", color="crimson", ms=4,
               label=r"CRB: $1/\sqrt{I(h)}$")
    ax[1].axvspan(1.15, 1.40, color="gray", alpha=0.15, label="coexistence")
    ax[1].axvline(H_T, color="0.4", ls=":", lw=1)
    ax[1].set_xlabel(r"true $h$"); ax[1].set_ylabel(r"optimal single-snapshot std$(\hat h)$")
    ax[1].set_title("(b) Cram\u00e9r\u2013Rao floor"); ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig("fig_infobound.pdf")
    print("  wrote fig_infobound.pdf")


if __name__ == "__main__":
    print("Re-plotting all figures from exported data files:")
    for fn in [plot_landscape, plot_phase, plot_observables, plot_binder,
               plot_chi_scaling, plot_histograms, plot_hysteresis, plot_gnn,
               plot_powerlaw, plot_infobound]:
        fn()
    print("Done.")
