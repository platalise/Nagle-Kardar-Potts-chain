#!/usr/bin/env python3
"""
make_paper_figures.py
=====================
Build every figure of the revised manuscript (Figs. 1-7) from the files in
data/ and from the closed forms.  No Monte Carlo is run here.

    Fig. 1  fig_theory_landscape.pdf   <- data/data_theory_landscape.csv
    Fig. 2  fig_phase_diagram.pdf      <- closed forms + ordered spinodal solved
                                          here (written to data/data_phase_diagram.csv)
    Fig. 3  fig_observables.pdf        <- data/data_observables.csv
    Fig. 4  fig_chi_scaling.pdf        <- data/data_observables.csv (uniform grid)
                                          data/data_chi_peak_refined_all.csv (refined)
    Fig. 5  fig_histograms.pdf         <- data/data_histograms_raw.npz
    Fig. 6  fig_hysteresis.pdf         <- data/data_hysteresis.csv
    Fig. 7  fig_binder.pdf             <- data/data_observables.csv

Usage (from the repository root):
    python scripts/make_paper_figures.py --outdir figures
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7,
                     "savefig.bbox": "tight", "legend.frameon": False})
COLORS = {400: "#1f77b4", 800: "#2ca02c", 1600: "#ff7f0e",
          3200: "#d62728", 6400: "#9467bd"}
Q = 3
BJ = 1.0
HT = 2.0 * (Q - 1) / (Q - 2) * np.log1p((Q - 2) * np.exp(-BJ))   # 1.2530467500...
HSP = Q * Q / (2.0 * np.exp(BJ) + Q - 2)                          # 1.3982616314...
PHIT = (Q - 2.0) / (Q - 1.0)
AMP = PHIT ** 2 / 4.0                                              # beta (Delta phi)^2/4
DATA = "data"


# ---------------------------------------------------------------------------
# closed forms and the master form (used for Fig. 2)
# ---------------------------------------------------------------------------
def bh_t(J, q):
    return 2.0 * (q - 1) / (q - 2) * np.log1p((q - 2) * np.exp(-J))


def bh_sp(J, q):
    return q * q / (2.0 * np.exp(J) + q - 2)


def _derivs(phi, bh, J, q):
    """f'(phi), f''(phi) of beta f from the master form (analytic)."""
    u = np.exp(J); c = u + q - 2; d = (u - 1) * (u + q - 1)
    A = bh * (q - 1) / (2 * q); k = bh / 2
    s = np.exp(k * phi)
    B = u * s + c / s
    Bp = k * (u * s - c / s)
    R = np.sqrt(B * B - 4 * d)
    pt = (q - 2) / (q - 1)
    f1 = A * (2 * phi - pt) - Bp / R
    f2 = 2 * A - k * k * B / R + B * Bp * Bp / R ** 3
    return f1, f2


def ordered_spinodal_curve(Js, q, guess):
    """Solve f'(phi)=f''(phi)=0, phi>0, by continuation in J."""
    out = []
    g = np.array(guess, float)
    for J in Js:
        sol, info, ier, _ = fsolve(lambda v: _derivs(v[0], v[1], J, q), g,
                                   full_output=True, xtol=1e-13)
        ok = ier == 1 and sol[0] > 1e-3 and np.max(np.abs(_derivs(*sol, J, q))) < 1e-10
        out.append((sol[0], sol[1]) if ok else (np.nan, np.nan))
        if ok:
            g = sol
    return np.array(out)


def _save(fig, outdir, name):
    fig.savefig(os.path.join(outdir, name + ".pdf"))
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=200)
    plt.close(fig)
    print("  wrote", os.path.join(outdir, name + ".pdf"))


# ---------------------------------------------------------------------------
def fig1(outdir):
    df = pd.read_csv(os.path.join(DATA, "data_theory_landscape.csv"))
    cols = [c for c in df.columns if c.startswith("f_h")]
    labels = {"f_h0.9000": r"$h=0.90$",
              "f_h1.2360": r"$h=h_{\rm sp}^{-}=1.236$",
              "f_h1.2530": r"$h=h_t=1.2530$",
              "f_h1.3000": r"$h=1.30$",
              "f_h1.3983": r"$h=h_{\rm sp}^{+}=1.3983$"}
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(cols)))
    for c, col in zip(cols, colors):
        lw = 2.2 if c == "f_h1.2530" else 1.4
        ax.plot(df["phi"], df[c], lw=lw, color=col, label=labels.get(c, c))
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(PHIT / 2, color="0.6", lw=0.6, ls=":")
    ax.text(PHIT / 2 + 0.01, 0.07, r"$\phi_t/2$", fontsize=8, color="0.35")
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\beta[f(\phi)-f(0)]$")
    ax.set_ylim(-0.05, 0.08)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, outdir, "fig_theory_landscape")


def fig2(outdir):
    Js = np.linspace(0.0, 2.0, 201)
    rows = {"J": Js}
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.7))

    # (a) q = 3
    ht3, sp3 = bh_t(Js, 3), bh_sp(Js, 3)
    osp3 = ordered_spinodal_curve(Js, 3, guess=(0.38, 2.7))
    rows.update(ht_q3=ht3, hsp_plus_q3=sp3, hsp_minus_q3=osp3[:, 1],
                phi_at_hsp_minus_q3=osp3[:, 0])
    a = ax[0]
    a.fill_between(Js, osp3[:, 1], sp3, color="0.9", lw=0)
    a.plot(Js, sp3, "--", color="crimson", lw=1.3, label=r"$h_{\rm sp}^{+}$ (closed form)")
    a.plot(Js, ht3, "-", color="k", lw=1.8, label=r"$h_t$ (closed form)")
    a.plot(Js, osp3[:, 1], ":", color="royalblue", lw=1.5,
           label=r"$h_{\rm sp}^{-}$ (numerical)")
    a.set_xlabel(r"$\beta J$")
    a.set_ylabel(r"$\beta h$")
    a.set_title(r"(a) $q=3$", fontsize=9)
    a.legend(fontsize=7.5, loc="upper right")
    a.text(0.03, 0.04, r"jump $\Delta\phi=\phi_t=1/2$ along the whole line",
           transform=a.transAxes, fontsize=7.5)

    # (b) general q
    colors = plt.cm.plasma(np.linspace(0.05, 0.75, 4))
    b = ax[1]
    for qv, c in zip([3, 4, 5, 8], colors):
        ht, sp = bh_t(Js, qv), bh_sp(Js, qv)
        rows[f"ht_q{qv}"] = ht
        rows[f"hsp_plus_q{qv}"] = sp
        b.plot(Js, ht, "-", lw=1.7, color=c,
               label=rf"$q={qv}$, $\phi_t={(qv - 2)}/{(qv - 1)}$")
        b.plot(Js, sp, "--", lw=0.9, color=c, alpha=0.7)
    b.set_xlabel(r"$\beta J$")
    b.set_ylabel(r"$\beta h$")
    b.set_title(r"(b) general $q$ (solid: $h_t$; dashed: $h_{\rm sp}^{+}$)",
                fontsize=9)
    b.legend(fontsize=7.5)
    fig.tight_layout()
    _save(fig, outdir, "fig_phase_diagram")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "data_phase_diagram.csv"),
                              index=False)
    j1 = np.argmin(np.abs(Js - 1.0))
    print(f"    q=3, bJ=1: h_sp- = {osp3[j1, 1]:.8f} at phi* = {osp3[j1, 0]:.8f};"
          f" NaNs in h_sp-: {int(np.isnan(osp3[:, 1]).sum())}")


def fig3(outdir):
    df = pd.read_csv(os.path.join(DATA, "data_observables.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.1))
    for L in sorted(df.L.unique()):
        s = df[df.L == L].sort_values("h")
        c = COLORS.get(int(L), "k")
        kw = dict(fmt="-o", ms=2.5, lw=0.9, color=c, capsize=1.5)
        ax[0].errorbar(s.h, s.M, yerr=s.M_err, label=f"$L={int(L)}$", **kw)
        ax[1].errorbar(s.h, s.chi, yerr=s.chi_err, **kw)
        ax[2].errorbar(s.h, s.Cv, yerr=s.Cv_err, **kw)
    for a, lab, yl in zip(ax, "abc", [r"$\langle\phi\rangle$", r"$\chi$", r"$C_V$"]):
        a.axvline(HT, color="0.5", ls="--", lw=0.8)
        a.set_xlabel(r"$\beta h$")
        a.set_ylabel(yl)
        a.set_title(f"({lab})", fontsize=9, loc="left")
    ax[0].legend(fontsize=7)
    ax[0].text(HT + 0.03, 0.03, r"$h_t$", fontsize=8, color="0.4")
    fig.tight_layout()
    _save(fig, outdir, "fig_observables")


def _peaks(path):
    d = pd.read_csv(path)
    out = {}
    for L, g in d.groupby("L"):
        i = g.chi.idxmax()
        out[int(L)] = (g.loc[i, "chi"], g.loc[i, "chi_err"], g.loc[i, "h"])
    return out


def fig4(outdir):
    uni = _peaks(os.path.join(DATA, "data_observables.csv"))
    rw = pd.read_csv(os.path.join(DATA, "data_chi_reweighted.csv")).set_index("L")
    Ls = np.array(sorted(rw.index))
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.7))

    a = ax[0]
    y = rw.loc[Ls, "chi_max_over_L"].values
    ye = rw.loc[Ls, "chi_max_over_L_err"].values
    a.axhline(AMP, ls="--", color="k", lw=1.1)
    a.text(2.6e-3, AMP + 6e-4, r"$\beta\phi_t^2/4=1/16$", fontsize=8.5,
           ha="right", va="bottom")
    a.errorbar(1.0 / Ls, y, yerr=ye, fmt="s-", color="crimson", ms=6, lw=1.2,
               capsize=2)
    for L, yy in zip(Ls, y):
        a.annotate(f"$L={L}$", xy=(1.0 / L, yy), xytext=(5, -11),
                   textcoords="offset points", fontsize=7.5)
    a.set_xlim(-1.0e-4, 2.75e-3)
    a.set_ylim(min(0.030, y.min() - 0.004), 0.066)
    a.set_xlabel(r"$1/L$")
    a.set_ylabel(r"$\chi_{\max}/L$")
    a.set_title("(a)", fontsize=9, loc="left")

    b = ax[1]
    ratio = np.array([rw.loc[L, "chi_max"] / uni[L][0] for L in Ls])
    rerr = ratio * np.sqrt(np.array([(rw.loc[L, "chi_max_err"] / rw.loc[L, "chi_max"]) ** 2
                                     + (uni[L][1] / uni[L][0]) ** 2 for L in Ls]))
    b.axhline(1.0, color="0.6", lw=1.0, ls=":")
    b.errorbar(Ls, ratio, yerr=rerr, fmt="o-", color="crimson", ms=6, lw=1.2,
               capsize=2)
    b.set_xscale("log")
    b.set_xticks(Ls)
    b.set_xticklabels([str(L) for L in Ls])
    b.minorticks_off()
    for L, r_ in zip(Ls, ratio):
        b.annotate(f"{r_:.2f}", xy=(L, r_), xytext=(0, 9),
                   textcoords="offset points", ha="center", fontsize=7.5)
    b.set_ylim(0.8, max(1.5, (ratio + rerr).max() + 0.12))
    b.set_xlabel(r"$L$")
    b.set_ylabel(r"$\chi_{\max}^{\rm rw}/\chi_{\max}^{\rm scan}$")
    b.set_title("(b)", fontsize=9, loc="left")
    fig.tight_layout()
    _save(fig, outdir, "fig_chi_scaling")


def fig5(outdir):
    df = pd.read_csv(os.path.join(DATA, "data_histograms_longrun.csv"))
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for L in [400, 1600, 6400]:
        s = df[df.L == L].sort_values("phi")
        ax.step(s.phi, s.density, where="mid", lw=1.4,
                color=COLORS.get(L, "k"), label=f"$L={L}$")
    ax.axvline(PHIT, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$P(\phi)$ at $h_t$")
    ax.legend(fontsize=7.5)
    ax.text(PHIT + 0.01, ax.get_ylim()[1] * 0.92, r"$\phi_t$", fontsize=8,
            color="0.35")
    fig.tight_layout()
    _save(fig, outdir, "fig_histograms")


def figS1(outdir):
    """SI figure: direct narrow-window scans are not ergodic at L >= 3200."""
    rf = pd.read_csv(os.path.join(DATA, "data_chi_peak_refined_all.csv"))
    cv = pd.read_csv(os.path.join(DATA, "data_reweighted_curves.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for a, L in zip(ax, [1600, 3200, 6400]):
        s = rf[rf.L == L].sort_values("h")
        c = cv[cv.L == L]
        c = c[(c.h >= s.h.min()) & (c.h <= s.h.max())]
        a.fill_between(L * (c.h - HT), (c.chi - c.chi_err) / L,
                       (c.chi + c.chi_err) / L, color="0.75", lw=0)
        a.plot(L * (c.h - HT), c.chi / L, "-", color="k", lw=1.2,
               label="long runs + reweighting")
        a.errorbar(L * (s.h - HT), s.chi / L, yerr=s.chi_err / L, fmt="o", ms=3,
                   color=COLORS[L], capsize=1.5, label=r"direct scan, $4\times10^4$ MCS/point")
        a.axhline(AMP, ls="--", color="0.5", lw=0.8)
        a.set_title(f"$L={L}$", fontsize=9)
        a.set_xlabel(r"$L\,\beta(h-h_t)$")
    ax[0].set_ylabel(r"$\chi/L$")
    ax[0].legend(fontsize=7, loc="lower center")
    fig.tight_layout()
    _save(fig, outdir, "figS1_direct_scan")


def fig6(outdir):
    df = pd.read_csv(os.path.join(DATA, "data_hysteresis.csv"))
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for L in sorted(df.L.unique()):
        c = COLORS.get(int(L), "k")
        up = df[(df.L == L) & (df.branch == "up")].sort_values("h")
        dn = df[(df.L == L) & (df.branch == "down")].sort_values("h")
        ax.plot(up.h, up.phi, "-o", ms=2, lw=1, color=c, label=f"$L={int(L)}$, up")
        ax.plot(dn.h, dn.phi, "--s", ms=2, lw=1, color=c, alpha=0.6,
                label=f"$L={int(L)}$, down")
    ax.axvline(HT, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel(r"$\beta h$")
    ax.set_ylabel(r"$\phi$ (single run)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, outdir, "fig_hysteresis")


def fig7(outdir):
    df = pd.read_csv(os.path.join(DATA, "data_observables.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(9.8, 3.6))
    a = ax[0]
    for L in sorted(df.L.unique()):
        s = df[df.L == L].sort_values("h")
        a.errorbar(s.h, s.U4, yerr=s.U4_err, fmt="-o", ms=2.5, lw=0.9,
                   color=COLORS.get(int(L), "k"), label=f"$L={int(L)}$",
                   capsize=1.5)
    a.axvline(HT, color="0.5", ls="--", lw=0.8, label=r"exact $h_t$")
    a.set_xlabel(r"$\beta h$")
    a.set_ylabel(r"$U_4$")
    a.set_title("(a) field scan", fontsize=9, loc="left")
    a.legend(fontsize=7, loc="lower right")

    b = ax[1]
    cv = pd.read_csv(os.path.join(DATA, "data_reweighted_curves.csv"))
    for L in sorted(cv.L.unique()):
        s_ = cv[cv.L == L].sort_values("h")
        x = L * (s_.h - HT)
        b.fill_between(x, s_.U4 - s_.U4_err, s_.U4 + s_.U4_err,
                       color=COLORS.get(int(L), "k"), alpha=0.25, lw=0)
        b.plot(x, s_.U4, "-", lw=1.3, color=COLORS.get(int(L), "k"),
               label=f"$L={int(L)}$")
    b.axvline(0.0, color="0.5", ls="--", lw=0.8)
    b.set_xlim(-120, 40)
    b.set_xlabel(r"$L\,\beta(h-h_t)$")
    b.set_ylabel(r"$U_4$")
    b.set_title(r"(b) long runs + reweighting", fontsize=9, loc="left")
    b.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    _save(fig, outdir, "fig_binder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--only", nargs="*", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    figs = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: figS1}
    for k, fn in figs.items():
        if args.only is None or k in args.only:
            fn(args.outdir)


if __name__ == "__main__":
    main()
