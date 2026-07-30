#!/usr/bin/env python3
"""
Figure 4: susceptibility scaling.

(a) chi_max vs L on log-log, against the first-order volume law.
(b) the parameter-free amplitude test: chi_max/L vs 1/L against the exact
    asymptote beta*(Delta phi)^2/4 = 1/16, which is fixed with no free parameter
    by the exactly known jump Delta phi = 1/2 at q = 3.

Two field-grid protocols are distinguished, because they are not equivalent:
the production scan uses a uniform grid (dh = 7.5e-3) for every L, which no
longer resolves the peak once its width ~1/L falls below dh; the refined runs
sample a narrow window around h_t with dh ~ 1/L.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UNIFORM = "data/data_observables.csv"
REFINED = "data/data_chi_peak_refined_all.csv"
AMP = 1.0 * (0.5 ** 2) / 4.0          # beta (Delta phi)^2 / 4 = 1/16


def peak_table(path):
    d = pd.read_csv(path)
    return {int(L): g.chi.max() for L, g in d.groupby("L")}


def main():
    uni = peak_table(UNIFORM)
    ref = peak_table(REFINED)

    Lu = np.array(sorted(uni))
    Cu = np.array([uni[L] for L in Lu])
    Lr = np.array(sorted(ref))
    Cr = np.array([ref[L] for L in Lr])

    # best available: refined where present, uniform otherwise
    best = {**uni, **ref}
    Lb = np.array(sorted(best))
    Cb = np.array([best[L] for L in Lb])

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.0))

    # ---- (a) the parameter-free amplitude test (refined protocol only) ----
    a = ax[0]
    a.axhline(AMP, ls="--", color="k", lw=1.2)
    a.text(2.4e-3, AMP * 1.012, r"$\beta(\Delta\phi)^2/4=1/16$",
           fontsize=9, va="bottom", ha="right")
    a.plot(1.0 / Lr, Cr / Lr, "s-", color="crimson", ms=7, lw=1.3)
    for L in Lr:
        a.annotate(f"$L={L}$", xy=(1.0 / L, ref[L] / L),
                   xytext=(4, -11), textcoords="offset points", fontsize=7.5)
    a.set_xlim(-1.2e-4, 2.75e-3)
    a.set_ylim(0.033, 0.066)
    a.set_xlabel(r"$1/L$")
    a.set_ylabel(r"$\chi_{\max}/L$")
    a.set_title("(a) amplitude vs exact asymptote")

    # ---- (b) why the refined grid is necessary ----
    b = ax[1]
    ratio = np.array([ref[L] / uni[L] for L in Lr])
    b.axhline(1.0, color="0.6", lw=1.0, ls=":")
    b.semilogx(Lr, ratio, "o-", color="crimson", ms=7, lw=1.3)
    b.set_xlabel(r"$L$")
    b.set_ylabel(r"$\chi_{\max}^{\rm refined}\,/\,\chi_{\max}^{\rm uniform}$")
    b.set_title("(b) under-resolution of the uniform grid")
    b.set_ylim(0.93, 1.46)
    for L, r_ in zip(Lr, ratio):
        b.annotate(f"{r_:.2f}", xy=(L, r_), xytext=(0, 7),
                   textcoords="offset points", ha="center", fontsize=7.5)

    fig.tight_layout()
    fig.savefig("fig_chi_scaling.pdf")
    fig.savefig("fig_chi_scaling.png", dpi=160)

    print("chi_max/L (refined):",
          ", ".join(f"L={L}: {ref[L]/L:.4f}" for L in Lr))
    print("target:", AMP)
    print("wrote fig_chi_scaling.pdf/.png")


if __name__ == "__main__":
    main()
