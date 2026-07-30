"""
run_powerlaw.py
===============
Sweep alpha across the three regimes and build the tricritical figure:
  fig_powerlaw.pdf  (3 panels)
     (a) h_t(alpha): transition field vs alpha, with the mean-field plateau
         (alpha<1) and the rise toward short-range (alpha>2).
     (b) order-of-transition indicator: susceptibility-peak exponent p in
         chi_max ~ L^p, per alpha (p~1 first order, p<1 continuous).
     (c) Binder-minimum depth (1 - U4^min) vs alpha at fixed largest L; a deep,
         L-growing dip flags first order.
  powerlaw_summary.json

Usage:
  python3 run_powerlaw.py --mode fast     # laptop demo (small L)
  python3 run_powerlaw.py --mode full     # publication (L up to 1024)
"""
import numpy as np
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nkpotts.powerlaw_mc import kernel, kac_norm_from_kernel, run_one, moments, Q, J, T

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7, "savefig.bbox": "tight"})
H_T_MF = 1.2530467500626585   # alpha<1 mean-field plateau (J=1)


def locate_transition(alpha, L, h_lo, h_hi, nh, burn, nmeas, freq, seed):
    beta = 1.0/T
    kap = kernel(L, alpha); Nalpha = kac_norm_from_kernel(kap)
    haxis = np.linspace(h_lo, h_hi, nh)
    chi = np.empty(nh); u4 = np.empty(nh); M = np.empty(nh)
    for k, h in enumerate(haxis):
        m = run_one(L, h, kap, Nalpha, Q, J, beta, burn, nmeas, freq,
                    seed + k + 1000*L + 100000*int(alpha*100))
        m1, c, _, u = moments(m, L, beta)
        chi[k] = c; u4[k] = u; M[k] = m1
    kpk = int(np.argmax(chi))
    return haxis, chi, u4, M, haxis[kpk], chi[kpk], u4.min()


def refine_window(alpha):
    """Rough transition window per alpha (from smoke-test trends)."""
    if alpha <= 1.0:
        return 1.05, 1.55
    elif alpha <= 1.5:
        return 1.2, 1.95
    elif alpha <= 2.0:
        return 1.4, 2.3
    else:
        return 1.6, 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fast", choices=["fast", "full"])
    args = ap.parse_args()

    if args.mode == "fast":
        alphas = [0.3, 0.6, 1.0, 1.3, 1.6, 2.0]
        Ls = [128, 256, 512]
        burn, nmeas, freq, nh = 2500, 3000, 5, 12
    else:
        alphas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]
        Ls = [256, 512, 1024]
        burn, nmeas, freq, nh = 8000, 12000, 5, 26

    ht = []; pexp = []; dipL = []; dip_all = {}
    for alpha in alphas:
        h_lo, h_hi = refine_window(alpha)
        chimax_L = []; u4min_L = []; ht_L = []
        for L in Ls:
            haxis, chi, u4, M, hpk, cmax, u4min = locate_transition(
                alpha, L, h_lo, h_hi, nh, burn, nmeas, freq, seed=13)
            chimax_L.append(cmax); u4min_L.append(u4min); ht_L.append(hpk)
        ht.append(ht_L[-1])
        # chi_max ~ L^p
        p = np.polyfit(np.log(Ls), np.log(chimax_L), 1)[0]
        pexp.append(p)
        dipL.append(1.0 - u4min_L[-1])   # Binder dip depth at largest L
        dip_all[alpha] = {"chimax_L": chimax_L, "u4min_L": u4min_L, "ht_L": ht_L,
                          "p": float(p)}
        print(f"alpha={alpha:.2f}  h_t~{ht_L[-1]:.3f}  chi_max~L^{p:.2f}  "
              f"Binder dip(1-U4min)={1.0-u4min_L[-1]:.3f}")

    summary = {"alphas": alphas, "ht": ht, "p_exponent": pexp,
               "binder_dip": dipL, "detail": dip_all, "Ls": Ls,
               "h_t_meanfield_plateau": H_T_MF}
    with open("powerlaw_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # tidy CSV for re-plotting
    import pandas as _pd
    _pd.DataFrame({"alpha": alphas, "h_t": ht, "p_chi_exponent": pexp,
                   "binder_dip": dipL}).to_csv("data_powerlaw.csv", index=False)
    # per-L detail
    det_rows = []
    for al in alphas:
        d = dip_all[al]
        for L, cm, u in zip(Ls, d["chimax_L"], d["u4min_L"]):
            det_rows.append(dict(alpha=al, L=L, chi_max=cm, U4_min=u))
    _pd.DataFrame(det_rows).to_csv("data_powerlaw_perL.csv", index=False)

    # -------- figure --------
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    a = np.array(alphas)

    ax[0].plot(a, ht, "o-", color="k", ms=5)
    ax[0].axhline(H_T_MF, color="crimson", ls="--", lw=1,
                  label=r"MF plateau $h_t=1.253$")
    ax[0].axvspan(0, 1, color="0.9")
    ax[0].set_xlabel(r"$\alpha$"); ax[0].set_ylabel(r"$h_t(\alpha)$")
    ax[0].legend(fontsize=7, frameon=False); ax[0].set_title("(a) transition field")

    ax[1].plot(a, pexp, "s-", color="#1f77b4", ms=5)
    ax[1].axhline(1.0, color="crimson", ls="--", lw=1, label="first order ($p=1$)")
    ax[1].axhline(0.5, color="gray", ls=":", lw=1, label="MF-continuous ($p=1/2$)")
    ax[1].set_xlabel(r"$\alpha$"); ax[1].set_ylabel(r"$p$ in $\chi_{\max}\sim L^{p}$")
    ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("(b) order indicator")

    ax[2].plot(a, dipL, "^-", color="#2ca02c", ms=5)
    ax[2].set_xlabel(r"$\alpha$"); ax[2].set_ylabel(r"$1-U_4^{\min}$ (dip depth)")
    ax[2].axvspan(0, 1, color="0.9")
    ax[2].set_title("(c) Binder dip")

    fig.tight_layout()
    fig.savefig("fig_powerlaw.pdf"); fig.savefig("fig_powerlaw.png", dpi=200)
    print("wrote fig_powerlaw.pdf and powerlaw_summary.json")


if __name__ == "__main__":
    main()
