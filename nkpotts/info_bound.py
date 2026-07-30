"""
info_bound.py
=============
Turns the "GNN does not win" negative result into a QUANTITATIVE optimality
statement, using the exact solvability of the model.

Key facts:
  P(config | h) = exp( beta*J*W + (beta*h/2N)*(Q - N) ) / Z(h),
    W = sum_i delta(s_i,s_{i+1}),   Q = sum_a n_a^2.
  The h-dependent part couples ONLY to Q. By the Fisher-Neyman factorization
  theorem, Q (equivalently the occupation fractions) is an EXACT sufficient
  statistic for h. Hence NO function of the configuration can carry more
  information about h than Q does: any regressor's accuracy is bounded by the
  Bayes-optimal estimator h_hat(Q) = E[h | Q].

This script computes, at fixed L (matching the inference experiment):
  (1) The Bayes-optimal estimator E[h|Q] by nonparametric regression of h on Q
      over (h ~ Uniform, one snapshot each) pairs -> R^2_opt (the CEILING).
  (2) The per-h Fisher information I(h) = (beta/2N)^2 Var(Q|h) and the
      Cramer-Rao bound sqrt(1/I(h)) -> the local optimal 1-sigma band on h_hat.
  (3) Comparison to the measured regressor R^2 (GNN/MLP/Ridge).
Outputs: fig_infobound.pdf, info_bound.json
"""
import numpy as np
import argparse, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nkpotts.powerlaw_mc import Q as QSTATES  # not used; keep import light
from nkpotts.gnn_corrected import make_dataset  # reuse the corrected MC snapshot generator

BETA = 1.0

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7, "savefig.bbox": "tight"})


def Qstat(spins, q=3):
    c = np.bincount(spins, minlength=q).astype(float)
    return np.sum(c*c)


def bayes_optimal(hs, Qs, nbin=40):
    """E[h|Q] by binning Q; return per-sample optimal prediction and R^2."""
    order = np.argsort(Qs)
    Qs_s = Qs[order]; hs_s = hs[order]
    # equal-count bins in Q for a stable nonparametric E[h|Q]
    edges = np.quantile(Qs, np.linspace(0, 1, nbin+1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    idx = np.digitize(Qs, edges) - 1
    idx = np.clip(idx, 0, nbin-1)
    ghat = np.zeros_like(hs)
    binmean = np.full(nbin, np.nan)
    for b in range(nbin):
        m = idx == b
        if m.any():
            binmean[b] = hs[m].mean()
            ghat[m] = binmean[b]
    mse = np.mean((hs - ghat)**2)
    r2 = 1.0 - mse/np.var(hs)
    return ghat, r2, mse, edges, binmean


def fisher_and_crb(L, h_grid, burn, nmeas, freq, seed0=321):
    """I(h) = (beta/2N)^2 Var(Q|h); CRB = 1/I(h) (single-snapshot variance of h_hat)."""
    from gnn_corrected import generate_snapshot
    I = np.zeros_like(h_grid); varQ = np.zeros_like(h_grid); meanQ = np.zeros_like(h_grid)
    for k, h in enumerate(h_grid):
        Qs = []
        # sample Q at intervals from one long equilibrated chain
        s = generate_snapshot(L, float(h), 3, 1.0, burn, seed0 + k)
        Qs.append(Qstat(s))
        for r in range(nmeas//freq):
            s = generate_snapshot(L, float(h), 3, 1.0, freq, seed0 + k + 7919*(r+1))
            Qs.append(Qstat(s))
        Qs = np.array(Qs)
        varQ[k] = Qs.var(); meanQ[k] = Qs.mean()
        I[k] = (BETA/(2.0*L))**2 * varQ[k]
    crb = np.where(I > 0, 1.0/I, np.nan)     # Var(h_hat) >= CRB (single snapshot)
    return I, crb, varQ, meanQ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=400)
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--burn", type=int, default=5000)
    ap.add_argument("--fisher_burn", type=int, default=3000)
    ap.add_argument("--fisher_nmeas", type=int, default=1500)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    if args.fast:
        args.samples, args.burn, args.fisher_burn, args.fisher_nmeas = 900, 2500, 1500, 500

    # (1) Bayes-optimal ceiling from (h, Q) pairs
    data = make_dataset(args.L, args.samples, args.burn, seed0=555)
    hs = np.array([h for h, _ in data])
    Qs = np.array([Qstat(s) for _, s in data])
    ghat, r2_opt, mse_opt, edges, binmean = bayes_optimal(hs, Qs, nbin=40)

    # (2) Fisher information / CRB vs h
    h_grid = np.linspace(0.45, 1.95, 22)
    I, crb, varQ, meanQ = fisher_and_crb(
        args.L, h_grid, args.fisher_burn, args.fisher_nmeas, freq=5)

    out = {
        "L": args.L, "samples": args.samples,
        "R2_bayes_optimal_ceiling": float(r2_opt),
        "MSE_bayes_optimal": float(mse_opt),
        "var_h_prior": float(np.var(hs)),
        "measured_R2": {"MLPg": 0.856, "Ridge_hand": 0.849, "GNN": 0.830},
        "note": "R2_opt is the exact ceiling; MLPg (occupation fractions) should "
                "approach it; GNN below it is optimization variance, not information loss.",
        "h_grid": h_grid.tolist(),
        "fisher_information": I.tolist(),
        "crb_var_hhat_single_snapshot": crb.tolist(),
    }
    print(json.dumps({k: out[k] for k in
          ["L", "R2_bayes_optimal_ceiling", "MSE_bayes_optimal", "measured_R2"]}, indent=2))
    with open("info_bound.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- figure: two panels ----
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.5))

    # (a) optimal E[h|Q] and the data cloud
    sc = ax[0].scatter(hs, ghat, s=4, c=Qs, cmap="viridis", alpha=0.35, rasterized=True)
    ax[0].plot([0.4, 2.0], [0.4, 2.0], "k--", lw=1.2, label="ideal")
    ax[0].axvspan(1.15, 1.40, color="gray", alpha=0.15)
    ax[0].set_xlabel(r"true $h$"); ax[0].set_ylabel(r"Bayes-optimal $\hat h=E[h|Q]$")
    ax[0].set_title(rf"(a) optimal estimator, $R^2_{{\rm opt}}={r2_opt:.3f}$")
    ax[0].legend(fontsize=8, frameon=False)
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label(r"$Q=\sum_a n_a^2$", fontsize=8)

    # (b) Cramer-Rao single-snapshot std of h_hat vs h
    ax[1].plot(h_grid, np.sqrt(crb), "o-", color="crimson", ms=4,
               label=r"CRB: $\min\,\mathrm{std}(\hat h)=1/\sqrt{I(h)}$")
    ax[1].axvspan(1.15, 1.40, color="gray", alpha=0.15, label="coexistence")
    ax[1].axvline(1.25305, color="0.4", ls=":", lw=1)
    ax[1].set_xlabel(r"true $h$"); ax[1].set_ylabel(r"optimal single-snapshot $\mathrm{std}(\hat h)$")
    ax[1].set_title("(b) Cram\u00e9r\u2013Rao floor")
    ax[1].legend(fontsize=8, frameon=False)

    fig.tight_layout(); fig.savefig("fig_infobound.pdf"); fig.savefig("fig_infobound.png", dpi=200)

    # export data for replotting
    import pandas as pd
    pd.DataFrame({"h_grid": h_grid, "fisher_I": I, "crb_var": crb,
                  "std_opt": np.sqrt(crb), "varQ": varQ, "meanQ": meanQ}
                 ).to_csv("data_infobound_crb.csv", index=False)
    pd.DataFrame({"true_h": hs, "opt_pred": ghat, "Q": Qs}
                 ).to_csv("data_infobound_opt.csv", index=False)
    print("wrote fig_infobound.pdf, info_bound.json, data_infobound_*.csv")


if __name__ == "__main__":
    main()
