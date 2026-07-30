"""Regenerate Fig. 2 as two panels:
(a) q=3 (J,h) phase diagram with spinodals [equivalent to K>0 sector of
    Campa et al. Fig. 1 under h=1/T, J=Kh; spinodal lines are new]
(b) general-q closed-form transition lines h_t(J;q) and spinodals — the
    general-q answer to the open question posed in Campa et al.
Outputs fig_phase_diagram.pdf + data_phase_diagram_generalq.csv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7, "savefig.bbox": "tight"})
BETA = 1.0


def ht_closed(J, q):
    return (2.0*(q-1.0)/(q-2.0)) * np.log(1.0 + (q-2.0)*np.exp(-BETA*J))


def h_spinodal(J, q):
    return q*q / (2.0*np.exp(BETA*J) + q - 2.0)


d = np.load("data/phase_diagram_data.npz")

fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))

# ---- (a) q=3 with spinodals (numeric lines from exact minimization) ----
ax[0].plot(d["J"], d["sp_up"], "--", color="crimson", lw=1.3,
           label=r"$h_{sp}^{+}$")
ax[0].plot(d["J"], d["ht"], "-", color="k", lw=1.8, label=r"$h_t(J)$")
ax[0].plot(d["J"], d["sp_low"], ":", color="royalblue", lw=1.3,
           label=r"$h_{sp}^{-}$")
ax[0].fill_between(d["J"], d["sp_low"], d["sp_up"], color="0.9")
ax[0].set_xlabel(r"$J/k_BT$")
ax[0].set_ylabel(r"$h/k_BT$")
ax[0].set_title(r"(a) $q=3$: transition and spinodals")
ax[0].legend(fontsize=8, frameon=False)
jj = d["jump"][np.isfinite(d["jump"])]
dev = max(float(np.max(np.abs(jj - 0.5))) if jj.size else 1e-6, 1e-6)
ax[0].text(0.97, 0.03,
           r"$\Delta\phi_t=\frac{1}{2}$ along the line" + "\n"
           + rf"(deviation $< {dev:.0e}$)",
           transform=ax[0].transAxes, ha="right", va="bottom", fontsize=7.5,
           bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

# ---- (b) general-q closed forms (the new content) ----
Js = np.linspace(0.0, 2.0, 200)
rows = {"J": Js}
colors = plt.cm.plasma(np.linspace(0.05, 0.75, 4))
for qv, c in zip([3, 4, 5, 8], colors):
    ht = ht_closed(Js, qv)
    sp = h_spinodal(Js, qv)
    jump = (qv - 2.0)/(qv - 1.0)
    ax[1].plot(Js, ht, "-", lw=1.7, color=c,
               label=rf"$q={qv}$  ($\Delta\phi_t={jump:.3g}$)")
    ax[1].plot(Js, sp, "--", lw=0.9, color=c, alpha=0.6)
    rows[f"ht_q{qv}"] = ht
    rows[f"hsp_q{qv}"] = sp
ax[1].set_xlabel(r"$J/k_BT$")
ax[1].set_ylabel(r"$h/k_BT$")
ax[1].set_title(r"(b) general $q$: $\beta h_t=\frac{2(q-1)}{q-2}\ln[1+(q-2)e^{-\beta J}]$")
ax[1].legend(fontsize=7.5, frameon=False,
             title="solid: $h_t$;  dashed: $h_{sp}^{+}$", title_fontsize=7.5)

fig.tight_layout()
fig.savefig("fig_phase_diagram.pdf")
fig.savefig("fig_phase_diagram.png", dpi=220)
pd.DataFrame(rows).to_csv("data/data_phase_diagram_generalq.csv", index=False)
print("wrote fig_phase_diagram.pdf + data_phase_diagram_generalq.csv")
