"""
theory_exact.py
================
EXACT thermodynamic-limit solution of the 1D q-state Potts chain with a
Curie-Weiss herding term:

    H = -J sum_i delta(s_i, s_{i+1}) - (h/N) sum_{i<j} delta(s_i, s_j)

Method: Hubbard-Stratonovich on the collective term + 1D transfer matrix.
Because the collective coupling is infinite-range, the saddle point is EXACT
(Laplace / large deviations). Variational free energy per site:

    f(x) = (h/2) |x|^2 - k_B T * ln lambda_max[ T(x) ],
    T(x)_{ss'} = exp( beta*J*delta_{ss'} + beta*h*(x_s + x_{s'})/2 ),

minimized over the simplex x (the saddle enforces sum_a x_a = 1, x_a = m_a).

Outputs:
  * closed-form paramagnetic spinodal (ordered spinodal: numerical)  h_sp = q^2 T / (2 e^{beta J} + q - 2)
                                            = q T tanh(1/(2 xi))
  * exact first-order transition h_t (free-energy degeneracy)
  * order-parameter jump, herding-overlap jump Delta u, barrier height
  * phase diagram h_t(J), spinodals(J) at fixed T
  * free-energy landscape figures
  * theory_results.json with all numbers for the manuscript

Author: (for Quang's PRE project), 2026.
"""
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = 3
T = 1.0          # k_B T
BETA = 1.0 / T


# ----------------------------------------------------------------------
# Variational free energy
# ----------------------------------------------------------------------
def x_of_phi(phi, q=Q):
    """Symmetric one-color parametrization on the simplex:
       x = (1/q + (q-1)phi/q, 1/q - phi/q, ..., 1/q - phi/q)."""
    x = np.full(q, 1.0/q - phi/q)
    x[0] = 1.0/q + (q - 1.0)*phi/q
    return x


def lam_max(x, J, h, beta=BETA, q=Q):
    u = beta * h * x
    K = beta * J
    Tm = np.exp(K * np.eye(q) + 0.5*(u[:, None] + u[None, :]))
    return np.linalg.eigvalsh(Tm)[-1]


def f_of_x(x, J, h, beta=BETA):
    return 0.5*h*np.dot(x, x) - (1.0/beta)*np.log(lam_max(x, J, h, beta))


def f_of_phi(phi, J, h):
    return f_of_x(x_of_phi(phi), J, h)


# ----------------------------------------------------------------------
# Closed-form paramagnetic spinodal
# ----------------------------------------------------------------------
def h_spinodal_closed(J, beta=BETA, q=Q):
    return q*q / (beta * (2.0*np.exp(beta*J) + q - 2.0))


def ht_closed(J, beta=BETA, q=Q):
    """Closed form for the first-order transition line, derived from the hidden
    reflection symmetry of f at coexistence (q=3 case due to Campa et al. 2026):
        beta * h_t = [2(q-1)/(q-2)] * ln(1 + (q-2) e^{-beta J}).
    """
    return (2.0*(q-1.0)/((q-2.0)*beta)) * np.log(1.0 + (q-2.0)*np.exp(-beta*J))


def xi_1d(J, beta=BETA, q=Q):
    lam1 = np.exp(beta*J) + q - 1.0
    lam2 = np.exp(beta*J) - 1.0
    if lam2 <= 0:
        return 0.0
    return 1.0/np.log(lam1/lam2)


# ----------------------------------------------------------------------
# Minima of f(phi)
# ----------------------------------------------------------------------
def golden_min(func, a, b, tol=1e-12):
    invphi = (np.sqrt(5.0) - 1.0)/2.0
    c, d = b - invphi*(b-a), a + invphi*(b-a)
    fc, fd = func(c), func(d)
    while abs(b-a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi*(b-a); fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi*(b-a); fd = func(d)
    xm = 0.5*(a+b)
    return xm, func(xm)


def find_minima(J, h, ngrid=2001):
    """Return list of (phi*, f*) local minima of f(phi) on phi in (-0.45, 0.999]."""
    phis = np.linspace(-0.45, 0.999, ngrid)
    fs = np.array([f_of_phi(p, J, h) for p in phis])
    minima = []
    for i in range(1, ngrid-1):
        if fs[i] < fs[i-1] and fs[i] < fs[i+1]:
            pm, fm = golden_min(lambda p: f_of_phi(p, J, h), phis[i-1], phis[i+1])
            minima.append((pm, fm))
    # endpoint minima
    if fs[0] < fs[1]:
        minima.append((phis[0], fs[0]))
    if fs[-1] < fs[-2]:
        minima.append((phis[-1], fs[-1]))
    # deduplicate
    out = []
    for pm, fm in sorted(minima):
        if not out or abs(pm - out[-1][0]) > 1e-6:
            out.append((pm, fm))
    return out


def split_minima(minima, tol=2e-3):
    """Split minima into (disordered ~ phi=0) and ordered (phi>tol)."""
    dis = [m for m in minima if abs(m[0]) < tol]
    orde = [m for m in minima if m[0] >= tol]
    orde.sort(key=lambda m: m[1])
    return (dis[0] if dis else None), (orde[0] if orde else None)


def delta_f(J, h):
    """f_ordered - f_disordered (None if a branch is missing)."""
    dis, orde = split_minima(find_minima(J, h))
    if dis is None or orde is None:
        return None
    return orde[1] - dis[1], dis, orde


def find_ht(J, hlo, hhi, tol=1e-10):
    """Bisection on f_ord - f_dis = 0."""
    lo, hi = hlo, hhi
    dlo = delta_f(J, lo)
    dhi = delta_f(J, hi)
    assert dlo is not None and dhi is not None, "coexistence window not bracketed"
    assert dlo[0] > 0 and dhi[0] < 0, f"bad bracket: {dlo[0]}, {dhi[0]}"
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        d = delta_f(J, mid)
        if d is None or d[0] > 0:
            lo = mid
        else:
            hi = mid
    ht = 0.5*(lo+hi)
    d, dis, orde = delta_f(J, ht)
    return ht, dis, orde


def find_lower_spinodal(J, hstart, tol=1e-9):
    """Smallest h at which an ordered local minimum exists."""
    lo, hi = 0.02, hstart
    def has_ord(h):
        _, orde = split_minima(find_minima(J, h))
        return orde is not None
    assert has_ord(hi) and not has_ord(lo)
    while hi - lo > tol:
        mid = 0.5*(lo+hi)
        if has_ord(mid):
            hi = mid
        else:
            lo = mid
    return 0.5*(lo+hi)


def observables_at(J, h, phi):
    """Energy per site and herding overlap u = |m|^2 at a minimum."""
    x = x_of_phi(phi)
    u = float(np.dot(x, x))
    # e = d(beta f)/d(beta) at fixed minimizer (envelope theorem), numerically
    db = 1e-6
    def bf(beta):
        return beta*(0.5*h*np.dot(x, x)) - np.log(lam_max(x, J, h, beta=beta))
    e = (bf(BETA+db) - bf(BETA-db)) / (2*db)
    return u, float(e)


def barrier(J, h, phi_dis, phi_ord):
    phis = np.linspace(phi_dis, phi_ord, 4001)
    fs = np.array([f_of_phi(p, J, h) for p in phis])
    i = int(np.argmax(fs))
    f_min = min(fs[0], fs[-1])
    return float(fs[i] - f_min), float(phis[i])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    J = 1.0
    res = {"q": Q, "T": T, "J": J}

    # --- spinodal (closed form) + numerical Hessian check
    h_sp = h_spinodal_closed(J)
    xi = xi_1d(J)
    res["h_sp_closed_form"] = h_sp
    res["xi_1D"] = xi
    res["h_sp_tanh_form"] = Q*T*np.tanh(1.0/(2.0*xi))
    # numerical check: curvature of f at phi=0 changes sign at h_sp
    dp = 1e-4
    for htest, name in [(h_sp*0.999, "just_below"), (h_sp*1.001, "just_above")]:
        curv = (f_of_phi(dp, J, htest) - 2*f_of_phi(0.0, J, htest) + f_of_phi(-dp, J, htest))/dp**2
        res[f"curvature_{name}_spinodal"] = float(curv)

    # --- transition point (bracket first)
    print("Scanning for coexistence window ...")
    hgrid = np.linspace(0.6, h_sp - 1e-4, 60)
    window = []
    for h in hgrid:
        d = delta_f(J, h)
        if d is not None:
            window.append((h, d[0]))
    hs = [w[0] for w in window]
    ds = [w[1] for w in window]
    # find sign change
    isign = None
    for i in range(len(ds)-1):
        if ds[i] > 0 and ds[i+1] < 0:
            isign = i
            break
    assert isign is not None, f"no sign change found: window={window[:5]}...{window[-5:]}"
    ht, dis_min, ord_min = find_ht(J, hs[isign], hs[isign+1])
    res["h_t"] = float(ht)
    res["phi_ordered_at_ht"] = float(ord_min[0])
    res["phi_disordered_at_ht"] = float(dis_min[0])

    u_dis, e_dis = observables_at(J, ht, dis_min[0])
    u_ord, e_ord = observables_at(J, ht, ord_min[0])
    res["u_disordered_at_ht"] = u_dis
    res["u_ordered_at_ht"] = u_ord
    res["delta_u_at_ht"] = u_ord - u_dis
    res["e_disordered_at_ht"] = e_dis
    res["e_ordered_at_ht"] = e_ord
    res["delta_e_at_ht"] = e_ord - e_dis
    res["chi_max_over_L_prediction"] = BETA * (ord_min[0] - dis_min[0])**2 / 4.0

    bar, phi_bar = barrier(J, ht, dis_min[0], ord_min[0])
    res["beta_barrier_per_site_at_ht"] = BETA*bar
    res["phi_at_barrier_top"] = phi_bar
    res["ln_tunneling_estimate_L400"] = BETA*bar*400
    res["ln_tunneling_estimate_L3200"] = BETA*bar*3200

    # --- lower (ordered-phase) spinodal
    h_sp_low = find_lower_spinodal(J, ht)
    res["h_sp_lower"] = float(h_sp_low)

    # --- CLOSED FORM for the transition line (see module-level ht_closed):
    #     beta*h_t = [2(q-1)/(q-2)] * ln(1 + (q-2) e^{-beta J})
    res["h_t_closed_form"] = float(ht_closed(J))
    res["h_t_closed_minus_numeric"] = float(ht_closed(J) - ht)

    # exact reflection symmetry f(phi) = f(phi_t - phi) at h = h_t (q=3: phi_t = 1/2)
    phi_t = (Q - 2.0)/(Q - 1.0)
    sym_err = max(abs(f_of_phi(p, J, ht_closed(J)) - f_of_phi(phi_t - p, J, ht_closed(J)))
                  for p in np.linspace(-0.2, 0.7, 91))
    res["reflection_symmetry_max_error_at_ht"] = float(sym_err)

    # closed form vs numeric bisection at several J
    res["ht_closed_vs_numeric"] = {}
    for Jv in [0.3, 0.7, 1.3]:
        spu = h_spinodal_closed(Jv)
        hc = ht_closed(Jv)
        htn = None
        for eps in [2e-3, 1e-3, 5e-4, 2e-4]:
            try:
                htn, _, _ = find_ht(Jv, hc - eps, min(hc + 10*eps, spu - 1e-4), tol=1e-9)
                break
            except AssertionError:
                continue
        res["ht_closed_vs_numeric"][f"J={Jv}"] = [float(hc), float(htn) if htn else None]

    # q=4 check of the general-q closed form
    def general_check(qv, Jv, beta=BETA):
        def x_phi(phi):
            x = np.full(qv, 1.0/qv - phi/qv); x[0] = 1.0/qv + (qv-1.0)*phi/qv
            return x
        def lm(x, h):
            u = beta*h*x
            Tm = np.exp(beta*Jv*np.eye(qv) + 0.5*(u[:, None]+u[None, :]))
            return np.linalg.eigvalsh(Tm)[-1]
        def fq(phi, h):
            x = x_phi(phi)
            return 0.5*h*np.dot(x, x) - np.log(lm(x, h))/beta
        hc = (2.0*(qv-1.0)/((qv-2.0)*beta))*np.log(1.0+(qv-2.0)*np.exp(-beta*Jv))
        phit = (qv-2.0)/(qv-1.0)
        err = max(abs(fq(p, hc) - fq(phit - p, hc)) for p in np.linspace(-0.1, phit+0.1, 41))
        return hc, float(fq(0.0, hc) - fq(phit, hc)), err
    hc4, deg4, err4 = general_check(4, 1.0)
    res["q4_h_t_closed"] = float(hc4)
    res["q4_degeneracy_f0_minus_fphit"] = deg4
    res["q4_reflection_symmetry_max_error"] = err4

    # off-axis check: coarse 2D scan of f over the simplex at h_t confirms
    # the minima lie on the one-color axis (psi = 0)
    def f2d(phi, psi, h):
        x = np.array([1.0/3 + 2*phi/3, 1.0/3 - phi/3 + psi, 1.0/3 - phi/3 - psi])
        return f_of_x(x, J, h)
    best = min(((p, s, f2d(p, s, ht)) for p in np.linspace(-0.3, 0.95, 60)
                for s in np.linspace(-0.4, 0.4, 41)), key=lambda z: z[2])
    res["offaxis_scan_argmin_phi_psi"] = [float(best[0]), float(best[1])]

    # --- pure CW check (J=0): known h_t = 4 ln 2, h_sp = qT
    ht0, d0, o0 = find_ht(0.0, 2.755, 2.99)
    res["check_J0_h_t"] = float(ht0)
    res["check_J0_h_t_exact_4ln2"] = float(4.0*np.log(2.0))
    res["check_J0_h_sp"] = h_spinodal_closed(0.0)
    res["check_J0_phi_jump"] = float(o0[0])   # should be 1/2

    print(json.dumps(res, indent=2))
    with open("theory_results.json", "w") as fjson:
        json.dump(res, fjson, indent=2)

    # ------------------------------------------------------------------
    # Figure 1: free-energy landscapes
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    phis = np.linspace(-0.25, 0.95, 500)
    hs_plot = [0.9, res["h_sp_lower"], ht, 1.30, h_sp]
    labels = [r"$h=0.90$",
              rf"$h=h_{{sp}}^-={res['h_sp_lower']:.3f}$",
              rf"$h=h_t={ht:.4f}$",
              r"$h=1.30$",
              rf"$h=h_{{sp}}^+={h_sp:.4f}$"]
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(hs_plot)))
    land = {"phi": phis}
    for h, lab, c in zip(hs_plot, labels, colors):
        fs = np.array([f_of_phi(p, J, h) for p in phis])
        curve = fs - f_of_phi(0.0, J, h)
        ax.plot(phis, curve, label=lab, color=c, lw=1.6)
        land[f"f_h{h:.4f}"] = curve
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\beta[f(\phi)-f(0)]$")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_ylim(-0.05, 0.08)
    fig.tight_layout()
    fig.savefig("fig_theory_landscape.pdf")
    fig.savefig("fig_theory_landscape.png", dpi=250)
    import pandas as _pd
    _pd.DataFrame(land).to_csv("data_theory_landscape.csv", index=False)  # DATA EXPORT

    # ------------------------------------------------------------------
    # Figure 2: phase diagram h_t(J), spinodals(J) at T=1, + jump inset
    # ------------------------------------------------------------------
    print("Building phase diagram ...")
    Js = np.linspace(0.0, 2.0, 41)
    ht_line, sp_up, sp_low, jumps = [], [], [], []
    for Jv in Js:
        spu = h_spinodal_closed(Jv)
        sp_up.append(spu)
        hgrid = np.linspace(0.3*spu, spu - 1e-4, 40)
        prev = None
        bracket = None
        for h in hgrid:
            d = delta_f(Jv, h)
            if d is None:
                continue
            if prev is not None and prev[1] > 0 and d[0] < 0:
                bracket = (prev[0], h)
                break
            prev = (h, d[0])
        if bracket is None:
            ht_line.append(np.nan); sp_low.append(np.nan); jumps.append(np.nan)
            continue
        htv, dmin, omin = find_ht(Jv, bracket[0], bracket[1], tol=1e-7)
        ht_line.append(htv)
        jumps.append(omin[0])
        try:
            sp_low.append(find_lower_spinodal(Jv, htv, tol=1e-6))
        except AssertionError:
            sp_low.append(np.nan)

    fig2, ax2 = plt.subplots(figsize=(5.0, 3.8))
    ax2.plot(Js, sp_up, "--", color="crimson", lw=1.3, label=r"$h_{sp}^{+}=q^2T/(2e^{\beta J}+q-2)$")
    ax2.plot(Js, ht_line, "-", color="k", lw=1.8, label=r"$h_t(J)$ (first order)")
    ax2.plot(Js, sp_low, ":", color="royalblue", lw=1.3, label=r"$h_{sp}^{-}(J)$")
    ax2.fill_between(Js, sp_low, sp_up, color="0.9")
    ax2.set_xlabel(r"$J/k_BT$")
    ax2.set_ylabel(r"$h/k_BT$")
    ax2.legend(fontsize=8, frameon=False)
    # The order-parameter jump is exactly Delta phi_t = 1/2 along the whole q=3
    # line; plotting it produces only numerical noise around 0.5, so we report it
    # as text with the measured deviation instead.
    jumps_arr = np.array([j for j in jumps if np.isfinite(j)])
    dev = float(np.max(np.abs(jumps_arr - 0.5))) if jumps_arr.size else 0.0
    ax2.text(0.97, 0.03,
             r"$\Delta\phi_t=\frac{1}{2}$ along the $q{=}3$ line" + "\n"
             + rf"(deviation $< {max(dev,1e-6):.0e}$)",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=7.5,
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    fig2.tight_layout()
    fig2.savefig("fig_phase_diagram.pdf")
    fig2.savefig("fig_phase_diagram.png", dpi=250)

    np.savez("phase_diagram_data.npz", J=Js, ht=ht_line, sp_up=sp_up,
             sp_low=sp_low, jump=jumps)
    print("Done. Wrote theory_results.json, fig_theory_landscape.*, fig_phase_diagram.*")
