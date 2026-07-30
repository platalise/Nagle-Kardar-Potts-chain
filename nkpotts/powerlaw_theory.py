"""
powerlaw_theory.py
==================
Analytic scaffolding for the power-law herding model

    H = -J sum_i delta(s_i,s_{i+1})
        - (h / 2 N_alpha) sum_{i!=j} delta(s_i,s_j) / r_ij^alpha ,

with r_ij the minimum-image ring distance and the Kac normalization
    N_alpha = sum_{j!=i} r_ij^{-alpha}   (same for every i by translation invariance).

The Kac factor keeps the herding energy per site bounded by h/2 for ALL alpha,
so a herding-driven transition exists at every alpha; alpha tunes its ORDER and
universality rather than its existence.

Regime structure in d=1 (sigma == alpha-1):
  * alpha <= 1  (sigma <= 0): STRONG long-range / nonadditive. Kac-normalized
    mean field is EXACT. Potts q>=3 => FIRST ORDER. The intensive free energy
    equals the Curie-Weiss one, so the transition line h_t is INDEPENDENT of
    alpha here (= the alpha=0 value found in theory_exact.py).
  * 1 < alpha < 3/2 (0 < sigma < 1/2): weak long-range, still mean-field
    exponents (Sak); first-order expected to persist near alpha=1 and weaken.
  * 3/2 < alpha < 2 (1/2 < sigma < 1): genuine long-range criticality with
    nontrivial exponents; the Potts first-order character can terminate at a
    q-dependent tricritical alpha*(q).
  * alpha >= 2 (sigma >= 1): effectively short range; 1D short-range Potts has
    no finite-T LRO, so the ordered phase requires h -> large and the transition
    is pushed up / washes into a crossover.

This module provides:
  * kac_norm(L, alpha)         -> N_alpha(L)
  * kac_norm_asymptotics       -> analytic N_alpha(L) growth
  * mean_field_ht(alpha, J)    -> h_t for alpha<=1 (equals exact CW+transfer value)
  * regime(alpha)              -> label
It reuses the exact q=3 transfer-matrix solver from theory_exact.py.
"""
import numpy as np
import json
import os


def kac_norm(L, alpha):
    r = np.arange(1, L)
    d = np.minimum(r, L - r).astype(float)
    return np.sum(d**(-alpha))


def kac_norm_asymptotics(L, alpha):
    """Leading growth of N_alpha(L)."""
    if alpha < 1:
        # sum ~ 2 * integral_1^{L/2} r^{-alpha} dr ~ 2/(1-alpha) (L/2)^{1-alpha}
        return 2.0/(1.0-alpha)*(L/2.0)**(1.0-alpha)
    elif abs(alpha-1) < 1e-9:
        return 2.0*np.log(L/2.0)
    else:
        return 2.0*_zeta(alpha)   # converges


def _zeta(a, nmax=200000):
    n = np.arange(1, nmax)
    return np.sum(n.astype(float)**(-a))


from nkpotts.theory_exact import ht_closed


def regime(alpha):
    if alpha <= 1.0:
        return "strong-LR (mean-field exact, first order)"
    elif alpha < 1.5:
        return "weak-LR, mean-field exponents"
    elif alpha < 2.0:
        return "long-range critical (nontrivial); possible tricritical alpha*"
    else:
        return "effectively short-range (transition pushed up / crossover)"


if __name__ == "__main__":
    J, T, q = 1.0, 1.0, 3
    out = {"J": J, "T": T, "q": q}

    # mean-field h_t for alpha<=1 is alpha-independent and equals CW+transfer value
    out["h_t_meanfield_alpha_le_1"] = float(ht_closed(J))
    out["h_t_at_J0_check_4ln2"] = float(4*np.log(2))

    # Kac norm vs asymptotics
    tab = {}
    for alpha in [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.5]:
        row = {}
        for L in [256, 1024, 4096]:
            row[f"N_alpha(L={L})"] = float(kac_norm(L, alpha))
        row["asymptotic(L=4096)"] = float(kac_norm_asymptotics(4096, alpha))
        row["regime"] = regime(alpha)
        row["sigma=alpha-1"] = alpha - 1.0
        tab[f"alpha={alpha}"] = row
    out["kac_and_regimes"] = tab

    print(json.dumps(out, indent=2))
    with open("powerlaw_theory.json", "w") as f:
        json.dump(out, f, indent=2)
