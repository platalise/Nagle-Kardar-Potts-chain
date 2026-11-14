"""
mc_corrected.py
===============
Corrected Monte Carlo for the 1D q-Potts chain + Curie-Weiss herding:

    H = -J sum_i delta(s_i,s_{i+1}) - (h/N) sum_{i<j} delta(s_i,s_j)

KEY FIX vs the original code
----------------------------
The global energy change for a single spin flip old->new is EXACTLY

    dE_glob = -(h/N) * ( n_new - n_old + 1 )       [BEFORE the flip counts]

The original code used  -(h/N)*(2 n_new + 1) + (h/N)*(2 n_old - 1)
= -(2h/N)*(n_new - n_old + 1), i.e. a spurious factor of 2 (h_eff = 2h).

This script provides:
  * measure_grid()  : M, chi, C_v, Binder U4, energy per site, with jackknife errors
  * hysteresis()    : up/down h-sweeps to expose metastability (first-order signature)
  * histograms()    : P(phi) at a chosen h to expose bimodality
  * a Wang-Landau-free "coexistence" sampler is not needed at this system size,
    but a simple field-mixing / longer-run option is provided via n_meas.

For publication runs use run_fullscale.sh. This file also has a __main__
smoke test that checks the chi-peak location against the exact h_t.
"""
import numpy as np
from numba import njit
import argparse, os, json

Q = 3
J = 1.0
T = 1.0


@njit(cache=True)
def _sweep(spins, counts, L, h, q, J, beta, nsweep):
    for _ in range(nsweep):
        for _ in range(L):
            idx = np.random.randint(0, L)
            old_s = spins[idx]
            new_s = np.random.randint(0, q)
            if old_s == new_s:
                continue
            left = spins[L-1] if idx == 0 else spins[idx-1]
            right = spins[(idx+1) % L]
            dE_loc = J*((old_s == left) + (old_s == right)
                        - (new_s == left) - (new_s == right))
            # CORRECT global term (counts are pre-flip):
            dE_glob = -(h/L)*(counts[new_s] - counts[old_s] + 1)
            dE = dE_loc + dE_glob
            if dE <= 0.0 or np.random.rand() < np.exp(-beta*dE):
                spins[idx] = new_s
                counts[old_s] -= 1
                counts[new_s] += 1


@njit(cache=True)
def _phi_from_counts(counts, L, q):
    # Potts magnetization M = (q * nmax/N - 1)/(q-1); here we return that.
    nmax = 0
    for a in range(q):
        if counts[a] > nmax:
            nmax = counts[a]
    return (q*(nmax/L) - 1.0)/(q - 1.0)


@njit(cache=True)
def _energy(spins, counts, L, h, q, J):
    e_loc = 0.0
    for i in range(L):
        if spins[i] == spins[(i+1) % L]:
            e_loc -= J
    ssq = 0.0
    for a in range(q):
        ssq += counts[a]*(counts[a]-1.0)
    e_glob = -(h/(2.0*L))*ssq
    return e_loc + e_glob


@njit(cache=True)
def run_one(L, h, q, J, beta, burn, nmeas, freq, seed):
    np.random.seed(seed)
    spins = np.random.randint(0, q, L)
    counts = np.zeros(q, dtype=np.int64)
    for s in spins:
        counts[s] += 1
    _sweep(spins, counts, L, h, q, J, beta, burn)
    n = nmeas // freq
    m = np.zeros(n); e = np.zeros(n)
    for k in range(n):
        _sweep(spins, counts, L, h, q, J, beta, freq)
        m[k] = _phi_from_counts(counts, L, q)
        e[k] = _energy(spins, counts, L, h, q, J)
    return m, e


@njit(cache=True)
def run_record(L, h, q, J, beta, burn, nmeas, freq, seed):
    """Long fixed-h run that records, every `freq` sweeps, the three integers
    needed for exact single-histogram reweighting in h:
        nmax = max_a n_a            (-> phi)
        S    = sum_a n_a (n_a - 1)  (-> herding energy, E_glob = -h S / 2L)
        nb   = #{i : s_i = s_{i+1}} (-> local energy,   E_loc  = -J nb)
    """
    np.random.seed(seed)
    spins = np.random.randint(0, q, L)
    counts = np.zeros(q, dtype=np.int64)
    for s in spins:
        counts[s] += 1
    _sweep(spins, counts, L, h, q, J, beta, burn)
    n = nmeas // freq
    nmax = np.zeros(n, dtype=np.int32)
    S = np.zeros(n, dtype=np.int64)
    nb = np.zeros(n, dtype=np.int32)
    for k in range(n):
        _sweep(spins, counts, L, h, q, J, beta, freq)
        mx = 0
        ss = 0
        for a in range(q):
            if counts[a] > mx:
                mx = counts[a]
            ss += counts[a] * (counts[a] - 1)
        b = 0
        for i in range(L):
            if spins[i] == spins[(i + 1) % L]:
                b += 1
        nmax[k] = mx
        S[k] = ss
        nb[k] = b
    return nmax, S, nb


def jackknife(m, e, L, beta):
    N = len(m)
    sm, sm2 = m.sum(), (m**2).sum()
    sm4 = (m**4).sum()
    se, se2 = e.sum(), (e**2).sum()
    chi = np.empty(N); cv = np.empty(N); mm = np.empty(N); u4 = np.empty(N)
    for i in range(N):
        mi = (sm - m[i])/(N-1)
        m2i = (sm2 - m[i]**2)/(N-1)
        m4i = (sm4 - m[i]**4)/(N-1)
        mm[i] = mi
        chi[i] = L*beta*(m2i - mi**2)
        u4[i] = 1.0 - m4i/(3.0*m2i**2) if m2i > 0 else 0.0
        ei = (se - e[i])/(N-1)
        e2i = (se2 - e[i]**2)/(N-1)
        cv[i] = beta*beta*(e2i - ei**2)/L
    def jkerr(a):
        return np.sqrt((N-1)*np.mean((a - a.mean())**2))
    return (mm.mean(), jkerr(mm), chi.mean(), jkerr(chi),
            cv.mean(), jkerr(cv), u4.mean(), jkerr(u4))


def measure_grid(L, h_axis, burn, nmeas, freq, seed0=12345):
    beta = 1.0/T
    rows = []
    for j, h in enumerate(h_axis):
        m, e = run_one(L, h, Q, J, beta, burn, nmeas, freq, seed0 + j + 1000*L)
        mm, me, chi, che, cv, cve, u4, u4e = jackknife(m, e, L, beta)
        rows.append(dict(L=L, h=h, M=mm, M_err=me, chi=chi, chi_err=che,
                         Cv=cv, Cv_err=cve, U4=u4, U4_err=u4e))
    return rows


@njit(cache=True)
def _seed_numba(seed):
    # Numba keeps its own RNG state, separate from NumPy's: np.random.seed()
    # called from plain Python does NOT seed the generator used inside the
    # jitted sweep.  Seeding must happen inside a jitted function.
    np.random.seed(seed)


def hysteresis(L, h_lo, h_hi, nh, sweeps_per_step, seed=7):
    """Up then down sweep of h, short equilibration each step -> hysteresis loop."""
    beta = 1.0/T
    np.random.seed(seed)      # initial configuration (NumPy RNG)
    _seed_numba(seed)         # Metropolis moves (Numba RNG) -> reproducible
    hs_up = np.linspace(h_lo, h_hi, nh)
    hs_dn = hs_up[::-1]
    spins = np.random.randint(0, Q, L)
    counts = np.zeros(Q, dtype=np.int64)
    for s in spins:
        counts[s] += 1
    m_up = []
    for h in hs_up:
        _sweep(spins, counts, L, h, Q, J, beta, sweeps_per_step)
        m_up.append(_phi_from_counts(counts, L, Q))
    m_dn = []
    for h in hs_dn:
        _sweep(spins, counts, L, h, Q, J, beta, sweeps_per_step)
        m_dn.append(_phi_from_counts(counts, L, Q))
    return hs_up, np.array(m_up), hs_dn, np.array(m_dn)


def histogram_phi(L, h, burn, nmeas, freq, seed=999):
    beta = 1.0/T
    m, _ = run_one(L, h, Q, J, beta, burn, nmeas, freq, seed)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        # quick check: chi peak near exact h_t = 1.25305 for a modest L
        L = 400
        h_axis = np.linspace(1.05, 1.45, 17)
        rows = measure_grid(L, h_axis, burn=4000, nmeas=4000, freq=5)
        chis = np.array([r["chi"] for r in rows])
        hpk = h_axis[np.argmax(chis)]
        print(f"[smoke] L={L}: chi-peak at h={hpk:.3f} (exact h_t=1.25305)")
        for r in rows:
            print(f"  h={r['h']:.3f}  M={r['M']:.3f}  chi={r['chi']:.3f}"
                  f"  U4={r['U4']:.3f}")
        # bimodality check at h_t
        m = histogram_phi(L, 1.25305, burn=4000, nmeas=8000, freq=2)
        hist, edges = np.histogram(m, bins=30)
        print("[smoke] P(phi) at h_t histogram (bin_center:count):")
        cen = 0.5*(edges[:-1]+edges[1:])
        for c, n in zip(cen, hist):
            if n > 0:
                print(f"   {c:+.3f}: {'#'*int(40*n/hist.max())} {n}")
