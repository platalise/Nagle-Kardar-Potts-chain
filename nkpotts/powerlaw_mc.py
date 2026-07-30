"""
powerlaw_mc.py
==============
Monte Carlo for the power-law herding chain

    H = -J sum_i delta(s_i,s_{i+1})
        - (h / 2 N_alpha) sum_{i!=j} delta(s_i,s_j)/r_ij^alpha ,

r_ij = min-image ring distance, N_alpha = sum_{j!=i} r_ij^{-alpha} (Kac).

The single-flip global energy change (site i: a->b, PRE-flip spins) is exact:

    dE_glob = -(h / N_alpha) * ( Phi_b(i) - Phi_a(i) ),
    Phi_c(i) = sum_{j!=i} delta(s_j,c) * kappa[ dist(i,j) ].

We evaluate Phi_a,Phi_b on demand in O(N) (precomputed kernel kappa[delta]),
giving O(N^2) per sweep. This is fine up to L ~ 1024 for order detection.

Order-of-transition diagnostics per (alpha, L):
  * chi(h), M(h): locate peak / transition h_t(alpha)
  * Binder U4 minimum (deep negative dip that DEEPENS with L => first order)
  * order-parameter histogram at the transition (bimodal => first order)
  * susceptibility-peak scaling chi_max ~ L^p (p->1 first order, p<1 continuous)
"""
import numpy as np
from numba import njit
import argparse, json

Q = 3
J = 1.0
T = 1.0


def kernel(L, alpha):
    kap = np.zeros(L)
    for d in range(1, L):
        dd = min(d, L - d)
        kap[d] = dd**(-alpha)
    return kap


def kac_norm_from_kernel(kap):
    return float(np.sum(kap))


@njit(cache=True)
def _dEglob(spins, i, a, b, kap, L, Nalpha, h):
    # Phi_b(i) - Phi_a(i), pre-flip
    acc = 0.0
    si = spins
    for j in range(L):
        if j == i:
            continue
        d = i - j
        if d < 0:
            d = -d
        kk = kap[d]
        sj = si[j]
        if sj == b:
            acc += kk
        elif sj == a:
            acc -= kk
    return -(h / Nalpha) * acc


@njit(cache=True)
def _sweep(spins, counts, kap, L, Nalpha, h, q, J, beta, nsweep):
    for _ in range(nsweep):
        for _ in range(L):
            i = np.random.randint(0, L)
            a = spins[i]
            b = np.random.randint(0, q)
            if a == b:
                continue
            left = spins[L-1] if i == 0 else spins[i-1]
            right = spins[(i+1) % L]
            dE_loc = J*((a == left)+(a == right)-(b == left)-(b == right))
            dE_glob = _dEglob(spins, i, a, b, kap, L, Nalpha, h)
            dE = dE_loc + dE_glob
            if dE <= 0.0 or np.random.rand() < np.exp(-beta*dE):
                spins[i] = b
                counts[a] -= 1
                counts[b] += 1


@njit(cache=True)
def _phi(counts, L, q):
    nmax = 0
    for a in range(q):
        if counts[a] > nmax:
            nmax = counts[a]
    return (q*(nmax/L) - 1.0)/(q-1.0)


@njit(cache=True)
def run_one(L, h, alpha_kernel, Nalpha, q, J, beta, burn, nmeas, freq, seed):
    np.random.seed(seed)
    spins = np.random.randint(0, q, L)
    counts = np.zeros(q, dtype=np.int64)
    for s in spins:
        counts[s] += 1
    _sweep(spins, counts, alpha_kernel, L, Nalpha, h, q, J, beta, burn)
    n = nmeas // freq
    m = np.zeros(n)
    for k in range(n):
        _sweep(spins, counts, alpha_kernel, L, Nalpha, h, q, J, beta, freq)
        m[k] = _phi(counts, L, q)
    return m


def moments(m, L, beta):
    N = len(m)
    m1 = m.mean(); m2 = (m**2).mean(); m4 = (m**4).mean()
    chi = L*beta*(m2 - m1**2)
    u4 = 1.0 - m4/(3.0*m2**2) if m2 > 0 else 0.0
    # jackknife error on chi
    sm, sm2 = m.sum(), (m**2).sum()
    chi_jk = np.array([L*beta*(((sm2-m[i]**2)/(N-1)) - ((sm-m[i])/(N-1))**2) for i in range(N)])
    chi_err = np.sqrt((N-1)*np.mean((chi_jk - chi_jk.mean())**2))
    return m1, chi, chi_err, u4


def scan_alpha(alpha, Ls, h_axis, burn, nmeas, freq, seed0=1):
    beta = 1.0/T
    rows = []
    for L in Ls:
        kap = kernel(L, alpha)
        Nalpha = kac_norm_from_kernel(kap)
        for jh, h in enumerate(h_axis):
            m = run_one(L, h, kap, Nalpha, Q, J, beta, burn, nmeas, freq,
                        seed0 + jh + 1000*L + 100000*int(alpha*100))
            m1, chi, chi_err, u4 = moments(m, L, beta)
            rows.append(dict(alpha=alpha, L=L, h=h, M=m1, chi=chi,
                             chi_err=chi_err, U4=u4))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        beta = 1.0/T
        # (1) alpha -> 0 must recover the CW+transfer transition h_t = 1.25305
        print("=== smoke: alpha=0.05 should peak near h_t=1.25305 ===")
        L = 128
        kap = kernel(L, 0.05); Nalpha = kac_norm_from_kernel(kap)
        haxis = np.linspace(1.05, 1.5, 10)
        chis = []
        for h in haxis:
            m = run_one(L, h, kap, Nalpha, Q, J, beta, 3000, 3000, 5, 42)
            _, chi, _, _ = moments(m, L, beta)
            chis.append(chi)
        hpk = haxis[int(np.argmax(chis))]
        print(f"  L={L} alpha=0.05: chi-peak at h={hpk:.3f} (expect ~1.25)")
        for h, c in zip(haxis, chis):
            print(f"    h={h:.3f}  chi={c:.2f}")

        # (2) compare Binder dip at small vs larger alpha at a couple L
        print("=== smoke: transition survives at larger alpha, order via U4 ===")
        for alpha in [0.5, 1.5]:
            for L in [128, 256]:
                kap = kernel(L, alpha); Nalpha = kac_norm_from_kernel(kap)
                # coarse locate peak
                haxis = np.linspace(1.0, 1.9, 10)
                best = (None, -1)
                u4min = 1.0
                for h in haxis:
                    m = run_one(L, h, kap, Nalpha, Q, J, beta, 2500, 2500, 5, 7)
                    _, chi, _, u4 = moments(m, L, beta)
                    if chi > best[1]:
                        best = (h, chi)
                    u4min = min(u4min, u4)
                print(f"  alpha={alpha} L={L}: chi-peak h~{best[0]:.2f} "
                      f"chi_max={best[1]:.1f}  min U4 over scan={u4min:.3f}")
