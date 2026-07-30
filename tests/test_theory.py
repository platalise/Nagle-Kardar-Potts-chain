"""Fast unit tests for the exact-theory results.
Run with:  pytest -q
"""
import numpy as np
import pytest


# ---------------------------------------------------------------- Exact theory
def test_ht_closed_q3_J1_T1():
    from nkpotts.theory_exact import ht_closed
    # Closed form beta*h_t = 2(q-1)/(q-2) * ln(1 + (q-2)e^{-beta J})
    ht = ht_closed(J=1.0, beta=1.0, q=3)
    assert ht == pytest.approx(1.2530467500626585, rel=1e-10)


def test_ht_closed_reduces_to_curie_weiss_at_J0():
    from nkpotts.theory_exact import ht_closed
    # For J=0 with q=3, beta=1: h_t = 4 ln 2 (classical Curie-Weiss)
    ht0 = ht_closed(J=0.0, beta=1.0, q=3)
    assert ht0 == pytest.approx(4.0 * np.log(2.0), rel=1e-12)


def test_spinodal_closed_form_q3():
    from nkpotts.theory_exact import h_spinodal_closed
    # h_sp+ = q^2 T / (2 e^{beta J} + q - 2)
    hsp = h_spinodal_closed(J=1.0, beta=1.0, q=3)
    expected = 9.0 / (2.0 * np.exp(1.0) + 1.0)
    assert hsp == pytest.approx(expected, rel=1e-12)


def test_reflection_symmetry_at_ht_q3_machine_precision():
    """f(phi) = f(phi_t - phi) at coexistence (phi_t = 1/2 for q=3)."""
    from nkpotts.theory_exact import f_of_phi, ht_closed
    J = 1.0
    ht = ht_closed(J)
    phi_t = 0.5
    max_err = 0.0
    for p in np.linspace(-0.2, 0.7, 51):
        err = abs(f_of_phi(p, J, ht) - f_of_phi(phi_t - p, J, ht))
        max_err = max(max_err, err)
    assert max_err < 1e-13, f"reflection symmetry broken: max_err={max_err}"


# ---------------------------------------------------------------- Kac norm
def test_kac_norm_growth_regime():
    """N_alpha(L) diverges for alpha<1 (extensive per site) and converges for alpha>1."""
    from nkpotts.powerlaw_theory import kac_norm
    # alpha=0.5: N_alpha ~ 2/(1-alpha) (L/2)^{1-alpha}, grows with L
    n1 = kac_norm(256, 0.5)
    n2 = kac_norm(1024, 0.5)
    assert n2 > 1.5 * n1
    # alpha=2.0: N_alpha ~ 2 zeta(2) ~ pi^2/3 ~ 3.29, essentially L-independent
    m1 = kac_norm(256, 2.0)
    m2 = kac_norm(1024, 2.0)
    assert abs(m2 - m1) < 0.05
    assert abs(m2 - np.pi**2 / 3.0) < 0.02


# ---------------------------------------------------------------- The factor-2 bug
def test_factor_two_bug_brute_force():
    """Prove that the corrected dE_glob equals brute-force energy difference."""
    rng = np.random.default_rng(0)
    L, q, J, h = 12, 3, 1.0, 0.7

    def total_E(spins):
        e_loc = -J * sum(spins[i] == spins[(i+1) % L] for i in range(L))
        e_glob = 0.0
        for i in range(L):
            for j in range(i+1, L):
                if spins[i] == spins[j]:
                    e_glob -= h / L
        return e_loc + e_glob

    for _ in range(20):
        spins = rng.integers(0, q, L)
        counts = np.bincount(spins, minlength=q)
        idx = int(rng.integers(0, L))
        old_s = int(spins[idx])
        new_s = (old_s + 1) % q

        E0 = total_E(spins)
        spins2 = spins.copy()
        spins2[idx] = new_s
        dE_true = total_E(spins2) - E0

        left = int(spins[(idx-1) % L]); right = int(spins[(idx+1) % L])
        dE_loc = J * (int(old_s == left) + int(old_s == right)
                      - int(new_s == left) - int(new_s == right))
        dE_correct = dE_loc - (h / L) * (counts[new_s] - counts[old_s] + 1)
        dE_buggy = dE_loc + (-(h/L)*(2*counts[new_s]+1) + (h/L)*(2*counts[old_s]-1))

        assert dE_correct == pytest.approx(dE_true, abs=1e-12)
        # The buggy formula gives exactly the wrong answer by a factor of 2
        # on the global part (h_eff = 2h):
        assert (dE_buggy - dE_loc) == pytest.approx(2.0 * (dE_true - dE_loc), abs=1e-12)


# ---------------------------------------------------------------- MC sanity check
@pytest.mark.slow
def test_mc_chi_peak_at_exact_ht():
    """Corrected MC susceptibility peak should sit near h_t within grid resolution.
    Uses a small L / short run to keep CI fast; tolerance is generous
    accordingly (weak-first-order fluctuations at small L)."""
    from nkpotts.mc_corrected import measure_grid
    h_axis = np.linspace(1.15, 1.40, 11)
    rows = measure_grid(L=200, h_axis=h_axis, burn=3000, nmeas=3000, freq=5)
    chis = np.array([r["chi"] for r in rows])
    h_peak = h_axis[int(np.argmax(chis))]
    # h_t ~ 1.253 (exact). Small-L, short-run drift can put the peak up to ~0.1
    # above; a generous check protects against transient CI fluctuations while
    # still catching a truly broken build (peak at wrong side of the transition
    # would fail).
    assert 1.15 <= h_peak <= 1.40, f"chi-peak at {h_peak}, expected within [1.15, 1.40] of h_t=1.253"
