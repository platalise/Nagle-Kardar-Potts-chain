"""Prove the factor-2 pitfall in the collective-update energy change.

The buggy formula (originally shipped in `1D_pott.py`, and easy to reinvent) is

    dE_glob_buggy = -(h/N)*(2*n_new + 1) + (h/N)*(2*n_old - 1)

which equals TWICE the correct value for H = -(h/N) sum_{i<j} delta_{s_i,s_j}:
the simulator secretly runs at h_eff = 2h and misplaces the transition.

The correct expression (pre-flip counts) is

    dE_glob_correct = -(h/N) * (n_new - n_old + 1).

Import is silent; run the demonstration with:  python -m nkpotts.verify_bug
"""
import numpy as np


def dE_glob_correct(counts, old_s, new_s, L, h):
    return -(h/L) * (int(counts[new_s]) - int(counts[old_s]) + 1)


def dE_glob_buggy(counts, old_s, new_s, L, h):
    return -(h/L)*(2*int(counts[new_s]) + 1) + (h/L)*(2*int(counts[old_s]) - 1)


def total_E(spins, L, q, J, h):
    e_loc = -J * sum(int(spins[i] == spins[(i+1) % L]) for i in range(L))
    e_glob = 0.0
    for i in range(L):
        for j in range(i+1, L):
            if spins[i] == spins[j]:
                e_glob -= h / L
    return e_loc + e_glob


def demonstrate(trials=5, L=12, q=3, J=1.0, h=0.7, seed=0):
    rng = np.random.default_rng(seed)
    for t in range(trials):
        spins = rng.integers(0, q, L)
        counts = np.bincount(spins, minlength=q)
        idx = int(rng.integers(0, L))
        old_s = int(spins[idx]); new_s = (old_s + 1) % q
        E0 = total_E(spins, L, q, J, h)
        spins2 = spins.copy(); spins2[idx] = new_s
        dE_true = total_E(spins2, L, q, J, h) - E0
        left = int(spins[(idx-1) % L]); right = int(spins[(idx+1) % L])
        dE_loc = J*(int(old_s == left) + int(old_s == right)
                    - int(new_s == left) - int(new_s == right))
        dE_ok = dE_loc + dE_glob_correct(counts, old_s, new_s, L, h)
        dE_buggy = dE_loc + dE_glob_buggy(counts, old_s, new_s, L, h)
        glob_true = dE_true - dE_loc
        glob_buggy = dE_buggy - dE_loc
        ratio = glob_buggy / glob_true if abs(glob_true) > 1e-12 else float("nan")
        print(f"trial {t}: dE_true={dE_true:+.6f}  dE_correct={dE_ok:+.6f}  "
              f"dE_buggy={dE_buggy:+.6f}  ratio_glob={ratio:.3f}")


if __name__ == "__main__":
    demonstrate()
