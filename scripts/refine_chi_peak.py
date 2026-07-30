#!/usr/bin/env python3
"""
Refine the susceptibility peak with an L-adaptive field grid.

Why: the production scan uses a uniform grid dh = 7.5e-3 for every L, but the
first-order peak width shrinks as ~1/L.  Measured from data_observables.csv, the
peak is spanned by 16 / 9 / 5 grid points at L = 400 / 800 / 1600, and by only a
single point at L = 3200 and 6400.  chi_max at the two largest sizes is therefore
an underestimate of uncontrolled size, which biases both the fitted exponent p
and the amplitude chi_max/L.

Fix: sample a narrow window around h_t with dh proportional to 1/L.  The window
half-width below is ~8/L, comfortably wider than the expected peak width ~6/L,
with 41 points across it.

Usage:
    python scripts/refine_chi_peak.py --L 3200 6400 --nmeas 40000
"""
import argparse
import numpy as np
import pandas as pd

from nkpotts.theory_exact import ht_closed
from nkpotts.mc_corrected import measure_grid


def adaptive_axis(L, npts=41, half_width_coeff=8.0, center=None):
    """Field window with width ~ half_width_coeff / L.

    Centred on `center` when given, otherwise on h_t.  Centring on h_t is only
    safe once the finite-size peak shift is small compared with the window:
    at L = 400 the peak sits ~0.02 above h_t, i.e. at the very edge of an
    8/L window, and the returned chi_max is then truncated.  Pass the peak
    location from the coarse scan for the smaller sizes.
    """
    c = ht_closed(1.0) if center is None else center
    w = half_width_coeff / L
    return np.linspace(c - w, c + w, npts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[3200, 6400])
    ap.add_argument("--npts", type=int, default=41)
    ap.add_argument("--burn", type=int, default=20000)
    ap.add_argument("--nmeas", type=int, default=40000)
    ap.add_argument("--freq", type=int, default=10)
    ap.add_argument("--center", type=float, default=None,
                    help="window centre; defaults to h_t")
    ap.add_argument("--half-width-coeff", type=float, default=8.0)
    ap.add_argument("--out", default="data/data_chi_peak_refined.csv")
    args = ap.parse_args()

    ht = ht_closed(1.0)
    all_rows = []
    for L in args.L:
        h_axis = adaptive_axis(L, args.npts, args.half_width_coeff,
                               args.center)
        dh = h_axis[1] - h_axis[0]
        print(f"L={L}: {args.npts} points, dh={dh:.2e} "
              f"(uniform grid was 7.5e-03, expected peak width ~{6.0/L:.2e})")
        rows = measure_grid(L, h_axis, burn=args.burn, nmeas=args.nmeas,
                            freq=args.freq)
        all_rows.extend(rows)

        chi = np.array([r["chi"] for r in rows])
        hpk = h_axis[int(np.argmax(chi))]
        n_half = int((chi >= chi.max() / 2).sum())
        print(f"  chi_max = {chi.max():.2f}   chi_max/L = {chi.max()/L:.5f} "
              "(asymptotic target 1/16 = 0.0625)")
        print(f"  peak at h = {hpk:.5f}  (exact h_t = {ht:.5f}), "
              f"{n_half} points above half-max")
        if n_half < 5:
            print("  WARNING: peak still under-resolved; increase --npts or "
                  "reduce half_width_coeff.")

    pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
