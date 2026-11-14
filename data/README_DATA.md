# Data files used by the revised manuscript (November 2026)

All Monte Carlo data are for q = 3, beta*J = 1 (k_B T = 1).
h_t = 1.2530467500728915 (closed form, Eq. (15) of the revised manuscript).

| File | Content | Produced by | Used for |
|---|---|---|---|
| `data_theory_landscape.csv` | beta[f(phi) - f(0)] at five fields | `nkpotts/theory_exact.py` | Fig. 1 |
| `data_phase_diagram.csv` | h_t, h_sp^+ (closed forms) and h_sp^- (continuation solution of f'=f''=0) vs beta*J, q = 3, 4, 5, 8 | `scripts/make_paper_figures.py` | Fig. 2 |
| `data_observables.csv` | uniform field scan, L = 400 ... 6400, 57 fields, 2e4 MCS burn-in + 4e4 MCS | `scripts/run_production_mc.py --mode grid` | Figs. 3, 4b (denominator), 7a |
| `longrun/L{L}_r{k}.npz` | long runs at h_t, 8 replicas per L; per sample: n_max, S = sum n_a(n_a-1), number of equal neighbour pairs | `scripts/run_reweighting.py run` | Figs. 4, 5, 7b |
| `data_chi_reweighted.csv` | per L: chi_max, peak field, U4 minimum, crossings, basin weights (jackknife over replicas) | `scripts/run_reweighting.py analyze` | Fig. 4, Sec. 4 numbers |
| `data_reweighted_curves.csv` | reweighted <phi>, chi, U4, C_V, p_ord on a continuous h axis | `scripts/run_reweighting.py analyze` | Fig. 7b |
| `data_histograms_longrun.csv` | P(phi) at h_t pooled over the 8 replicas | `scripts/run_reweighting.py analyze` | Fig. 5 |
| `longrun_meta.json` | crossings per replica, run parameters | `scripts/run_reweighting.py analyze` | Sec. 4 numbers |
| `data_hysteresis.csv` | up/down sweeps, L = 3200, 6400, 61 fields, 200 MCS per field, single run per size (seed 11+L) | `scripts/run_production_mc.py --mode hyst --Ls 3200 6400` | Fig. 6 |
| `data_chi_peak_refined_all.csv` | direct narrow-window scans near h_t (41-51 fields, 4e4 MCS each) | `scripts/refine_chi_peak.py` | SI Fig. S1 only (shows non-ergodicity at L >= 3200) |
| `master_form_checks.json` | all checks of Secs. 2.3-2.5 | `scripts/verify_master_form.py` | SI Tables S2-S4 |

## Revision history of the data (November 2026)

* `data_histograms.csv`, `data_histograms_raw.npz` (L = 400, 1600 only; short runs)
  and `data_hysteresis.csv` (L = 800, 3200) were not the data behind the
  L = 6400 curves of Figs. 5 and 6 of the submitted version; no file with
  L = 6400 histograms or hysteresis was found in the repository. They are kept
  in `superseded_2026-07/`. Fig. 6 is now built from new runs
  (`data_hysteresis.csv`, L = 3200 and 6400). Fig. 5 is now built from the long
  runs at h_t (`data_histograms_longrun.csv`).
* `nkpotts/mc_corrected.py::hysteresis()` previously seeded only NumPy's RNG,
  not Numba's, so hysteresis runs were not reproducible. It now seeds both.
* The submitted Sec. 4.1 quoted chi_max from direct narrow-window scans
  (`data_chi_peak_refined_all.csv`). At L >= 3200 these runs contain only a few
  barrier crossings each, so chi(h) jumps between neighbouring fields and its
  maximum is not an equilibrium estimate. The revised analysis uses the long
  runs + exact reweighting instead.
* `data_chi_scaling.csv` (L <= 3200, uniform grid) is superseded and kept in
  `superseded_2026-07/`.
