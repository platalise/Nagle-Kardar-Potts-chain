# nkpotts

**The Nagle–Kardar–Potts chain for general q: exact transition line, spinodals, and order-parameter jump.**

[![tests](https://github.com/OWNER/nkpotts/actions/workflows/tests.yml/badge.svg)](https://github.com/OWNER/nkpotts/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)

Exact general-$q$ solution and Monte Carlo for the one-dimensional $q$-state
Potts chain with an added all-to-all ("herding") coupling — the Potts analogue of
the Nagle–Kardar model. Companion code for the paper:

> *The Nagle–Kardar–Potts chain for general q: exact transition line, spinodals, and order-parameter jump*, submitted to J. Stat. Mech. arXiv:XXXX.XXXXX.

## Scientific positioning

The exact free energy of this model was derived by **Campa *et al.*, *J. Stat.
Mech.* (2026) 023212** ([arXiv:2512.02930](https://arxiv.org/abs/2512.02930)) via
the spin-1 Blume–Emery–Griffiths mapping + Hubbard–Stratonovich + transfer
matrix, who solved $q=3$ and posed the general-$q$ extension as an open question.
This repository reproduces their free energy and answers that question:

1. **Closed forms for general $q$**, obtained directly from the occupation
   fractions with no spin mapping: transition line
   $\beta h_t = \frac{2(q-1)}{q-2}\ln[1+(q-2)e^{-\beta J}]$, reflection symmetry
   $f(\phi)=f(\phi_t-\phi)$ with jump $\phi_t=(q-2)/(q-1)$, and the paramagnetic
   spinodal $\beta h_{sp}^{+} = q^2/(2e^{\beta J}+q-2)$ (the ordered spinodal is
   solved numerically). At $q=3$ the transition line reduces exactly to the
   analytic line of Campa *et al.*; the reflection symmetry at $q=3$ is theirs
   (their Sec. 3.1) and the general-$q$ extension is ours.
2. **The Ising limit as an independent check.** As $q \to 2$ the transition line
   and the paramagnetic spinodal both degenerate onto $\beta h = 2e^{-\beta J}$,
   the exact critical line of the Ising Nagle–Kardar chain, and the jump closes
   continuously.
3. **Monte Carlo up to $L = 6400$** with corrected collective-update energetics
   (a factor-of-two error in the herding energy silently displaces the
   transition). Coexistence, hysteresis, the Binder dip and a near-extensive
   susceptibility all appear. Note that the exact barrier is small
   ($\beta\Delta f\,L \lesssim 4.5$ at these sizes), so the asymptotic
   first-order amplitude $\beta(\Delta\phi)^2/4 = 1/16$ is **not** attained —
   measured $\chi_{max}/L \approx 0.034$–0.044. The order of the transition is
   established analytically, not numerically.

The power-law family and the information-theoretic inference bound previously
explored here are **not** part of this paper and are deferred to separate work;
their code remains in the repository but is unused by the manuscript figures.

## Install

```bash
git clone https://github.com/OWNER/nkpotts.git
cd nkpotts
pip install -e .           # core: theory + MC + figures
pip install -e ".[ml]"     # + torch + torch_geometric for GNN experiments
```

## 60-second demo

```python
from nkpotts.theory_exact import ht_closed, h_spinodal_closed

# Exact transition and paramagnetic spinodal (q=3, J=1, T=1)
ht  = ht_closed(J=1.0)                 # 1.2530467500626585
hsp = h_spinodal_closed(J=1.0)         # 1.3982616314726728
print(f"h_t = {ht:.6f},  h_sp+ = {hsp:.6f}")
```

Simulate 3-state Potts + herding at the transition and locate the susceptibility peak:

```python
import numpy as np
from nkpotts.mc_corrected import measure_grid

h_axis = np.linspace(1.15, 1.35, 11)
rows = measure_grid(L=400, h_axis=h_axis, burn=5000, nmeas=8000, freq=5)
chi = np.array([r["chi"] for r in rows])
print("chi-peak at h =", h_axis[chi.argmax()], "(exact h_t = 1.253)")
```

## Reproduce the paper

Every figure is reproducible from data files via a single script:

```bash
# 1. Generate all data (heavy; ~few hours on a modern node + GPU for GNN)
bash scripts/run_fullscale.sh

# 2. Rebuild every figure from the exported CSV/NPZ (fast, no MC/torch)
python scripts/replot_all.py
```

Individual steps:

```bash
python -m nkpotts.theory_exact                              # exact theory + landscape + phase diagram
python scripts/run_production_mc.py --mode all-full         # observables, Binder, histograms, hysteresis
python scripts/run_powerlaw.py --mode full                  # power-law tricritical sweep
python -m nkpotts.gnn_corrected --seeds 5 --samples 4000    # GNN vs MLP vs Ridge (needs GPU)
python -m nkpotts.info_bound --L 400 --samples 4000         # Bayes-optimal ceiling + Cramér–Rao floor
```

## Figures at a glance

The `figures/` directory contains all paper figures as PDFs. A few highlights:

| | |
|---|---|
| **Fig. 1** exact coexistence landscape | **Fig. 2** $(J, h)$ phase diagram with spinodals |
| **Fig. 3** magnetization, susceptibility, heat capacity | **Fig. 4** $\chi_{\max} \sim L^{0.88}$ (first-order volume law) |
| **Fig. 5** $P(\phi)$ bimodality at coexistence | **Fig. 6** hysteresis loops widen with $L$ |
| **Fig. 8** power-law tunable transition | **Fig. 9** Bayes-optimal ceiling & Cramér–Rao floor |

## Repository layout

```
nkpotts/                  Python package
├── theory_exact.py       Hubbard-Stratonovich + transfer-matrix free energy
├── mc_corrected.py       O(1) Metropolis MC (correct dE_glob)
├── powerlaw_theory.py    Kac normalization + long-range regime taxonomy
├── powerlaw_mc.py        Kac-normalized power-law MC
├── gnn_corrected.py      GNN vs MLP vs Ridge inverse regression
├── info_bound.py         Bayes-optimal ceiling + Cramér–Rao floor
└── verify_bug.py         Brute-force proof of the factor-2 pitfall

scripts/                  Runnable pipelines
├── run_production_mc.py  observables, Binder, histograms, hysteresis, χ-scaling
├── run_powerlaw.py       power-law tricritical sweep
└── replot_all.py         rebuild every figure from data files

tests/                    pytest suite (exact theory, bug proof, Kac norm)
data/                     exported CSV/NPZ from the paper runs
figures/                  paper figures (PDF)
paper/                    manuscript.tex, manuscript.pdf, cover_letter
```

## The factor-2 pitfall (a practical note)

A common way to code the collective-update energy change for
$H = -(h/N)\sum_{i<j}\delta_{s_i,s_j}$ silently uses

```python
dE_glob = -(h/L) * (2*counts[new] + 1) + (h/L) * (2*counts[old] - 1)   # WRONG
```

which is *twice* the correct value: it effectively simulates $h_{\text{eff}} = 2h$
and misplaces the transition. The correct expression is

```python
dE_glob = -(h/L) * (counts[new] - counts[old] + 1)                     # RIGHT
```

`nkpotts.verify_bug` and `tests/test_theory.py::test_factor_two_bug_brute_force`
prove this against brute-force total-energy differences.

## Citation

If this code or its results are useful to you, please cite both the paper and
the exact solution it builds on:

```bibtex
@article{yourname2026nkpotts,
  title  = {The Nagle–Kardar–Potts chain for general q: exact transition line, spinodals, and order-parameter jump},
  author = {Author Name},
  year   = {2026},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  note = {Submitted to SciPost Physics Core}
}

@article{campa2026,
  title  = {Phase diagram of the one-dimensional three-state Potts model
            with an additional mean-field interaction},
  author = {Campa, Alessandro and Hovhannisyan, Vahan and Ruffo, Stefano and Trombettoni, Andrea},
  journal = {J. Stat. Mech.},
  year   = {2026},
  pages  = {023212},
  eprint = {2512.02930},
  archivePrefix = {arXiv}
}
```

GitHub's "Cite this repository" button reads `CITATION.cff` for a formatted
citation.

## License

Code: [MIT](LICENSE). Paper: CC BY 4.0 upon publication in SciPost Physics Core.

## Acknowledgments

We thank the authors of Campa *et al.* [arXiv:2512.02930] for the exact solution
this work builds on.
