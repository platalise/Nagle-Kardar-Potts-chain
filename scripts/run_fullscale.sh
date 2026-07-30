#!/usr/bin/env bash
# =====================================================================
# run_fullscale.sh — publication-scale pipeline for nkpotts.
#
# Regenerates every data file and every figure in the paper. Total wall
# time depends on hardware; MC parts parallelize over L and h, GNN wants
# a GPU. Run from the repo root:
#
#   bash scripts/run_fullscale.sh
# =====================================================================
set -euo pipefail

echo "=== [0/6] Exact theory (seconds) ==="
python -m nkpotts.theory_exact
# -> theory_results.json, fig_theory_landscape.pdf, fig_phase_diagram.pdf,
#    phase_diagram_data.npz, data_theory_landscape.csv

echo "=== [1/6] Monte Carlo observables + Binder + chi-scaling ==="
python scripts/run_production_mc.py --mode grid \
    --Ls 400 800 1600 3200 6400 \
    --burn 20000 --nmeas 40000 --freq 10
# -> data_observables.csv, data_chi_scaling.csv,
#    fig_observables.pdf, fig_binder.pdf, fig_chi_scaling.pdf

echo "=== [2/6] Coexistence histograms (long fixed-h runs) ==="
python scripts/run_production_mc.py --mode hist \
    --Ls 400 1600 6400 \
    --burn 20000 --nmeas 120000
# -> data_histograms.csv, data_histograms_raw.npz, fig_histograms.pdf

echo "=== [3/6] Hysteresis loops ==="
python scripts/run_production_mc.py --mode hyst --Ls 800 3200
# -> data_hysteresis.csv, fig_hysteresis.pdf

echo "=== [4/6] Power-law herding family (tricritical sweep) ==="
python -m nkpotts.powerlaw_theory
python scripts/run_powerlaw.py --mode full
# -> data_powerlaw.csv, data_powerlaw_perL.csv, fig_powerlaw.pdf,
#    powerlaw_theory.json, powerlaw_summary.json

echo "=== [5/6] GNN inverse inference vs domain-wall control (GPU) ==="
python -m nkpotts.gnn_corrected --seeds 5 --samples 4000 --epochs 150 --L 400
# -> fig_gnn.pdf, gnn_summary.json, data_gnn_*.csv

echo "=== [6/6] Exact information bound (Bayes-optimal ceiling + Cramer-Rao) ==="
python -m nkpotts.info_bound --L 400 --samples 4000
# -> fig_infobound.pdf, info_bound.json, data_infobound_*.csv

echo ""
echo "All data + figures regenerated. To rebuild figures ONLY (fast, no MC/torch):"
echo "  python scripts/replot_all.py"
