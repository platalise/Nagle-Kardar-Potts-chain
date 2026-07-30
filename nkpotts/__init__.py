"""nkpotts: exact solution, Monte Carlo, and inference for the
Nagle-Kardar-Potts chain (Sec. 2 of the accompanying paper).

Quick start
-----------
>>> from nkpotts.theory_exact import ht_closed, h_spinodal_closed
>>> ht_closed(J=1.0)          # exact first-order transition line (q=3, T=1)
1.2530467500626585

Modules
-------
theory_exact     : Hubbard-Stratonovich + transfer-matrix free energy,
                   closed-form h_t and spinodals, reflection symmetry.
mc_corrected     : O(1) Metropolis MC with correct collective-update energetics.
powerlaw_theory  : Kac normalization and long-range regime taxonomy.
powerlaw_mc      : Kac-normalized power-law MC.
gnn_corrected    : GNN vs global-MLP vs domain-wall Ridge inverse regression.
info_bound       : Fisher-Neyman sufficient statistic, Bayes-optimal ceiling,
                   Cramer-Rao floor.
verify_bug       : brute-force check of the factor-2 pitfall in dE_glob.
"""
from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version("nkpotts")
except PackageNotFoundError:
    __version__ = "0.1.0"
