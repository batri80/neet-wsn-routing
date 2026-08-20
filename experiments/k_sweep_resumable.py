"""
Resumable K sweep: saves each K's raw results to disk immediately after
completion. Safe to interrupt and rerun -- already-saved K values are
skipped, so nothing is ever recomputed unnecessarily.
"""
from simulate import run_episode
from scipy import stats
import numpy as np
import time
import os

WINDOW = 362
N, E0, R_c = 100, 1.0, 35.0
N_SEEDS = 50
K_VALUES = [10, 20, 30, 40, 50]
OUTDIR = 'k_sweep_results'

for K in K_VALUES:
    fpath = f'{OUTDIR}/K{K}.npy'
    if os.path.exists(fpath):
        vals = np.load(fpath)
        print(f"K={K:3d}: [already saved] mean={vals.mean():.5f}  std={vals.std():.5f}  n={len(vals)}")
        continue
    t0 = time.time()
    vals = []
    for seed in range(N_SEEDS):
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy='eqopt_k', K=K,
                           max_rounds=WINDOW, seed=seed, record_series=True)
        psi = res['series']['Psi']
        if len(psi) >= WINDOW:
            vals.append(psi[:WINDOW].max())
    vals = np.array(vals)
    np.save(fpath, vals)
    elapsed = time.time() - t0
    print(f"K={K:3d}: [computed+saved] mean={vals.mean():.5f}  std={vals.std():.5f}  "
          f"n={len(vals)}/{N_SEEDS}  time={elapsed:.1f}s")

print("\n=== All K values loaded. Running full analysis. ===\n")
results = {K: np.load(f'{OUTDIR}/K{K}.npy') for K in K_VALUES}

print("=== Pairwise significance ===")
for i in range(len(K_VALUES)):
    for j in range(i+1, len(K_VALUES)):
        a, b = results[K_VALUES[i]], results[K_VALUES[j]]
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
        print(f"K={K_VALUES[i]} vs K={K_VALUES[j]}: p={p:.4f}  d={d:.3f}  {sig}")

print("\n=== Trend check: Spearman correlation of K vs Psi_max (pooled) ===")
all_K, all_psi = [], []
for K, vals in results.items():
    all_K += [K]*len(vals)
    all_psi += list(vals)
rho, p_trend = stats.spearmanr(all_K, all_psi)
print(f"rho={rho:.3f}  p={p_trend:.5f}  "
      f"{'SIGNIFICANT downward trend' if p_trend<0.05 and rho<0 else 'no significant trend'}")
