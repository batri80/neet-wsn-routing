"""
Proper statistical resolution of whether K=20/30/40 differ meaningfully
or are indistinguishable given noise -- more seeds, real significance
tests, not just point-estimate comparison.
"""
from simulate import run_episode
from scipy import stats
import numpy as np
import time

WINDOW = 362
N, E0, R_c = 100, 1.0, 35.0
N_SEEDS = 25

results = {}
for K in [10, 20, 30, 40]:
    t0 = time.time()
    vals = []
    for seed in range(N_SEEDS):
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy='eqopt_k', K=K,
                           max_rounds=WINDOW, seed=seed, record_series=True)
        psi = res['series']['Psi']
        if len(psi) >= WINDOW:
            vals.append(psi[:WINDOW].max())
    results[K] = np.array(vals)
    elapsed = time.time() - t0
    print(f"K={K:3d}: mean={np.mean(vals):.5f}  std={np.std(vals):.5f}  "
          f"n={len(vals)}/{N_SEEDS}  time={elapsed:.1f}s")

print("\n=== Pairwise significance ===")
Ks = [10, 20, 30, 40]
for i in range(len(Ks)):
    for j in range(i+1, len(Ks)):
        a, b = results[Ks[i]], results[Ks[j]]
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
        print(f"K={Ks[i]} vs K={Ks[j]}: p={p:.4f}  d={d:.3f}  {sig}")
