"""
Increase sample size at existing K values (10, 20, 30, 40) to determine
whether the declining-Psi_max trend is real, before spending compute
extending the range to K=50/60.
"""
from simulate import run_episode
from scipy import stats
import numpy as np
import time

WINDOW = 362
N, E0, R_c = 100, 1.0, 35.0
N_SEEDS = 50  # doubled from the previous 25

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

print("\n=== Pairwise significance (n=50 per group) ===")
Ks = [10, 20, 30, 40]
for i in range(len(Ks)):
    for j in range(i+1, len(Ks)):
        a, b = results[Ks[i]], results[Ks[j]]
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (a.mean() - b.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
        print(f"K={Ks[i]} vs K={Ks[j]}: p={p:.4f}  d={d:.3f}  {sig}")

print("\n=== Trend check: Spearman correlation of K vs Psi_max (pooled) ===")
all_K, all_psi = [], []
for K, vals in results.items():
    all_K += [K]*len(vals)
    all_psi += list(vals)
rho, p_trend = stats.spearmanr(all_K, all_psi)
print(f"rho={rho:.3f}  p={p_trend:.5f}  "
      f"{'SIGNIFICANT downward trend' if p_trend<0.05 and rho<0 else 'no significant trend'}")
