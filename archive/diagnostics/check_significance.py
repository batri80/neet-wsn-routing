"""
Quick significance check on the matched-window Psi_max comparison at
N=100 (the case that looked worst) before doing any further mechanistic
investigation -- a small mean gap may just be noise given the observed
per-strategy std (~0.010-0.018) from the full Experiment A run.
"""
import numpy as np
from scipy import stats
from simulate import run_episode

N, window = 100, 362  # matches the N=100 matched window from the supplement run
E0, R_c = 1.0, 35.0

vals = {}
for strat in ['static', 'random', 'greedy', 'mincost', 'eqopt']:
    v = []
    for seed in range(15):
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy=strat, max_rounds=window,
                           seed=seed, record_series=True)
        psi = res['series']['Psi']
        if len(psi) >= window:
            v.append(psi[:window].max())
    vals[strat] = np.array(v)
    print(f"{strat:8s}: mean={v and np.mean(v):.5f}  std={v and np.std(v):.5f}  n={len(v)}")

print()
for strat in ['random', 'greedy', 'mincost', 'static']:
    if len(vals['eqopt']) >= 2 and len(vals[strat]) >= 2:
        u, p = stats.mannwhitneyu(vals['eqopt'], vals[strat], alternative='two-sided')
        pooled_std = np.sqrt((vals['eqopt'].var(ddof=1) + vals[strat].var(ddof=1)) / 2)
        d = (vals['eqopt'].mean() - vals[strat].mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
        print(f"eqopt vs {strat:8s}: p={p:.4f}  cohens_d={d:.3f}  "
              f"{'SIGNIFICANT' if p < 0.05 else 'not significant'}")
