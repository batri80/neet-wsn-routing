"""
Does K=40 restore HND advantage over static/mincost/greedy at N=100,
where one-step EqOpt was shown to lose to all three (Experiment A,
Section 14.8 of the model reference doc)?
"""
from simulate import run_episode
from scipy import stats
import numpy as np
import time

N, E0, R_c = 100, 1.0, 35.0
N_SEEDS = 20
MAX_ROUNDS = 3000

results = {}
t0 = time.time()
for strat, kwargs in [('static', {}), ('random', {}), ('greedy', {}),
                       ('mincost', {}), ('eqopt', {}), ('eqopt_k', {'K': 40})]:
    hnds = []
    for seed in range(N_SEEDS):
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy=strat, max_rounds=MAX_ROUNDS,
                           seed=seed, record_series=False, **kwargs)
        hnds.append(res['HND'])
    results[strat] = np.array(hnds)
    print(f"{strat:10s}: mean HND={np.mean(hnds):.1f}  std={np.std(hnds):.1f}  "
          f"(elapsed so far: {time.time()-t0:.0f}s)")

print("\n=== eqopt_k vs each baseline ===")
ek = results['eqopt_k']
for strat in ['static', 'random', 'greedy', 'mincost', 'eqopt']:
    other = results[strat]
    u, p = stats.mannwhitneyu(ek, other, alternative='two-sided')
    pooled_std = np.sqrt((ek.var(ddof=1) + other.var(ddof=1)) / 2)
    d = (ek.mean() - other.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
    result = 'eqopt_k WINS' if ek.mean() > other.mean() else 'eqopt_k loses'
    sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
    print(f"eqopt_k vs {strat:10s}: {result}  p={p:.4f}  d={d:.3f}  {sig}")
