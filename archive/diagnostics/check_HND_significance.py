"""
Proper per-seed HND comparison: eqopt_k (K=40, cooldown-stabilized) vs.
static, mincost, and one-step eqopt -- the three baselines that matter
most (static/mincost are what one-step EqOpt specifically lost to;
one-step eqopt is the direct ablation comparison).
"""
from simulate import run_episode
from scipy import stats
import numpy as np
import time

N, E0, R_c = 100, 1.0, 35.0
MAX_ROUNDS = 3000
N_SEEDS = 20

eqopt_k_hnd = [183,264,280,268,177,223,221,110,418,332,147,187,186,322,213,259,239,507,342,274]

results = {'eqopt_k': np.array(eqopt_k_hnd)}
for strat in ['static', 'mincost', 'eqopt']:
    t0 = time.time()
    hnds = []
    for seed in range(N_SEEDS):
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy=strat, max_rounds=MAX_ROUNDS,
                           seed=seed, record_series=False)
        hnds.append(res['HND'])
    results[strat] = np.array(hnds)
    print(f"{strat:10s}: mean={np.mean(hnds):.1f}  std={np.std(hnds):.1f}  time={time.time()-t0:.0f}s")

print("\n=== eqopt_k vs each baseline (Mann-Whitney + Cohen's d) ===")
ek = results['eqopt_k']
for strat in ['static', 'mincost', 'eqopt']:
    other = results[strat]
    u, p = stats.mannwhitneyu(ek, other, alternative='two-sided')
    pooled_std = np.sqrt((ek.var(ddof=1) + other.var(ddof=1)) / 2)
    d = (ek.mean() - other.mean()) / pooled_std if pooled_std > 1e-12 else float('nan')
    result = 'eqopt_k WINS' if ek.mean() > other.mean() else 'eqopt_k loses'
    sig = 'SIGNIFICANT' if p < 0.05 else 'not significant'
    print(f"eqopt_k vs {strat:10s}: {result}  p={p:.4f}  d={d:.3f}  {sig}")
