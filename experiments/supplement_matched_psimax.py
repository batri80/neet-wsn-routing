"""
Supplementary matched-window Psi_max, ALL five N values (censoring turned
out to be pervasive, not confined to large N -- see exp_A censor-rate
table). Window at each N is anchored to random's natural LND, since
random is the only strategy that reliably completes without hitting the
round cap.
"""
import numpy as np
import pandas as pd
from simulate import run_episode

NS = [30, 60, 100, 200, 500]
STRATEGIES = ['static', 'random', 'greedy', 'mincost', 'eqopt']
N_RUNS = 15
E0 = 1.0
R_c = 35.0

results = []
for N in NS:
    print(f"--- N={N} ---")
    random_lnds = []
    for seed in range(N_RUNS):
        res = run_episode(N=N, E0=E0, R_c=R_c, strategy='random', max_rounds=3000,
                           seed=seed, record_series=False)
        random_lnds.append(res['LND'])
    window = int(min(random_lnds) * 0.9)
    print(f"  window = {window} rounds (90% of shortest random LND={min(random_lnds)})")

    for strat in STRATEGIES:
        vals = []
        for seed in range(N_RUNS):
            res = run_episode(N=N, E0=E0, R_c=R_c, strategy=strat, max_rounds=window,
                               seed=seed, record_series=True)
            psi = res['series']['Psi']
            if len(psi) >= window:
                vals.append(psi[:window].max())
        mean_val = np.mean(vals) if vals else float('nan')
        results.append(dict(N=N, strategy=strat, window=window,
                             matched_Psi_max=mean_val, n_reached_window=len(vals)))
        print(f"  {strat:8s}: matched_Psi_max={mean_val:.5f}  (n={len(vals)}/{N_RUNS})")

pd.DataFrame(results).to_csv('results/exp_A_matched_psimax_supplement.csv', index=False)
print("\nsaved to results/exp_A_matched_psimax_supplement.csv")
