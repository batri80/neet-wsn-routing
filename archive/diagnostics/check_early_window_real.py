"""
Same early-window question as before, but using the REAL simulate.py loop
exclusively (record_series=True), not a hand-rolled reimplementation --
removes any risk of the diagnostic script itself diverging from the
actual simulation.
"""
from simulate import run_episode
import numpy as np

for WINDOW in [100, 300, 600, 1000]:
    print(f"--- window: rounds 0-{WINDOW} ---")
    for strat in ['random', 'eqopt']:
        psin_vals = []
        for seed in range(10):
            res = run_episode(N=60, E0=1.0, strategy=strat, max_rounds=1500,
                               seed=seed, record_series=True)
            psin = res['series']['Psi_N']
            n = min(WINDOW, len(psin))
            if n > 0:
                psin_vals.append(psin[:n].mean())
        print(f"  {strat:8s}: mean Psi_N over window = {np.mean(psin_vals):.4f}  "
              f"(n_episodes_with_enough_rounds={len(psin_vals)}/10)")
