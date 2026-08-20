"""
Same question, but only include episodes that actually SURVIVED to the
full window length -- excluding (not truncating) episodes that died
earlier, since silently averaging over a shorter trajectory biases
toward that strategy's easier, pre-death rounds only.
"""
from simulate import run_episode
import numpy as np

for WINDOW in [100, 300, 600, 1000]:
    print(f"--- window: rounds 0-{WINDOW} (survivors only) ---")
    for strat in ['random', 'eqopt']:
        psin_vals = []
        n_survived = 0
        for seed in range(15):
            res = run_episode(N=60, E0=1.0, strategy=strat, max_rounds=1500,
                               seed=seed, record_series=True)
            psin = res['series']['Psi_N']
            if len(psin) >= WINDOW:   # only count if it actually reached this far
                psin_vals.append(psin[:WINDOW].mean())
                n_survived += 1
        if psin_vals:
            print(f"  {strat:8s}: mean Psi_N = {np.mean(psin_vals):.4f}  "
                  f"(survivors to round {WINDOW}: {n_survived}/15)")
        else:
            print(f"  {strat:8s}: no episodes survived to round {WINDOW}")
