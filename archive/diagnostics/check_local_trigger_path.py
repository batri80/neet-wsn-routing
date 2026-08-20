"""
Check whether local-trigger reattachment for eqopt_k is actually using
K-step scoring, or silently falling back to one-step (or something else)
-- this could explain why isolated global-trigger decisions are correct
but full-episode outcomes are worse, if most of the actual decision
volume happens through an unfixed local path.
"""
from simulate import run_episode
import numpy as np

for strat, kwargs in [('eqopt', {}), ('eqopt_k', {'K': 10})]:
    rows = [run_episode(N=100, E0=1.0, strategy=strat, max_rounds=362, seed=s,
                         record_series=False, **kwargs) for s in range(5)]
    n_local = np.mean([r['n_local'] for r in rows])
    n_global = np.mean([r['n_global'] for r in rows])
    print(f"{strat:10s}: mean n_local={n_local:.1f}  mean n_global={n_global:.1f}")
