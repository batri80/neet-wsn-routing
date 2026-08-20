"""
Compare how often/when eqopt (one-step) vs eqopt_k actually trigger
global reconfiguration under the SAME (one-step-calibrated) thresholds --
checking whether eqopt_k is being mismanaged by stale calibration rather
than a flaw in K-step selection itself.
"""
from simulate import run_episode
import numpy as np

WINDOW = 362
for strat, kwargs in [('eqopt', {}), ('eqopt_k', {'K': 10})]:
    n_global_list, psi_at_trigger = [], []
    for seed in range(10):
        res = run_episode(N=100, E0=1.0, strategy=strat, max_rounds=WINDOW,
                           seed=seed, record_series=True, **kwargs)
        n_global_list.append(res['n_global'])
    print(f"{strat:10s}: mean n_global triggers = {np.mean(n_global_list):.1f}")
