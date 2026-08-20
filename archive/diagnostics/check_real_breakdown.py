"""
Pull local/global reconfig counts directly from simulate.py's own
results dict -- no hand-rolled reimplementation, no risk of the script
diverging from reality (as diagnose_reattachment_v2.py evidently did).
"""
from simulate import run_episode
import numpy as np

for psi_c in [0.065, 999]:
    n_local_list, n_global_list, n_total_list = [], [], []
    for seed in range(10):
        res = run_episode(N=100, E0=1.0, strategy='eqopt', Psi_c=psi_c,
                           max_rounds=362, seed=seed, record_series=False)
        n_local_list.append(res['n_local'])
        n_global_list.append(res['n_global'])
        n_total_list.append(res['n_reconfigs'])
    print(f"Psi_c={psi_c:6.3f}: mean n_local={np.mean(n_local_list):.1f}  "
          f"mean n_global={np.mean(n_global_list):.1f}  mean n_total={np.mean(n_total_list):.1f}")
