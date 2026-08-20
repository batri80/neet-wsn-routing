"""
Properly suppress BOTH trigger channels (Psi_c AND Rc_thresh) to get a
genuine near-zero-reconfiguration baseline, then re-sweep. The previous
sweep only varied Psi_c and left Rc_thresh=0.664 active throughout,
so R_h alone kept triggering ~23 reconfigs/episode even at Psi_c=999 --
never actually testing the "reconfiguration off" condition.
"""
from simulate import run_episode
import numpy as np

configs = [
    ("both active (realistic)",      0.065, 0.664),
    ("Psi_c relaxed only",           999,   0.664),
    ("Rc_thresh relaxed only",       0.065, 999),
    ("BOTH relaxed (true baseline)", 999,   999),
]

for label, psi_c, rc_thresh in configs:
    vals_eqopt, vals_static, reconfigs = [], [], []
    for seed in range(10):
        r_e = run_episode(N=100, E0=1.0, strategy='eqopt', Psi_c=psi_c, Rc_thresh=rc_thresh,
                           max_rounds=362, seed=seed, record_series=True)
        r_s = run_episode(N=100, E0=1.0, strategy='static',
                           max_rounds=362, seed=seed, record_series=True)
        pe, ps = r_e['series']['Psi'], r_s['series']['Psi']
        if len(pe) >= 362: vals_eqopt.append(pe[:362].max())
        if len(ps) >= 362: vals_static.append(ps[:362].max())
        reconfigs.append(r_e['n_reconfigs'])
    print(f"{label:30s}: mean_reconfigs={np.mean(reconfigs):6.1f}  "
          f"eqopt Psi_max={np.mean(vals_eqopt):.5f}  static Psi_max={np.mean(vals_static):.5f}  "
          f"{'eqopt WINS' if np.mean(vals_eqopt) < np.mean(vals_static) else 'eqopt loses'}")
