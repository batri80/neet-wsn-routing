import numpy as np
from simulate import run_episode

res = run_episode(N=60, E0=1.0, strategy='eqopt', max_rounds=800,
                   seed=3, record_series=True)
rh = res['series']['Rh']
diffs = np.diff(rh)

for thresh in [-0.01, -0.005, -0.001, -0.0001]:
    phase_boundaries = np.where(diffs < thresh)[0]
    violations = 0
    total_checked = 0
    start = 0
    for b in list(phase_boundaries) + [len(rh) - 1]:
        phase = rh[start:b+1]
        if len(phase) > 2:
            phase_diffs = np.diff(phase)
            violations += np.sum(phase_diffs < -1e-9)
            total_checked += len(phase_diffs)
        start = b + 1
    rate = 100 * violations / max(total_checked, 1)
    print(f"threshold={thresh:8.4f}: boundaries detected={len(phase_boundaries):4d} "
          f"(of {res['n_reconfigs']} recorded)  violation rate={rate:.2f}%")
