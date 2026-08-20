import numpy as np
from simulate import run_episode

res = run_episode(N=60, E0=1.0, strategy='eqopt', max_rounds=800,
                   seed=3, record_series=True)
rh = res['series']['Rh']

diffs = np.diff(rh)
drop_threshold = -0.01
phase_boundaries = np.where(diffs < drop_threshold)[0]
print(f"detected {len(phase_boundaries)} likely reconfiguration-driven drops "
      f"out of {len(rh)} rounds (recorded reconfigs: {res['n_reconfigs']})")

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
print(f"within-phase monotonicity violations: {violations}/{total_checked} steps "
      f"({100*violations/max(total_checked,1):.2f}%)")
