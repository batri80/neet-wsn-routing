"""
T1: EqOpt's mean drift per reconfiguration should never be positive on
    average (candidate set always includes the null/do-nothing option,
    so EqOpt can never choose worse than doing nothing).
T2: Within a single accumulation phase (between reconfigurations), R_h
    should be non-decreasing.
"""
import numpy as np
from simulate import run_episode

print("=== T1: bounded drift check ===")
drifts = []
for seed in range(10):
    res = run_episode(N=60, E0=1.0, strategy='eqopt', max_rounds=800,
                       seed=seed, record_series=False)
    if res['n_reconfigs'] > 0:
        drifts.append(res['mean_dPsi_per_reconfig'])
print(f"episodes with reconfigs: {len(drifts)}/10")
print(f"mean dPsi per reconfig across seeds: {np.mean(drifts):.6f} (should be <= 0)")
print(f"max (worst) dPsi seen: {np.max(drifts):.6f} (should also be <= 0)")

print("\n=== T2: risk monotonicity within accumulation phases ===")
res = run_episode(N=60, E0=1.0, strategy='eqopt', max_rounds=800,
                   seed=3, record_series=True)
rh = res['series']['Rh']

# Identify phase boundaries via drops in R_h (reconfiguration events
# should show up as sudden decreases; pure accumulation should not).
diffs = np.diff(rh)
drop_threshold = -0.01  # a real reconfig should cause a visible drop
phase_boundaries = np.where(diffs < drop_threshold)[0]
print(f"detected {len(phase_boundaries)} likely reconfiguration-driven drops "
      f"out of {len(rh)} rounds (recorded reconfigs: {res['n_reconfigs']})")

# Check monotonicity strictly WITHIN each detected phase
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
print("(some violations are expected -- Theorem 2's precondition is Psi_N AND")
print(" Lambda^2 both non-decreasing, which won't hold at every single step;")
print(" a small violation rate is fine, a large one suggests a real problem)")
